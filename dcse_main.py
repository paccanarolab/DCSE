import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from collections import OrderedDict
import gzip
import pandas as pd
import numpy as np

# Metrics (using scikit-learn for AUC calculations)
from sklearn.metrics import roc_auc_score, average_precision_score

# Default random seed and number of CSV rows read per chunk during preprocessing.
SEED = 50
CHUNK_SIZE = 500_000


def set_seed(seed=SEED):
    """Fix random seeds for reproducible shuffling and initialization."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for drug-pair / side-effect tuples.

    Each sample is a tuple of integer indices (left drug, right drug, side effect)
    and a binary label. Arrays are kept on CPU and transferred to the training
    device batch by batch.

    When flip_pairs is True, each stored row is used twice during training: once
    as (drug from stitch_id1, drug from stitch_id2) and once with those two drugs
    swapped. The side effect and label are unchanged in both cases.
    """

    def __init__(self, left_drug, right_drug, se, labels, flip_pairs=False):
        self.left_drug = left_drug
        self.right_drug = right_drug
        self.se = se
        self.labels = labels
        # When enabled, each row is also served with the drug order reversed.
        self.flip_pairs = flip_pairs
        self.n = len(labels)

    def __len__(self):
        if self.flip_pairs:
            # Each drug pair is trained in both orderings, doubling the number of samples.
            return 2 * self.n
        return self.n

    def __getitem__(self, idx):
        if self.flip_pairs and idx >= self.n:
            # Same row as index (idx - n), with left and right drug indices swapped.
            idx = idx - self.n
            return self.right_drug[idx], self.left_drug[idx], self.se[idx], self.labels[idx]
        return self.left_drug[idx], self.right_drug[idx], self.se[idx], self.labels[idx]


class RightSidePosBias(nn.Module):
    """
    DCSE model for predicting side-effect probability in a drug combination.

    The architecture learns embeddings for drugs and side effects, combines the
    two drugs through an MLP, and scores compatibility with each side-effect
    embedding. A separate bias pathway modulates predictions by drug pair and side effect.
    """

    def __init__(self, num_drugs, num_ses, embedding_size=20, layers=[512, 256, 128], bias_layers=[4, 8, 16]):
        super().__init__()

        # Embedding layers
        # Learnable signatures for drugs and side effects.
        self.drug_embedding = nn.Embedding(num_embeddings=num_drugs, embedding_dim=embedding_size,
                                           padding_idx=0, sparse=False)
        self.se_embedding = nn.Embedding(num_embeddings=num_ses, embedding_dim=embedding_size,
                                         padding_idx=0, sparse=False)
        self.drug_bias = nn.Embedding(num_embeddings=num_drugs, embedding_dim=1, padding_idx=0)
        self.se_bias = nn.Embedding(num_embeddings=num_ses, embedding_dim=1, padding_idx=0)

        # Initialize drug and se embeddings from a uniform distribution
        nn.init.uniform_(self.drug_embedding.weight.data, a=0, b=0.01)
        nn.init.uniform_(self.se_embedding.weight.data, a=0, b=0.01)
        nn.init.uniform_(self.drug_bias.weight.data, a=0, b=0.01)
        nn.init.uniform_(self.se_bias.weight.data, a=0, b=0.01)

        # MLP over the two drug bias terms.
        self.bias_layers = nn.Sequential()
        self.bias_layers.add_module("bias_layer_0", nn.Linear(2, bias_layers[0], bias=True))
        self.bias_layers.add_module("relu_0", nn.ReLU())
        for i in range(1, len(bias_layers)):
            self.bias_layers.add_module(f"bias_layer_{i+1}", nn.Linear(bias_layers[i-1], bias_layers[i], bias=True))
            self.bias_layers.add_module(f"relu_{i+1}", nn.ReLU())

        # ReLU activation for comb_bias_layer
        self.comb_bias_layer = nn.Sequential(
            nn.Linear(bias_layers[-1], 1, bias=True),
            nn.ReLU()
        )

        # MLP over the concatenated drug embeddings.
        self.mlp_layers = nn.Sequential()
        self.mlp_layers.add_module("mlp_layer_0", nn.Linear(2*embedding_size, layers[0], bias=True))
        self.mlp_layers.add_module("relu_0", nn.ReLU())
        for i in range(1, len(layers)):
            self.mlp_layers.add_module(f"mlp_layer_{i+1}", nn.Linear(layers[i-1], layers[i], bias=True))
            self.mlp_layers.add_module(f"relu_{i+1}", nn.ReLU())

        # Last combination layer, separate, with ReLU activation
        self.last_comb_layer = nn.Sequential(
            nn.Linear(layers[-1], embedding_size, bias=True),
            nn.ReLU()
        )
        self.prediction = nn.Sigmoid()

    def forward(self, left_drug, right_drug, se):
        # Embeddings
        drug_left = F.relu(self.drug_embedding(left_drug))
        drug_right = F.relu(self.drug_embedding(right_drug))
        se_latent = F.relu(self.se_embedding(se))
        se_b = F.relu(self.se_bias(se))

        # Bias computations
        drug_bias_vector = torch.cat([F.relu(self.drug_bias(left_drug)), F.relu(self.drug_bias(right_drug))], dim=1)
        drug_bias_vector = self.bias_layers(drug_bias_vector)
        comb_bias = self.comb_bias_layer(drug_bias_vector)

        # MLP part
        mlp_vector = torch.cat([drug_left, drug_right], dim=1)
        mlp_vector = self.mlp_layers(mlp_vector)
        # Apply separate last combination layer with ReLU
        mlp_vector = self.last_comb_layer(mlp_vector)

        # Final prediction
        drug_se_combined_vector = torch.sum(mlp_vector * se_latent, dim=1, keepdim=True)
        prediction = self.prediction(drug_se_combined_vector - (comb_bias * se_b))
        return prediction


def add_to_mapping(mapping, values):
    """Assign consecutive integer indices to unseen drug or event identifiers."""
    for value in values:
        value = str(value)
        if value not in mapping:
            mapping[value] = len(mapping)


def build_mappings(training_data_path, chunksize=CHUNK_SIZE):
    """
    Build integer ID mappings from the training file.

    The training CSV is scanned in chunks so the full dataset does not need to
    fit in memory at once. Mappings are derived from training data only and are
    later applied to the test files.

    Returns:
        tuple: (stitch_id_map, event_id_map, num_drugs, num_events, training_rows)
    """
    print("Preprocessing data...")
    print("Creating drug ID mappings...")

    stitch_id_map = OrderedDict()
    training_rows = 0
    for chunk in pd.read_csv(training_data_path, usecols=['stitch_id1'], chunksize=chunksize):
        training_rows += len(chunk)
        add_to_mapping(stitch_id_map, chunk['stitch_id1'].values)
    for chunk in pd.read_csv(training_data_path, usecols=['stitch_id2'], chunksize=chunksize):
        add_to_mapping(stitch_id_map, chunk['stitch_id2'].values)

    print("Creating event ID mappings...")
    event_id_map = OrderedDict()
    for chunk in pd.read_csv(training_data_path, usecols=['event_umls_id'], chunksize=chunksize):
        add_to_mapping(event_id_map, chunk['event_umls_id'].values)

    num_drugs = len(stitch_id_map)
    num_events = len(event_id_map)
    print("Data preprocessing complete!")
    print(f"Number of unique drugs: {num_drugs}")
    print(f"Number of unique events: {num_events}")
    print(f"Training data rows: {training_rows}")

    return dict(stitch_id_map), dict(event_id_map), num_drugs, num_events, training_rows


def map_dataframe_to_arrays(df, stitch_id_map, event_id_map):
    """
    Convert raw identifiers in a DataFrame chunk to NumPy index arrays.

    Raises:
        ValueError: If any drug or event ID is missing from the training mappings.
    """
    stitch_ix1 = df['stitch_id1'].astype(str).map(stitch_id_map)
    stitch_ix2 = df['stitch_id2'].astype(str).map(stitch_id_map)
    event_ix = df['event_umls_id'].astype(str).map(event_id_map)

    if stitch_ix1.isna().any() or stitch_ix2.isna().any() or event_ix.isna().any():
        raise ValueError("Found a drug or event identifier that is not present in the training mappings.")

    return (
        stitch_ix1.to_numpy(dtype=np.int32, copy=True),
        stitch_ix2.to_numpy(dtype=np.int32, copy=True),
        event_ix.to_numpy(dtype=np.int32, copy=True),
        df['label'].to_numpy(dtype=np.float32, copy=True),
    )


def build_training_arrays(training_data_path, stitch_id_map, event_id_map, training_rows, chunksize=CHUNK_SIZE):
    """
    Materialize the indexed training set as compact NumPy arrays.

    Returns:
        tuple: (left_drug, right_drug, se, labels)
    """
    left_drug = np.empty(training_rows, dtype=np.int32)
    right_drug = np.empty(training_rows, dtype=np.int32)
    se = np.empty(training_rows, dtype=np.int32)
    labels = np.empty(training_rows, dtype=np.float32)

    # Fill pre-allocated arrays chunk by chunk
    usecols = ['stitch_id1', 'stitch_id2', 'event_umls_id', 'label']
    offset = 0
    total_chunks = int(np.ceil(training_rows / chunksize))

    for chunk in tqdm(
        pd.read_csv(training_data_path, usecols=usecols, chunksize=chunksize),
        total=total_chunks,
        desc='Index training data',
    ):
        left, right, event, batch_labels = map_dataframe_to_arrays(chunk, stitch_id_map, event_id_map)
        end = offset + len(batch_labels)
        left_drug[offset:end] = left
        right_drug[offset:end] = right
        se[offset:end] = event
        labels[offset:end] = batch_labels
        offset = end

    if offset != training_rows:
        raise ValueError(f"Expected {training_rows} training rows but found {offset}.")

    print(f"Training arrays created with {training_rows} samples")
    return left_drug, right_drug, se, labels


def create_dataloader(left_drug, right_drug, se, labels, batch_size, shuffle, flip_pairs=False,
                      seed=SEED, num_workers=4, pin_memory=False):
    """
    Wrap indexed arrays in a PyTorch Dataset and DataLoader.

    Returns:
        tuple: (dataset, dataloader)
    """
    # Create dataset and dataloader
    dataset = MyDataset(left_drug, right_drug, se, labels, flip_pairs=flip_pairs)
    generator = None
    # Fixed shuffle order when training with a set random seed
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    print(f"Dataset created with {len(dataset)} samples, {len(dataloader)} batches (batch_size={batch_size})")
    return dataset, dataloader


def _to_device(tensors, device):
    """Move a list of tensors to the target device."""
    return [t.to(device=device, non_blocking=True) for t in tensors]


def train_model(model, train_dataloader, num_epochs=5, learning_rate=1e-4, weight_decay=1e-6,
                metrics_every=0):
    """
    Train the model and report running loss.

    AUROC and AUPRC are computed periodically over the batches seen since the
    last metric report. This provides training feedback without evaluating the
    full dataset after every batch.

    Args:
        metrics_every: Report AUROC/AUPRC every this many batches. Set to 0 to disable.

    Returns:
        list: Per-epoch training history dictionaries.
    """
    # Setup optimizer and loss function
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    history = []

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        metric_preds = []
        metric_labels = []
        # Create progress bar for training
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}')

        for batch_idx, (left_drug, right_drug, se, labels) in progress_bar:
            left_drug, right_drug, se, labels = _to_device(
                [left_drug, right_drug, se, labels], device
            )

            optimizer.zero_grad()
            outputs = model(left_drug, right_drug, se)
            # Use float targets for BCELoss.
            target = labels.unsqueeze(1).float()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            # Update progress bar
            progress_bar.set_postfix({'Loss': loss.item()})

            metric_preds.extend(outputs.view(-1).detach().cpu().numpy())
            metric_labels.extend(labels.cpu().numpy())
            if metrics_every > 0 and (batch_idx + 1) % metrics_every == 0:
                # Calculate metrics over recent batches
                batch_auroc = roc_auc_score(metric_labels, metric_preds)
                batch_auprc = average_precision_score(metric_labels, metric_preds)
                print(f'  Batch {batch_idx + 1}: AUROC={batch_auroc:.4f}, AUPRC={batch_auprc:.4f}')
                metric_preds = []
                metric_labels = []

        # if metric_preds:
        #     batch_auroc = roc_auc_score(metric_labels, metric_preds)
        #     batch_auprc = average_precision_score(metric_labels, metric_preds)
        #     print(f'  Batch {len(train_dataloader)} (remainder): AUROC={batch_auroc:.4f}, AUPRC={batch_auprc:.4f}')

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}')
        # Save history
        history.append({'train_loss': epoch_loss})

    print("Training completed!")
    return history


def _count_csv_rows(path):
    """Count data rows in a CSV file (excluding the header)."""
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rt') as handle:
        return sum(1 for _ in handle) - 1


def evaluate_model_from_file(model, testing_data_path, stitch_id_map, event_id_map, dataset_name,
                             batch_size=8192, chunksize=CHUNK_SIZE, num_workers=4, pin_memory=False):
    """
    Evaluate a trained model on a test CSV file read in chunks.

    Returns:
        tuple: (auroc, auprc)
    """
    print(f"Evaluating model on {dataset_name}...")
    device = next(model.parameters()).device
    # Set the model to evaluation mode
    model.eval()

    total_rows = _count_csv_rows(testing_data_path)
    total_batches = (total_rows + batch_size - 1) // batch_size

    # Initialize lists to store the predictions and labels
    predictions = []
    labels = []
    usecols = ['stitch_id1', 'stitch_id2', 'event_umls_id', 'label']

    # Disable gradient calculation
    with torch.no_grad():
        progress_bar = tqdm(total=total_batches, desc=dataset_name)
        for chunk in pd.read_csv(testing_data_path, usecols=usecols, chunksize=chunksize):
            left, right, event, batch_labels = map_dataframe_to_arrays(chunk, stitch_id_map, event_id_map)
            chunk_dataset = MyDataset(left, right, event, batch_labels)
            chunk_dataloader = DataLoader(
                chunk_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
            )

            # Iterate over the test dataloader
            for left_drug, right_drug, se, batch_labels in chunk_dataloader:
                left_drug, right_drug, se, batch_labels = _to_device(
                    [left_drug, right_drug, se, batch_labels], device
                )
                outputs = model(left_drug, right_drug, se)
                # Append the predictions and labels to the respective lists
                predictions.append(outputs.view(-1).cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
                progress_bar.update(1)
        progress_bar.close()

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    # Calculate AUROC and AUPRC
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)
    print(f'{dataset_name} AUROC: {auroc:.4f}')
    print(f'{dataset_name} AUPRC: {auprc:.4f}')
    return auroc, auprc


def main():
    """
    Train DCSE on the prospective training set and evaluate on warm-start and
    cold-start test sets.
    """
    import argparse

    parser = argparse.ArgumentParser(description='DCSE Prospective DDI Prediction')
    parser.add_argument('--training_data', type=str, default='data/training_prosp.gz',
                        help='Path to training data file (default: data/training_prosp.gz)')
    parser.add_argument('--warm_start_testing_data', type=str, default='data/prospective_warm_start.gz',
                        help='Path to warm-start testing data file (default: data/prospective_warm_start.gz)')
    parser.add_argument('--cold_start_testing_data', type=str,
                        default='data/prospective_cold_start_new_pairs_same_drugs.gz',
                        help='Path to cold-start testing data file '
                             '(default: data/prospective_cold_start_new_pairs_same_drugs.gz)')
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='Training and evaluation batch size (default: 1024)')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='DataLoader worker processes (default: 6)')
    args = parser.parse_args()

    set_seed(SEED)
    print(f"Random seed set to {SEED}")

    # Set up data paths
    print(f"Training data: {args.training_data}")
    testing_paths = {
        'warm_start': args.warm_start_testing_data,
        'cold_start': args.cold_start_testing_data,
    }
    for name, path in testing_paths.items():
        print(f"{name} testing data: {path}")

    # Preprocess the data
    stitch_id_map, event_id_map, num_drugs, num_events, training_rows = build_mappings(args.training_data)

    # Set up device for PyTorch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # On macOS, PyTorch can use Apple GPUs through Metal Performance Shaders (MPS).
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Page-locked host memory lets the DataLoader copy batches to the GPU faster.
    pin_memory = device.type == 'cuda'
    print(
        f"Using device: {device} "
        f"(cuda_available={torch.cuda.is_available()}, mps_available={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()})"
    )

    train_left, train_right, train_se, train_labels = build_training_arrays(
        args.training_data, stitch_id_map, event_id_map, training_rows,
    )

    # Create PyTorch datasets and dataloaders
    train_dataset, train_dataloader = create_dataloader(
        train_left, train_right, train_se, train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        flip_pairs=True,  # Train on both (drug A, drug B) and (drug B, drug A) orderings.
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Model configuration
    signature_size = 512
    layers = [2 * signature_size, 2 * signature_size, 2 * signature_size]
    bias_layers = [2, 2]

    # Instantiate the model
    model = RightSidePosBias(
        num_drugs=num_drugs,
        num_ses=num_events,
        embedding_size=signature_size,
        layers=layers,
        bias_layers=bias_layers,
    )
    model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Ready for model training with {num_drugs} drugs and {num_events} events.")
    print(f"Training set size (with flipped pairs): {len(train_dataset)}")

    # Train the model
    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=5,
        learning_rate=1e-4,
        weight_decay=1e-6,
    )

    print("Training completed successfully!")

    # Evaluate the model on test data
    test_results = {}
    for name, path in testing_paths.items():
        print(f"\n--- Evaluation: {name} ---")
        auroc, auprc = evaluate_model_from_file(
            model, path, stitch_id_map, event_id_map, dataset_name=name,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=pin_memory,
        )
        test_results[name] = {'auroc': auroc, 'auprc': auprc}

    print("\n=== Evaluation summary ===")
    for name, metrics in test_results.items():
        print(f"{name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")


if __name__ == "__main__":
    main()
