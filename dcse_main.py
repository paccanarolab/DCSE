import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Metrics (using scikit-learn for AUC calculations)
from sklearn.metrics import roc_auc_score, average_precision_score

import os

class MyDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for drug-drug interaction data."""
    
    def __init__(self, data):
        self.left_drug = data['left_drug_input']
        self.right_drug = data['right_drug_input']
        self.se = data['se_input']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.left_drug[idx], self.right_drug[idx], self.se[idx], self.labels[idx]

class RightSidePosBias(nn.Module):
    def __init__(self, num_drugs, num_ses, embedding_size=20, layers=[512, 256, 128], bias_layers=[4, 8, 16]):
        super().__init__()

        # Embedding layers
        self.drug_embedding = nn.Embedding(num_embeddings=num_drugs, embedding_dim=embedding_size,
                                           padding_idx=0, sparse=False)
        self.se_embedding = nn.Embedding(num_embeddings=num_ses, embedding_dim=embedding_size,
                                         padding_idx=0, sparse=False)
        self.drug_bias = nn.Embedding(num_embeddings=num_drugs, embedding_dim=1,
                                      padding_idx=0) 
        self.se_bias = nn.Embedding(num_embeddings=num_ses, embedding_dim=1,
                                      padding_idx=0) 
        
        # Initialize drug and se embeddings from a uniform distribution
        nn.init.uniform_(self.drug_embedding.weight.data, a=0, b=0.01)
        nn.init.uniform_(self.se_embedding.weight.data, a=0, b=0.01)
        nn.init.uniform_(self.drug_bias.weight.data, a=0, b=0.01)
        nn.init.uniform_(self.se_bias.weight.data, a=0, b=0.01)
        
        # Bias layers using a loop within nn.Sequential
        self.bias_layers = nn.Sequential()
        self.bias_layers.add_module("bias_layer_0", nn.Linear(2, bias_layers[0], bias=True))
        self.bias_layers.add_module("relu_0", nn.ReLU())
        for i in range(1, len(bias_layers)):
            self.bias_layers.add_module(f"bias_layer_{i+1}", nn.Linear(bias_layers[i-1], bias_layers[i], bias=True))
            self.bias_layers.add_module(f"relu_{i+1}", nn.ReLU())

        # ReLU activation for comb_bias_layer
        self.comb_bias_layer = nn.Sequential(
            nn.Linear(bias_layers[-1], 1, bias=True),
            nn.ReLU()  # ReLU activation added here
        )

        # MLP layers using a loop within nn.Sequential
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

def preprocess_data(training_data, testing_data):
    """
    Preprocess the training and testing data by creating mappings for drug IDs and event IDs.
    
    Args:
        training_data (pd.DataFrame): Training dataset
        testing_data (pd.DataFrame): Testing dataset
    
    Returns:
        tuple: (processed_training_data, processed_testing_data, num_drugs, num_events)
    """
    print("Preprocessing data...")
    
    # Create drug ID mappings
    print("Creating drug ID mappings...")
    stitch_ids = pd.Series(training_data['stitch_id1'].tolist() + training_data['stitch_id2'].tolist()).unique()
    stitch_id_map = {stitch_id: i for i, stitch_id in enumerate(stitch_ids)}
    
    # Apply drug ID mappings
    training_data = training_data.copy()
    testing_data = testing_data.copy()
    
    training_data['stitch_ix1'] = training_data['stitch_id1'].map(stitch_id_map)
    training_data['stitch_ix2'] = training_data['stitch_id2'].map(stitch_id_map)
    testing_data['stitch_ix1'] = testing_data['stitch_id1'].map(stitch_id_map)
    testing_data['stitch_ix2'] = testing_data['stitch_id2'].map(stitch_id_map)
    
    # Create event ID mappings
    print("Creating event ID mappings...")
    event_ids = training_data['event_umls_id'].unique()
    event_id_map = {event_id: i for i, event_id in enumerate(event_ids)}
    
    # Apply event ID mappings
    training_data['event_umls_ix'] = training_data['event_umls_id'].map(event_id_map)
    testing_data['event_umls_ix'] = testing_data['event_umls_id'].map(event_id_map)
    
    # Calculate dimensions
    num_drugs = len(stitch_ids)
    num_events = len(event_ids)
    
    print(f"Data preprocessing complete!")
    print(f"Number of unique drugs: {num_drugs}")
    print(f"Number of unique events: {num_events}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Testing data shape: {testing_data.shape}")
    
    return training_data, testing_data, num_drugs, num_events

def augment_training_data(train_set):
    """
    Augment training data by adding flipped drug pairs to ensure the model sees both (A,B) and (B,A) combinations.
    
    Args:
        train_set (pd.DataFrame): Training dataset with columns 'stitch_ix1', 'stitch_ix2', etc.
    
    Returns:
        pd.DataFrame: Augmented training dataset with flipped pairs
    """
    print("Augmenting training data with flipped drug pairs...")
    
    # Create flipped version by swapping stitch_ix1 and stitch_ix2
    flipped_train_set = train_set.copy()
    flipped_train_set['stitch_ix1'], flipped_train_set['stitch_ix2'] = flipped_train_set['stitch_ix2'], flipped_train_set['stitch_ix1']
    
    # Concatenate original and flipped data
    augmented_train_set = pd.concat([train_set, flipped_train_set], ignore_index=True)
    
    print(f"Original training set size: {len(train_set)}")
    print(f"Augmented training set size: {len(augmented_train_set)}")
    
    return augmented_train_set

def create_tensor_data(df, device):
    """
    Convert DataFrame to tensor data dictionary.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'stitch_ix1', 'stitch_ix2', 'event_umls_ix', 'label'
        device (torch.device): Device to move tensors to
    
    Returns:
        dict: Dictionary with tensor data
    """
    return {
        # Embedding layers require integer indices of type int64.
        'left_drug_input': torch.tensor(df['stitch_ix1'].astype('int64').values, dtype=torch.long).to(device),
        'right_drug_input': torch.tensor(df['stitch_ix2'].astype('int64').values, dtype=torch.long).to(device),
        'se_input': torch.tensor(df['event_umls_ix'].astype('int64').values, dtype=torch.long).to(device),
        # Use float targets for BCELoss.
        'labels': torch.tensor(df['label'].astype('float32').values, dtype=torch.float32).to(device),
    }

def create_dataset(df, device, batch_size=1024, shuffle=True):
    """
    Create PyTorch dataset and dataloader.
    
    Args:
        df (pd.DataFrame): Dataset DataFrame
        device (torch.device): Device to move tensors to
        batch_size (int): Batch size (default: 512 for modest hardware)
        shuffle (bool): Whether to shuffle the data (default: True)
    
    Returns:
        tuple: (dataset, dataloader)
    """
    print(f"Creating PyTorch dataset with batch size: {batch_size}, shuffle: {shuffle}")
    
    # Shuffle the dataset if requested
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    
    # Create tensor data
    data = create_tensor_data(df, device)
    
    # Create dataset and dataloader
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Dataloader will have {len(dataloader)} batches")
    
    return dataset, dataloader

def train_model(model, train_dataloader, num_epochs=3, learning_rate=1e-3, weight_decay=1e-6):
    """
    Train the DCSE model.
    
    Args:
        model: PyTorch model to train
        train_dataloader: Training data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
    
    Returns:
        list: Training history
    """
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Training variables
    history = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_auc = 0.0
        running_aupr = 0.0
        
        # Create progress bar for training
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}')
        
        for i, data in progress_bar:
            left_drug, right_drug, se, labels = data
            
            optimizer.zero_grad()
            outputs = model(left_drug, right_drug, se)
            target = labels.unsqueeze(1).float()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            
            # Calculate metrics for this batch
            batch_auc = roc_auc_score(labels.cpu().numpy(), outputs.view(-1).detach().cpu().numpy())
            batch_aupr = average_precision_score(labels.cpu().numpy(), outputs.view(-1).detach().cpu().numpy())
            running_auc += batch_auc * labels.size(0)
            running_aupr += batch_aupr * labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': loss.item(), 'AUC': batch_auc, 'AUPR': batch_aupr})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_auc = running_auc / len(train_dataloader.dataset)
        epoch_aupr = running_aupr / len(train_dataloader.dataset)
        
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Train AUC: {epoch_auc:.4f}, Train AUPR: {epoch_aupr:.4f}')
        
        # Save history
        history.append({
            'train_loss': epoch_loss,
            'train_auc': epoch_auc,
            'train_aupr': epoch_aupr
        })
    
    print("Training completed!")
    
    return history

def evaluate_model(model, test_dataloader):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PyTorch model
        test_dataloader: Test data loader
    
    Returns:
        tuple: (auroc, auprc) - Area Under ROC Curve and Area Under Precision-Recall Curve
    """
    print("Evaluating model on test data...")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store the predictions and labels
    predictions = []
    labels = []
    
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the test dataloader
        for data in test_dataloader:
            left_drug, right_drug, se, batch_labels = data
            outputs = model(left_drug, right_drug, se)
            
            # Append the predictions and labels to the respective lists
            predictions.extend(outputs.view(-1).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    # Calculate AUROC and AUPRC
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)
    
    # Print the results
    print(f'Test AUROC: {auroc:.4f}')
    print(f'Test AUPRC: {auprc:.4f}')
    
    return auroc, auprc

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DCSE Prospective DDI Prediction')
    parser.add_argument('--training_data', type=str, default='data/training_prosp.gz',
                        help='Path to training data file (default: data/training_prosp.gz)')
    parser.add_argument('--testing_data', type=str, default='data/testing_prosp.gz',
                        help='Path to testing data file (default: data/testing_prosp.gz)')
    
    args = parser.parse_args()
    
    
    # Load the data
    print(f"Loading training data from: {args.training_data}")
    training_prosp = pd.read_csv(args.training_data)
    print(f"Training data shape: {training_prosp.shape}")
    
    print(f"Loading testing data from: {args.testing_data}")
    testing_prosp = pd.read_csv(args.testing_data)
    print(f"Testing data shape: {testing_prosp.shape}")
    
    print("Data loaded successfully!")
    
    # Preprocess the data
    training_prosp, testing_prosp, num_drugs, num_events = preprocess_data(training_prosp, testing_prosp)
    
    # Set up data for algorithm (following notebook convention)
    train_set = training_prosp
    test_df_for_alg = testing_prosp
    
    # Augment training data with flipped drug pairs
    train_set = augment_training_data(train_set)
    
    # Set up device for PyTorch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # On macOS, PyTorch can use Apple GPUs through Metal Performance Shaders (MPS).
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device} "
        f"(cuda_available={torch.cuda.is_available()}, mps_available={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()})"
    )
    
    # Create PyTorch datasets and dataloaders
    train_dataset, train_dataloader = create_dataset(train_set, device, batch_size=512, shuffle=True)
    test_dataset, test_dataloader = create_dataset(test_df_for_alg, device, batch_size=512, shuffle=False)
    
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
        bias_layers=bias_layers
    )
    model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Ready for model training with {num_drugs} drugs and {num_events} events.")
    print(f"Training set shape: {train_set.shape}")
    print(f"Test set shape: {test_df_for_alg.shape}")
    
    # Train the model
    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=5,
        learning_rate=1e-4,
        weight_decay=1e-6
    )
    
    print("Training completed successfully!")
    
    # Evaluate the model on test data
    auroc, auprc = evaluate_model(model, test_dataloader)


if __name__ == "__main__":
    main()
