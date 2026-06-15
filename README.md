# DCSE: Drug Combinations Side Effects Prediction

This repository contains the implementation of DCSE (Drug Combinations Side Effects), a novel machine learning method for predicting polypharmacy side effects. This code supports the paper:

**"Robust prediction of drug combination side effects in realistic settings"**

## Abstract

Side effects caused by drug combinations pose a major challenge in healthcare. Knowledge of these side effects is limited because often they are not detected in clinical trials, which typically involve a restricted number of participants and tested drug combinations. We introduce DCSE (Drug Combinations Side Effects), a novel machine learning method for predicting polypharmacy side effects. DCSE learns latent signatures for drugs, drug pairs, and side effects to predict the probability that a side effect occurs in a given drug combination. We first evaluate its performance in the commonly adopted experimental settings in the literature. However, these rely on balanced testing datasets and sampled negative examples, which do not capture the highly imbalanced and structured set of unknown side effects encountered in practice. Therefore, a key contribution of this paper is the introduction of more realistic experimental settings under prospective evaluations. Here, we attempt to predict side effects reported between 2009 and 2014 after training only on data available prior to that period. These evaluations include warm-start scenarios, in which some side effects are already known for a drug pair, and cold-start scenarios, in which the model predicts side effects for previously uncharacterized drug pairs. Our results indicate that DCSE consistently outperforms state-of-the-art methods, demonstrating its robustness and efficacy in real-world applications.


## Dataset

The datasets used in this study are available at: [https://doi.org/10.6084/m9.figshare.30355195](https://doi.org/10.6084/m9.figshare.30355195)

**Please download the datasets from the above link and place them in the `data/` directory before running the code.** Expected files:

- `data/training_prosp.gz`
- `data/prospective_warm_start.gz`
- `data/prospective_cold_start_new_pairs_same_drugs.gz`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/paccanarolab/DCSE
cd DCSE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

By default, the script trains once on the prospective training set and then evaluates on **both** prospective test sets (warm-start and cold-start), reporting AUROC and AUPRC for each.

```bash
python dcse_main.py
```

### Command Line Arguments

Use these only if your data files are in a different location:

- `--training_data`: Path to training data file (default: `data/training_prosp.gz`)
- `--warm_start_testing_data`: Path to warm-start test data (default: `data/prospective_warm_start.gz`)
- `--cold_start_testing_data`: Path to cold-start test data (default: `data/prospective_cold_start_new_pairs_same_drugs.gz`)

Example with custom paths:

```bash
python dcse_main.py \
  --training_data data/training_prosp.gz \
  --warm_start_testing_data data/prospective_warm_start.gz \
  --cold_start_testing_data data/prospective_cold_start_new_pairs_same_drugs.gz
```

## Data Format

The input data should be CSV files with the following columns:
- `stitch_id1`: First drug identifier
- `stitch_id2`: Second drug identifier  
- `event_umls_id`: Side effect identifier
- `label`: Binary label (0 or 1)

## Model Architecture

DCSE uses a neural network architecture that learns distributed representations for:
- Individual drugs
- Side effects
- Drug combinations (using an MLP)

The model predicts the probability of a side effect occurring in a given drug combination.

## Output

The script will output:
- Training progress with loss, AUC, and AUPR metrics
- Evaluation metrics (AUROC and AUPRC) for the warm-start and cold-start test sets

## Citation

This work was submitted to PLOS Computational Biology. Citation information will be updated upon publication.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- ruben.jf@pm.me
- alberto.paccanaro@fgv.br

