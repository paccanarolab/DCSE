# DCSE: Drug Combinations Side Effects Prediction

This repository contains the implementation of DCSE (Drug Combinations Side Effects), a novel machine learning method for predicting polypharmacy side effects. This code supports the paper:

**"Robust prediction of drug combination side effects in realistic settings"**

## Abstract

Side effects caused by drug combinations pose a major challenge in healthcare. Knowledge of these side effects is limited because often they are not detected in clinical trials, which typically involve a restricted number of participants and tested drug combinations. We introduce DCSE (Drug Combinations Side Effects), a novel machine learning method for predicting polypharmacy side effects. DCSE learns latent signatures for drugs, drug pairs, and side effects to predict the probability that a side effect occurs in a given drug combination. We first evaluate its performance in the commonly adopted experimental settings in the literature. Then, a key contribution of this paper is the introduction of more realistic experimental settings that incorporate warm-start and cold-start scenarios under a prospective evaluation. Here, we attempt to predict side effects reported between 2009 and 2014 after training only on data available prior to that period. Our results indicate that DCSE consistently outperforms state-of-the-art methods, demonstrating its robustness and efficacy in real-world applications.

## Dataset

The datasets used in this study are available at: [https://doi.org/10.6084/m9.figshare.30355195](https://doi.org/10.6084/m9.figshare.30355195)

**Please download the datasets from the above link and place them in the `data/` directory before running the code.**

**Note:** From the downloaded data:
- `training_prosp.gz` corresponds to `Prospective_warm_start`
- `testing_prosp.gz` corresponds to `Prospective_cold_start_new_pairs`

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

### Basic Training and Evaluation

```bash
python dcse_main.py --training_data data/training_prosp.gz --testing_data data/testing_prosp.gz
```

### Using Default Data Paths

```bash
python dcse_main.py
```

### Command Line Arguments

- `--training_data`: Path to training data file (default: `data/training_prosp.gz`)
- `--testing_data`: Path to testing data file (default: `data/testing_prosp.gz`)

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
- Final test set evaluation with AUROC and AUPRC scores

## Citation

This work was submitted to PLOS Computational Biology. Citation information will be updated upon publication.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- ruben.jf@pm.me
- alberto.paccanaro@fgv.br

