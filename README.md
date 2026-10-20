# Alzheimer's disease screening from EEG: Transfer Entropy, brain networks and machine learning

Pipeline for distinguishing Alzheimer's disease (AD) from healthy controls (CN) using **directed effective
connectivity (Transfer Entropy)** estimated from resting-state EEG, graph-theory features and classical machine
learning classifiers.

> **Research code only.** This project is not a medical device and must not be used for clinical diagnosis.

## Pipeline

1. Download the public EEG dataset ([OpenNeuro `ds004504`](https://openneuro.org/datasets/ds004504)) and keep AD and CN participants (36 AD + 29 CN).
2. Preprocess each recording: 1-40 Hz FIR band-pass filter, resampling to 200 Hz.
3. Discretize each channel into three states using its own standard deviation.
4. Estimate directed Transfer Entropy between every pair of the 19 electrodes (optional surrogate-based significance mask).
5. Extract network features from the TE graph and spectral features from the signal.
6. Univariate statistics: Mann-Whitney U with Benjamini-Hochberg FDR correction.
7. Compare six classifiers (elastic-net logistic regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM-RBF) with 5-fold stratified cross-validation repeated 10 times.
8. Interpretability (logistic regression coefficients, permutation importance) and average TE graphs over the 10-20 layout.

## Installation

```bash
git clone https://github.com/Caetanietal/eeg-alzheimer-transfer-entropy.git
cd eeg-alzheimer-transfer-entropy

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python eeg_alzheimer_transfer_entropy_ml.py
```

The first run downloads the dataset into `data/ds004504/` and computes the features (slow step, parallelized).
Features and TE matrices are then cached in `outputs/`, so later runs are fast. Set `RECOMPUTE_FEATURES = True`
at the top of the script to force recomputation.

Main settings (top of the script): `BANDPASS_HZ`, `TARGET_SFREQ`, `TE_HISTORY`, `N_SURROGATES`,
`APPLY_SURROGATE_MASK`, `RANDOM_STATE`, `N_JOBS`.

You can also run it in Google Colab or as a notebook (e.g. with `jupytext`), since the script is organized in cells.

## Outputs

Everything is written to `outputs/` (not versioned):

- `features.csv`, `te_matrices.npz`: extracted features and per-participant TE matrices
- `univariate_stats.csv`, `model_comparison.csv`, `lightgbm_permutation_importance.csv`
- `figures/`: boxplots of top features, ROC curves and confusion matrices, logistic regression coefficients, permutation importance, mean TE graphs

## Data

The EEG data are **not** included in this repository. They are downloaded automatically from OpenNeuro
(`ds004504`), please follow the dataset's own license and citation requirements.

## References

- Miltiadous, A. et al. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. *Data*, 8(6), 95.
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
- Rubinov, M. & Sporns, O. (2010). Complex network measures of brain connectivity. *NeuroImage*, 52(3), 1059-1069.

## Author

Pedro Santos Rosa Muradas, Federal University of ABC (UFABC).

## License

MIT, see [LICENSE](LICENSE).
