# -*- coding: utf-8 -*-
"""Alzheimer's disease screening from EEG: Transfer Entropy, brain networks and machine learning.

Run with:  python eeg_alzheimer_transfer_entropy_ml.py
Install dependencies first:  pip install -r requirements.txt

# Alzheimer's disease screening from EEG: Transfer Entropy, brain networks and machine learning

This notebook implements the full pipeline of the project *Alzheimer's Disease Diagnosis through Low-Cost
Neurophysiological Screening by Effective Brain Connectivity (EEG) and Machine Learning Models*.

**Pipeline**

1. Download the public EEG dataset (OpenNeuro `ds004504`) and keep Alzheimer's (AD) and healthy controls (CN).
2. Preprocess each recording: 1-40 Hz band-pass FIR filter, resampling to 200 Hz.
3. Discretize each channel into three states using its own standard deviation.
4. Estimate directed **Transfer Entropy (TE)** between every pair of the 19 electrodes
   (optionally keeping only edges that pass a surrogate-based significance test).
5. Extract **network features** from the TE graph and **spectral features** from the signal.
6. Univariate statistics (Mann-Whitney U with Benjamini-Hochberg FDR correction).
7. Compare six classifiers with repeated stratified cross-validation.
8. Interpret the models (logistic regression coefficients, permutation importance) and plot the average TE graphs.

## 1. Setup
"""


import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.patches import FancyArrowPatch
from pyinform.transferentropy import transfer_entropy
from scipy.signal import welch
from scipy.stats import kurtosis, mannwhitneyu, skew
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
trapezoid = getattr(np, "trapezoid", None) or np.trapz

DATA_DIR = Path("data/ds004504")      #raw dataset
OUTPUT_DIR = Path("outputs")          #features, TE matrices and figures
FIG_DIR = OUTPUT_DIR / "figures"
FEATURES_CSV = OUTPUT_DIR / "features.csv"
TE_MATRICES_NPZ = OUTPUT_DIR / "te_matrices.npz"

BANDPASS_HZ = (1, 40)        #FIR band-pass filter
TARGET_SFREQ = 200           #Hz, resampling frequency
TE_HISTORY = 2               #history length k of the Transfer Entropy
N_SURROGATES = 25            #circular-shift surrogates per electrode pair
P_THRESHOLD = 0.05
APPLY_SURROGATE_MASK = False
RANDOM_STATE = 42
N_JOBS = -1                  #parallel workers (-1 = all cores)
RECOMPUTE_FEATURES = False   #True ignores cached outputs/features.csv

FIG_DIR.mkdir(parents=True, exist_ok=True)

"""## 2. Data

Dataset: **OpenNeuro ds004504** (Miltiadous et al., 2023): resting-state, eyes-closed EEG, 19 electrodes
(10-20 system), 500 Hz. Groups in `participants.tsv`: `A` = Alzheimer's, `F` = frontotemporal dementia (not used here),
`C` = healthy controls. Only A and C are analysed (36 AD + 29 CN = 65 participants).
"""

def download_dataset(data_dir: Path, need_eeg: bool = True) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / "participants.tsv").exists():
        import openneuro as on
        on.download(dataset="ds004504", target_dir=data_dir, include=["participants.tsv", "participants.json", "dataset_description.json"])
    if not need_eeg:
        return

    table = pd.read_csv(data_dir / "participants.tsv", sep="\t")
    wanted = table.loc[table["Group"].isin(["A", "C"]), "participant_id"]
    missing = [s for s in wanted if not eeg_path(s).exists()]

    if missing:
        import openneuro as on
        on.download(dataset="ds004504", target_dir=data_dir, include=[f"{s}/eeg" for s in missing])


def eeg_path(subject_id: str) -> Path:
    return DATA_DIR / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"


download_dataset(DATA_DIR, need_eeg=RECOMPUTE_FEATURES or not FEATURES_CSV.exists())

participants = pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")
participants = participants[participants["Group"].isin(["A", "C"])].reset_index(drop=True)
print(participants["Group"].value_counts().rename({"A": "Alzheimer (AD)", "C": "Control (CN)"}))
participants.head()

"""## 3. Preprocessing, discretization and Transfer Entropy

**Transfer Entropy** (Schreiber, 2000) measures the reduction in uncertainty about the future of a target signal
*Y* given the past of a source *X*, beyond what the past of *Y* already explains. It is directional and non-linear.
Estimation needs discrete data, so each channel is symbolized in three states using its own standard deviation
(σ): below -σ, between -σ and +σ, above +σ.

The optional surrogate test builds `N_SURROGATES` copies of the source signal by random circular shifts. A shift
destroys the temporal relation with the target but keeps the signal statistics. The empirical p-value of an edge
is the share of surrogates whose TE is greater than or equal to the observed TE.
"""

def preprocess_eeg(set_file: Path) -> mne.io.BaseRaw:
    """Load an EEGLAB recording, band-pass filter it (zero-phase FIR) and resample it"""
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
    raw.filter(*BANDPASS_HZ, verbose=False)
    raw.resample(TARGET_SFREQ, verbose=False)
    return raw


def discretize_by_sigma(data: np.ndarray) -> np.ndarray:
    """Symbolize every channel into 3 states (0, 1, 2) using thresholds at -sigma and +sigma."""
    symbolic = np.empty(data.shape, dtype=int)
    for ch, signal in enumerate(data):
        sigma = signal.std()
        symbolic[ch] = np.digitize(signal, [-sigma, sigma])
    return symbolic


def _te_for_pair(i, j, symbolic, k, n_surrogates, seed):
    """TE from electrode i to electrode j, plus the surrogate p-value when requested."""
    source, target = symbolic[i], symbolic[j]
    te = transfer_entropy(source, target, k=k)
    if n_surrogates == 0:
        return i, j, te, None
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, len(source), size=n_surrogates)
    surrogate_te = np.array([transfer_entropy(np.roll(source, s), target, k=k) for s in shifts])
    p_value = (np.sum(surrogate_te >= te) + 1) / (n_surrogates + 1)
    return i, j, te, p_value


def compute_te_matrix(symbolic, k=TE_HISTORY, apply_mask=APPLY_SURROGATE_MASK, n_surrogates=N_SURROGATES, p_threshold=P_THRESHOLD, seed=RANDOM_STATE):
    """Directed TE matrix: te[i, j] is the information flowing from electrode i to electrode j."""
    n = symbolic.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    n_surr = n_surrogates if apply_mask else 0
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_te_for_pair)(i, j, symbolic, k, n_surr, seed + i * n + j) for i, j in pairs)

    te = np.zeros((n, n))
    for i, j, value, p_value in results:
        if apply_mask and p_value >= p_threshold:
            continue             #edge not significant: leave it at zero
        te[i, j] = value
    return te

"""## 4. Network features

Each TE matrix is a weighted, directed graph (19 nodes = electrodes). Extracted features:

* **Weight distribution**: mean, standard deviation, coefficient of variation, skewness, kurtosis and entropy of the edge weights.
* **Nodal flow** per electrode: outflow, inflow and **netflow** (outflow minus inflow).
* **Asymmetry** (non-reciprocity of the flow) and **spectral radius** of the matrix.
* **Global efficiency** (distance = inverse of the weight), **mean weighted clustering** (Onnela), **eigenvector centrality** per electrode.
* **Participation coefficient** per electrode, using the anatomical lobes (frontal, central, temporal, parietal, occipital) as modules.
* **Maximum spanning tree** (MST) backbone: leaf fraction and diameter.
"""

EPS = 1e-12
LOBES = {
    "F": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "C": ["C3", "C4", "Cz"],
    "T": ["T3", "T4", "T5", "T6"],
    "P": ["P3", "P4", "Pz"],
    "O": ["O1", "O2"],
}


def weighted_global_efficiency(w_sym: np.ndarray) -> float:
    """Mean inverse shortest-path length, with edge length = 1 / weight"""
    lengths = 1.0 / (w_sym + EPS)
    np.fill_diagonal(lengths, 0)
    graph = nx.from_numpy_array(lengths)
    distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    n = len(lengths)
    return float(np.mean([1.0 / distances[i][j] for i in range(n) for j in range(n) if i != j]))


def extract_network_features(te: np.ndarray, ch_names: list) -> dict:
    """Graph features of one participant's TE matrix"""
    features = {}
    n_nodes = te.shape[0]
    weights = te[te > 0]

    #distribution of the edge weights
    features["mean_te"] = weights.mean()
    features["std_te"] = weights.std()
    features["cv_te"] = weights.std() / (weights.mean() + EPS)
    features["skew_te"] = skew(weights)
    features["kurtosis_te"] = kurtosis(weights)
    p = weights / (weights.sum() + EPS)
    features["weight_entropy"] = -np.sum(p * np.log(p + EPS))

    #nodal strength and net information flow
    out_strength = te.sum(axis=1)
    in_strength = te.sum(axis=0)
    net_flow = out_strength - in_strength
    for i, ch in enumerate(ch_names):
        features[f"{ch}_out"] = out_strength[i]
        features[f"{ch}_in"] = in_strength[i]
        features[f"{ch}_netflow"] = net_flow[i]

    #asymmetry of the flow and spectral radius
    features["asymmetry_norm"] = np.linalg.norm(te - te.T)
    features["spectral_radius"] = np.abs(np.linalg.eigvals(te)).max()

    #weighted measures
    te_sym = np.maximum(te, te.T)
    g_directed = nx.from_numpy_array(te, create_using=nx.DiGraph)
    g_undirected = nx.from_numpy_array(te_sym)

    features["global_efficiency"] = weighted_global_efficiency(te_sym)
    features["mean_clustering"] = np.mean(list(nx.clustering(g_undirected, weight="weight").values()))

    try:
        centrality = nx.eigenvector_centrality_numpy(g_directed, weight="weight")
    except Exception:
        centrality = {i: 0.0 for i in range(n_nodes)}
    for i, ch in enumerate(ch_names):
        features[f"{ch}_ev_centrality"] = centrality[i]

    #participation coefficient with the anatomical lobes as modules
    lobe_of = {i: lobe for i, ch in enumerate(ch_names) for lobe, members in LOBES.items() if ch in members}
    strength = te + te.T
    for i, ch in enumerate(ch_names):
        total = strength[i].sum()
        if total == 0:
            features[f"{ch}_participation"] = 0.0
            continue
        per_lobe = {lobe: 0.0 for lobe in LOBES}
        for j, lobe in lobe_of.items():
            per_lobe[lobe] += strength[i, j]
        features[f"{ch}_participation"] = 1 - sum((v / total) ** 2 for v in per_lobe.values())

    mst = nx.maximum_spanning_tree(g_undirected, weight="weight")
    leaves = [node for node, degree in mst.degree() if degree == 1]
    features["mst_leaf_fraction"] = len(leaves) / n_nodes
    features["mst_diameter"] = nx.diameter(mst)
    return features

"""## 5. Spectral features

Classic EEG measures used as a reference, to test whether directed connectivity adds information beyond them:
relative band power (delta, theta, alpha, beta, gamma) from Welch's power spectral density, Shannon and spectral
entropy, amplitude coefficient of variation, signal kurtosis, the theta/alpha ratio and a global slowing index
(delta + theta) / (alpha + beta).

> Relative power is normalised by the total power of the whole spectrum (0 to 100 Hz, the Nyquist frequency after
> resampling). The data are low-pass filtered at 40 Hz, so the gamma band (30-45 Hz) effectively covers 30-40 Hz.
"""

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}

def extract_spectral_features(data: np.ndarray, sfreq: float, ch_names: list) -> dict:
    """Spectral and amplitude features of one recording (data: channels x samples)"""
    features = {}
    band_power = {band: [] for band in BANDS}

    for i, ch in enumerate(ch_names):
        signal = data[i]
        freqs, psd = welch(signal, fs=sfreq, nperseg=1024)
        total_power = trapezoid(psd, freqs)

        for band, (f_low, f_high) in BANDS.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            relative = trapezoid(psd[mask], freqs[mask]) / (total_power + EPS)
            features[f"{ch}_{band}_power"] = relative
            band_power[band].append(relative)

        hist, _ = np.histogram(signal, bins=50, density=True)
        p = hist[hist > 0]
        features[f"{ch}_signal_entropy"] = -np.sum(p * np.log(p + EPS))

        p_spectrum = psd / (psd.sum() + EPS)
        features[f"{ch}_spectral_entropy"] = -np.sum(p_spectrum * np.log(p_spectrum + EPS))

        features[f"{ch}_amplitude_cv"] = signal.std() / (np.mean(np.abs(signal)) + EPS)
        features[f"{ch}_kurtosis"] = kurtosis(signal)

    mean_power = {band: np.mean(values) for band, values in band_power.items()}
    features["mean_theta_power"] = mean_power["theta"]
    features["mean_alpha_power"] = mean_power["alpha"]
    features["theta_alpha_ratio"] = mean_power["theta"] / (mean_power["alpha"] + EPS)
    features["global_slowing_index"] = ((mean_power["delta"] + mean_power["theta"]) / (mean_power["alpha"] + mean_power["beta"] + EPS))
    return features

"""## 6. Feature extraction for all participants

Features and TE matrices are cached in `outputs/`, so this slow step only runs once
(set `RECOMPUTE_FEATURES = True` to force it).
"""

def analyze_subject(subject_id: str):
    """Full pipeline for one participant: returns (feature dict, TE matrix)"""
    raw = preprocess_eeg(eeg_path(subject_id))
    data = raw.get_data()
    symbolic = discretize_by_sigma(data)
    te = compute_te_matrix(symbolic)

    features = {
        **extract_network_features(te, raw.ch_names),
        **extract_spectral_features(data, raw.info["sfreq"], raw.ch_names),
    }
    features["subject"] = subject_id
    return features, te, raw.ch_names


if FEATURES_CSV.exists() and TE_MATRICES_NPZ.exists() and not RECOMPUTE_FEATURES:
    feature_df = pd.read_csv(FEATURES_CSV)
    stored = np.load(TE_MATRICES_NPZ, allow_pickle=True)
    te_matrices = {key: stored[key] for key in stored.files if key != "ch_names"}
    CH_NAMES = list(stored["ch_names"])
    print(f"Loaded cached features: {feature_df.shape}")
else:
    rows, te_matrices = [], {}
    for subject in tqdm(participants["participant_id"], desc="Participants"):
        try:
            row, te, CH_NAMES = analyze_subject(subject)
        except Exception as error:
            print(f"Skipped {subject}: {error}")
            continue
        rows.append(row)
        te_matrices[subject] = te
    feature_df = pd.DataFrame(rows).merge(participants, left_on="subject", right_on="participant_id")
    feature_df.to_csv(FEATURES_CSV, index=False)
    np.savez(TE_MATRICES_NPZ, ch_names=np.array(CH_NAMES), **te_matrices)

METADATA_COLS = ["subject", "participant_id", "Gender", "Age", "Group", "MMSE"]
FEATURE_COLS = [c for c in feature_df.columns if c not in METADATA_COLS]
X = feature_df[FEATURE_COLS]
y = (feature_df["Group"] == "A").astype(int)   #1 = Alzheimer, 0 = control
assert not X.isna().any().any(), "NaN found in the feature table"
print(f"{len(FEATURE_COLS)} features for {len(X)} participants ({y.sum()} AD, {(1 - y).sum()} CN)")

"""## 7. Univariate statistics

Every feature is compared between AD and CN with the Mann-Whitney U test. P-values are corrected for multiple
comparisons with the Benjamini-Hochberg procedure (false discovery rate of 5%).
"""

def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR)"""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1.0)
    return result


SPECTRAL_PATTERN = re.compile(r"(_power|_signal_entropy|_spectral_entropy|_amplitude_cv|_kurtosis)$"
                              r"|^theta_alpha_ratio$|^global_slowing_index$")
category_of = lambda name: "Spectral" if SPECTRAL_PATTERN.search(name) else "Network (TE)"

raw_p = [mannwhitneyu(X.loc[y == 1, c], X.loc[y == 0, c], alternative="two-sided").pvalue for c in FEATURE_COLS]
stats_df = pd.DataFrame({"feature": FEATURE_COLS, "category": [category_of(c) for c in FEATURE_COLS],"p_raw": raw_p, "p_fdr": benjamini_hochberg(raw_p)}).sort_values("p_raw")
stats_df.to_csv(OUTPUT_DIR / "univariate_stats.csv", index=False)

significant = stats_df[stats_df["p_fdr"] < 0.05]
print(f"{len(significant)} of {len(stats_df)} features significant after FDR correction (alpha = 0.05)")
print(significant["category"].value_counts().to_string())
stats_df.head(15)

top_features = stats_df["feature"].head(8).tolist()
plot_df = feature_df[["Group"] + top_features].assign(Diagnosis=lambda d: d["Group"].map({"A": "AD", "C": "CN"}))

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feature in zip(axes.ravel(), top_features):
    sns.boxplot(data=plot_df, x="Diagnosis", y=feature, hue="Diagnosis", order=["CN", "AD"], hue_order=["CN", "AD"],
                palette={"CN": "#9CC3EA", "AD": "#F08A8A"}, legend=False, showfliers=False, ax=ax)
    sns.stripplot(data=plot_df, x="Diagnosis", y=feature, order=["CN", "AD"], color="black", size=3, alpha=.6, ax=ax)
    q = stats_df.loc[stats_df["feature"] == feature, "p_fdr"].iloc[0]
    ax.set(title=f"{feature}\nFDR p = {q:.1e}", xlabel="", ylabel="")
fig.suptitle("Features that best separate Alzheimer's from controls", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "top_features_boxplots.png", dpi=200, bbox_inches="tight")
plt.show()

"""## 8. Classification

Six classifiers, each inside a `Pipeline` with `StandardScaler`, are evaluated with **5-fold stratified cross-validation
repeated 10 times** (50 train/test splits). The positive class is Alzheimer's.

> The report does not list the hyperparameters of the boosting and forest models. The values below are reasonable
> small-sample settings and can be tuned; nothing is selected using test folds.
"""

MODELS = {
    "LogReg (ElasticNet)": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),

    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),

    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),

    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=1, verbose=-1),

    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced",  random_state=RANDOM_STATE),
}

make_pipeline = lambda name: Pipeline([("scale", StandardScaler()), ("clf", clone(MODELS[name]))])


def fold_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"ROC-AUC": roc_auc_score(y_true, y_proba),
            "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
            "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
            "Precision": precision_score(y_true, y_pred, zero_division=0), "F1": f1_score(y_true, y_pred)}


def evaluate_split(name, train_idx, test_idx):
    model = make_pipeline(name).fit(X.iloc[train_idx], y.iloc[train_idx])
    return fold_metrics(y.iloc[test_idx], model.predict_proba(X.iloc[test_idx])[:, 1])


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
splits = list(cv.split(X, y))

summary = {}
for name in tqdm(MODELS, desc="Models"):
    folds = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(evaluate_split)(name, tr, te) for tr, te in splits))
    summary[name] = {metric: f"{folds[metric].mean():.3f} ± {folds[metric].std():.3f}" for metric in folds}
    summary[name]["_auc"] = folds["ROC-AUC"].mean()

results_df = (pd.DataFrame(summary).T.sort_values("_auc", ascending=False).drop(columns="_auc"))
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv")
results_df

"""## 9. ROC curves and confusion matrices

Out-of-sample predictions from a **single** stratified 5-fold split (every participant is predicted by a model that
never saw them), pooled to draw one ROC curve and one confusion matrix per classifier.
"""

single_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_proba = {name: cross_val_predict(make_pipeline(name), X, y, cv=single_cv, method="predict_proba")[:, 1] for name in MODELS}

fig = plt.figure(figsize=(15, 10))
grid = fig.add_gridspec(3, 4)
ax_roc = fig.add_subplot(grid[:, :2])

for name, proba in oof_proba.items():
    fpr, tpr, _ = roc_curve(y, proba)
    ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y, proba):.2f})")
ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
ax_roc.set(xlabel="1 - Specificity", ylabel="Sensitivity", title="ROC curves (out-of-sample)")
ax_roc.legend(loc="lower right")

for k, (name, proba) in enumerate(oof_proba.items()):
    ax = fig.add_subplot(grid[k // 2, 2 + k % 2])
    cm = confusion_matrix(y, (proba >= 0.5).astype(int), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, ax=ax, xticklabels="CN AD".split(), yticklabels="CN AD".split())
    ax.set(title=name, xlabel="Predicted", ylabel="Actual")
fig.tight_layout()
fig.savefig(FIG_DIR / "roc_and_confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.show()

"""## 10. Interpretability

* **Logistic regression**: standardized coefficients of the model fitted on all participants (positive favours AD, negative favours CN).
* **LightGBM permutation importance**: how much the ROC-AUC on held-out participants drops when one feature is shuffled
  (averaged over the 5 test folds).
"""

logreg = make_pipeline("LogReg (ElasticNet)").fit(X, y)
coefficients = pd.Series(logreg.named_steps["clf"].coef_[0], index=FEATURE_COLS)
top_coef = coefficients.reindex(coefficients.abs().sort_values(ascending=False).head(20).index).sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top_coef.index, top_coef.values, color=np.where(top_coef > 0, "#E4504F", "#4C9AE6"))
ax.axvline(0, color="black", lw=1)
ax.set(xlabel="Standardized coefficient", title="Top 20 logistic-regression features (red: AD, blue: CN)")
fig.tight_layout()
fig.savefig(FIG_DIR / "logreg_coefficients.png", dpi=200, bbox_inches="tight")
plt.show()

importances = []
for train_idx, test_idx in single_cv.split(X, y):
    model = make_pipeline("LightGBM").fit(X.iloc[train_idx], y.iloc[train_idx])
    result = permutation_importance(model, X.iloc[test_idx], y.iloc[test_idx], scoring="roc_auc", n_jobs=-1)
    importances.append(result.importances_mean)

perm = pd.Series(np.mean(importances, axis=0), index=FEATURE_COLS).sort_values(ascending=False)
perm.to_csv(OUTPUT_DIR / "lightgbm_permutation_importance.csv", header=["mean_auc_drop"])
print("Features with importance above zero:", int((perm > 0).sum()))

fig, ax = plt.subplots(figsize=(8, 5))
top_perm = perm.head(15)[::-1]
ax.barh(top_perm.index, top_perm.values, color="#2B35F5")
ax.set(xlabel="Mean drop in ROC-AUC when shuffled", title="LightGBM permutation importance (top 15)")
fig.tight_layout()
fig.savefig(FIG_DIR / "lightgbm_permutation_importance.png", dpi=200, bbox_inches="tight")
plt.show()
perm.head(10)

"""## 11. Average effective-connectivity graphs

Mean TE matrix of each group drawn over the real 10-20 electrode layout. Arrow colour and thickness are proportional
to TE, using the same scale for both groups, and only the strongest edges (top 25% of the pooled average) are drawn.
"""

LEGACY_TO_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
montage_pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
positions = {ch: montage_pos[LEGACY_TO_1020.get(ch, ch)][:2] for ch in CH_NAMES}

def mean_te(group_code):
    ids = feature_df.loc[feature_df["Group"] == group_code, "subject"]
    return np.mean([te_matrices[s] for s in ids], axis=0)

means = {"Alzheimer (AD)": mean_te("A"), "Control (CN)": mean_te("C")}
off_diagonal = ~np.eye(len(CH_NAMES), dtype=bool)
pooled = np.mean(list(means.values()), axis=0)
edge_cutoff = np.percentile(pooled[off_diagonal], 75)
vmin, vmax = edge_cutoff, max(m[off_diagonal].max() for m in means.values())
cmap = plt.get_cmap("Reds")

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, (title, matrix) in zip(axes, means.items()):
    for i, source in enumerate(CH_NAMES):
        for j, target in enumerate(CH_NAMES):
            weight = matrix[i, j]
            if i == j or weight < edge_cutoff:
                continue
            strength = (weight - vmin) / (vmax - vmin + EPS)
            ax.add_patch(FancyArrowPatch(positions[source], positions[target], connectionstyle="arc3,rad=0.15", arrowstyle="-|>", mutation_scale=9, lw=0.6 + 2.5 * strength, color=cmap(0.25 + 0.7 * strength), alpha=0.85, zorder=1))
    for ch, (x, y_pos) in positions.items():
        ax.scatter(x, y_pos, s=520, facecolor="white", edgecolor="black", zorder=2)
        ax.text(x, y_pos, ch, ha="center", va="center", fontsize=7, zorder=3)
    ax.set_title(f"{title} (n = {int((feature_df['Group'] == ('A' if 'AD' in title else 'C')).sum())})")
    ax.set_aspect("equal")
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG_DIR / "mean_te_graphs.png", dpi=200, bbox_inches="tight")
plt.show()

"""## References

* Miltiadous, A. et al. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and
  Healthy Subjects from Routine EEG. *Data*, 8(6), 95. OpenNeuro `ds004504`.
* Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
* Rubinov, M. & Sporns, O. (2010). Complex network measures of brain connectivity. *NeuroImage*, 52(3), 1059-1069.
"""