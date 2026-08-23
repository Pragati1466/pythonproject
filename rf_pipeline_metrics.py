"""
Leakage-free Random Forest anomaly-detection pipeline for the AV cybersecurity
dataset (test_data.csv). Produces the metrics used in the project summary:

  - Accuracy / Precision / Recall / F1 (stratified k-fold CV)
  - Feature engineering + feature-space reduction (A -> B) via RF importance
  - Per-attack-type detection rate + overall false-positive rate
  - Random Forest vs Logistic Regression / SVM / Gradient Boosting benchmark
  - Robustness under simulated adversarial (FGSM-style) noise perturbation

NOTE: The raw CSV bundles post-attack outcome columns (Attack_Type,
Attack_Severity, Response_Action, Attack_Duration, Attack_Frequency) that are
only populated *after* an attack is confirmed - training on them causes
data leakage (accuracy.py in this repo does this). This script instead uses
only genuine pre-detection sensor/vehicle telemetry so the reported numbers
reflect real detection difficulty on this dataset.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
N_SPLITS = 5

# ---------------------------------------------------------------------------
# 1. Load + feature-engineer
# ---------------------------------------------------------------------------
df = pd.read_csv("test_data.csv", delimiter="\t")

df["Vehicle_Speed_kmh"] = df["Vehicle_Speed"].str.extract(r"(\d+\.?\d*)").astype(float)
df["Network_Traffic_MBs"] = df["Network_Traffic"].str.extract(r"(\d+\.?\d*)").astype(float)
loc = df["Location"].str.split(",", expand=True)
df["Latitude"] = loc[0].astype(float)
df["Longitude"] = loc[1].astype(float)
df["Hour"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%y %H:%M").dt.hour
df["Error_Code_Present"] = (df["Error_Code"].astype(str) != "None").astype(int)

# Genuine pre-detection features only (no post-attack outcome leakage)
numerical_features = [
    "Sensor_Data", "Vehicle_Speed_kmh", "Network_Traffic_MBs",
    "Latitude", "Longitude", "Hour",
]
categorical_features = [
    "Sensor_Type", "Sensor_Status", "Vehicle_Model",
    "Firmware_Version", "Geofencing_Status", "Error_Code_Present",
]

X_raw = df[numerical_features + categorical_features].copy()
y = df["Adversarial_Attack"].astype(int)
attack_type = df["Attack_Type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# Fit once to report the raw -> encoded feature-space size (A)
X_encoded_full = preprocessor.fit_transform(X_raw)
n_features_A = X_encoded_full.shape[1]
n_raw_features = len(numerical_features) + len(categorical_features)

print("=" * 70)
print(f"Raw engineered features (before encoding): {n_raw_features}")
print(f"Encoded feature space A (after one-hot):    {n_features_A}")
print("=" * 70)

# ---------------------------------------------------------------------------
# 2. Cross-validated Random Forest performance (Accuracy/Precision/Recall/F1)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])

# Nested CV: inner GridSearchCV tunes hyperparameters on each outer training
# fold, outer StratifiedKFold produces unbiased held-out predictions.
rf_param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}
rf_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=inner_cv, scoring="f1", n_jobs=-1)
y_pred_cv = cross_val_predict(rf_search, X_raw, y, cv=cv)

# Fit once on full data to report the best hyperparameters found
rf_search.fit(X_raw, y)
print(f"Best RF hyperparameters (full-data grid search): {rf_search.best_params_}")
best_rf_params = {k.replace("clf__", ""): v for k, v in rf_search.best_params_.items()}

acc = accuracy_score(y, y_pred_cv)
prec = precision_score(y, y_pred_cv)
rec = recall_score(y, y_pred_cv)
f1 = f1_score(y, y_pred_cv)
tn, fp, fn, tp = confusion_matrix(y, y_pred_cv).ravel()
fpr = fp / (fp + tn)

print(f"\n[Random Forest | {N_SPLITS}-fold Stratified CV | N={len(y)} samples, {n_features_A} features]")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%")
print(f"Confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"False-Positive Rate: {fpr*100:.2f}%")
print(f"Threat-Detection Rate (Recall): {rec*100:.2f}%")

# ---------------------------------------------------------------------------
# 3. Feature-space reduction via RF importance (A -> B)
# ---------------------------------------------------------------------------
best_rf_pipeline = rf_search.best_estimator_
ohe = best_rf_pipeline.named_steps["prep"].named_transformers_["cat"]
encoded_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
importances = best_rf_pipeline.named_steps["clf"].feature_importances_

order = np.argsort(importances)[::-1]
cumulative = np.cumsum(importances[order])
# Keep smallest set of top features covering >=90% cumulative importance
n_features_B = int(np.searchsorted(cumulative, 0.90) + 1)
top_feature_idx = order[:n_features_B]

print(f"\nFeature importance ranking (top {n_features_B} of {n_features_A}, "
      f">=90% cumulative importance):")
for rank, idx in enumerate(top_feature_idx, start=1):
    print(f"  {rank}. {encoded_names[idx]:<25} importance={importances[idx]:.4f}")

# Re-evaluate RF restricted to the reduced encoded feature subset
X_full_dense = np.asarray(X_encoded_full.todense()) if hasattr(X_encoded_full, "todense") else X_encoded_full
X_reduced = X_full_dense[:, top_feature_idx]

rf_reduced = RandomForestClassifier(random_state=RANDOM_STATE, **best_rf_params)
y_pred_reduced = cross_val_predict(rf_reduced, X_reduced, y, cv=cv)
acc_reduced = accuracy_score(y, y_pred_reduced)
f1_reduced = f1_score(y, y_pred_reduced)

print(f"\nFeature space reduced: {n_features_A} -> {n_features_B} variables "
      f"({(1 - n_features_B/n_features_A)*100:.1f}% reduction)")
print(f"Accuracy retained after reduction: {acc_reduced*100:.2f}% "
      f"(full-feature accuracy: {acc*100:.2f}%)")
print(f"F1 retained after reduction:       {f1_reduced*100:.2f}% "
      f"(full-feature F1: {f1*100:.2f}%)")

# ---------------------------------------------------------------------------
# 4. Per-attack-type detection rate
# ---------------------------------------------------------------------------
print(f"\nAttack scenarios evaluated: {attack_type[attack_type != 'None'].nunique()} distinct types "
      f"across {y.sum()} adversarial samples")
detect_df = pd.DataFrame({"attack_type": attack_type, "y": y, "y_pred": y_pred_cv})
per_type = detect_df[detect_df["y"] == 1].groupby("attack_type")[["y_pred"]].apply(
    lambda g: (g["y_pred"] == 1).mean()
)
print("Per-attack-type detection rate:")
for t, rate in per_type.items():
    print(f"  {t:<22}: {rate*100:.1f}%")
overall_threat_detection = (detect_df[detect_df["y"] == 1]["y_pred"] == 1).mean()
print(f"Overall threat-detection rate: {overall_threat_detection*100:.2f}%")
print(f"Overall false-positive rate:   {fpr*100:.2f}%")

# ---------------------------------------------------------------------------
# 5. Random Forest vs baseline models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, **best_rf_params),
}

print("\nModel comparison (5-fold Stratified CV, same feature set):")
results = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    preds = cross_val_predict(pipe, X_raw, y, cv=cv)
    a = accuracy_score(y, preds)
    p = precision_score(y, preds, zero_division=0)
    r = recall_score(y, preds, zero_division=0)
    f = f1_score(y, preds, zero_division=0)
    results[name] = dict(accuracy=a, precision=p, recall=r, f1=f)
    print(f"  {name:<22} Acc={a*100:5.2f}%  Prec={p*100:5.2f}%  Rec={r*100:5.2f}%  F1={f*100:5.2f}%")

best_baseline = max(
    (v for k, v in results.items() if k != "Random Forest"), key=lambda v: v["f1"]
)
rf_result = results["Random Forest"]
improvement = (rf_result["f1"] - best_baseline["f1"]) / max(best_baseline["f1"], 1e-9) * 100
print(f"\nRF F1 improvement over strongest baseline: {improvement:.1f}%")

# ---------------------------------------------------------------------------
# 6. Robustness under simulated adversarial (FGSM-style) perturbation
# ---------------------------------------------------------------------------
print("\nRobustness under adversarial (sign-noise / FGSM-style) perturbation:")
rf_final = RandomForestClassifier(random_state=RANDOM_STATE, **best_rf_params)
rf_final.fit(X_encoded_full, y)

rng = np.random.RandomState(RANDOM_STATE)
n_numeric = len(numerical_features)
for eps in [0.0, 0.1, 0.2, 0.3]:
    X_pert = X_full_dense.copy()
    noise = eps * rng.choice([-1, 1], size=(X_pert.shape[0], n_numeric))
    X_pert[:, :n_numeric] = X_pert[:, :n_numeric] + noise
    preds = rf_final.predict(X_pert)
    a = accuracy_score(y, preds)
    p = precision_score(y, preds, zero_division=0)
    r = recall_score(y, preds, zero_division=0)
    print(f"  eps={eps:.1f}  Accuracy={a*100:.2f}%  Precision={p*100:.2f}%  Recall={r*100:.2f}%")
