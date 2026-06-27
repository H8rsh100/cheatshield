"""
CheatShield AI — ML Training Pipeline
=======================================
Trains a 5-model voting ensemble, K-Means archetype clustering,
DBSCAN anomaly detection, and PCA visualization on behavioral data.

Supports both real PUBG-derived data and synthetic fallback.
Outputs trained models, scaler, and evaluation metrics.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import hashlib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


def file_hash(path: str) -> str:
    """Compute MD5 hash of a file for versioning."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def train():
    """Run the complete training pipeline."""
    print("=" * 65)
    print("  CheatShield AI — ML Training Pipeline")
    print("=" * 65)

    # ── 1. Load Data ─────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    FEATURES = [c for c in df.columns if c != "label"]
    X, y = df[FEATURES].values, df["label"].values

    print(f"\n📂 Dataset: {len(df)} rows | {FEATURES}")
    print(f"   Class distribution: {(y == 0).sum()} legit / {(y == 1).sum()} cheaters "
          f"({y.mean()*100:.1f}% cheat rate)")

    # ── 2. Train/Test Split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n📊 Split: {len(X_train)} train / {len(X_test)} test")

    # ── 3. Scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # ── 4. PCA (2D visualization) ────────────────────────────────────────
    pca = PCA(n_components=2)
    pca.fit(X_train_sc)
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
    print(f"\n🔵 PCA: explained variance = {pca.explained_variance_ratio_.sum():.3f}")

    # ── 5. Individual Classifiers ────────────────────────────────────────
    lr  = LogisticRegression(C=1.0, max_iter=1000)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                        max_iter=500, early_stopping=True, random_state=42)
    lda = LinearDiscriminantAnalysis()

    # ── 6. Voting Ensemble ───────────────────────────────────────────────
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("svm", svm), ("rf", rf), ("mlp", mlp), ("lda", lda)],
        voting="soft"
    )

    # Cross-validation (5-fold stratified)
    print("\n⏳ Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        ensemble, X_train_sc, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        return_train_score=False
    )

    cv_metrics = {
        "cv_accuracy":  float(np.mean(cv_results["test_accuracy"])),
        "cv_precision": float(np.mean(cv_results["test_precision"])),
        "cv_recall":    float(np.mean(cv_results["test_recall"])),
        "cv_f1":        float(np.mean(cv_results["test_f1"])),
    }

    print(f"   CV Accuracy:  {cv_metrics['cv_accuracy']:.4f} "
          f"(±{np.std(cv_results['test_accuracy']):.4f})")
    print(f"   CV Precision: {cv_metrics['cv_precision']:.4f}")
    print(f"   CV Recall:    {cv_metrics['cv_recall']:.4f}")
    print(f"   CV F1:        {cv_metrics['cv_f1']:.4f}")

    # Final fit on full training set
    print("\n🧠 Training final ensemble...")
    ensemble.fit(X_train_sc, y_train)
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "ensemble.pkl"))

    # Test set evaluation
    preds = ensemble.predict(X_test_sc)
    proba = ensemble.predict_proba(X_test_sc)

    print("\n=== ENSEMBLE CLASSIFICATION REPORT ===")
    report = classification_report(y_test, preds, target_names=["Legit", "Cheater"],
                                    output_dict=True)
    print(classification_report(y_test, preds, target_names=["Legit", "Cheater"]))

    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    # ── 7. Random Forest Feature Importance ──────────────────────────────
    rf_model = ensemble.named_estimators_["rf"]
    importance = dict(zip(FEATURES, rf_model.feature_importances_.tolist()))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("\n📈 Feature Importance (Random Forest):")
    for feat, imp in sorted_importance.items():
        bar = "█" * int(imp * 50)
        print(f"   {feat:25s} {imp:.4f} {bar}")

    # ── 8. K-Means: Player Archetypes ────────────────────────────────────
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_train_sc)
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans.pkl"))
    print(f"\n🎮 K-Means: 4 archetypes fitted")

    # ── 9. DBSCAN: Anomaly Detection ─────────────────────────────────────
    db = DBSCAN(eps=1.5, min_samples=5)
    db_labels = db.fit_predict(X_train_sc)
    n_anomalies = (db_labels == -1).sum()
    n_clusters  = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    joblib.dump(db, os.path.join(MODEL_DIR, "dbscan.pkl"))
    print(f"🔍 DBSCAN: {n_clusters} clusters, {n_anomalies} anomalies "
          f"({n_anomalies/len(db_labels)*100:.1f}%)")

    # ── 10. Save Feature Names ───────────────────────────────────────────
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

    # ── 11. Save Training Metrics ────────────────────────────────────────
    metrics = {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "dataset_hash":     file_hash(DATA_PATH),
        "dataset_rows":     len(df),
        "dataset_cheaters": int(y.sum()),
        "train_size":       len(X_train),
        "test_size":        len(X_test),
        "features":         FEATURES,
        **cv_metrics,
        "test_accuracy":    float(report["accuracy"]),
        "test_precision":   float(report["Cheater"]["precision"]),
        "test_recall":      float(report["Cheater"]["recall"]),
        "test_f1":          float(report["Cheater"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "feature_importance": sorted_importance,
        "dbscan_anomalies": int(n_anomalies),
        "dbscan_clusters":  int(n_clusters),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Metrics saved to: {metrics_path}")
    print(f"\n✅ All models saved to: {MODEL_DIR}/")
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    train()