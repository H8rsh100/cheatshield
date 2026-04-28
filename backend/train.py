import pandas as pd
import numpy as np
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

os.makedirs("backend/models", exist_ok=True)

df = pd.read_csv("backend/data.csv")
FEATURES = [c for c in df.columns if c != "label"]
X, y = df[FEATURES].values, df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, "backend/models/scaler.pkl")

# ── PCA (for visualization, 2D) ─────────────────────────────────────────────
pca = PCA(n_components=2)
pca.fit(X_train_sc)
joblib.dump(pca, "backend/models/pca.pkl")

# ── Individual classifiers ───────────────────────────────────────────────────
lr  = LogisticRegression(C=1.0, max_iter=1000)
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                    max_iter=500, early_stopping=True, random_state=42)
lda = LinearDiscriminantAnalysis()

# ── Voting Ensemble ──────────────────────────────────────────────────────────
ensemble = VotingClassifier(
    estimators=[("lr", lr), ("svm", svm), ("rf", rf), ("mlp", mlp), ("lda", lda)],
    voting="soft"
)
ensemble.fit(X_train_sc, y_train)
joblib.dump(ensemble, "backend/models/ensemble.pkl")

preds = ensemble.predict(X_test_sc)
print("=== ENSEMBLE REPORT ===")
print(classification_report(y_test, preds, target_names=["Legit", "Cheater"]))

# ── K-Means: Player Archetypes (unsupervised) ────────────────────────────────
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_train_sc)
joblib.dump(km, "backend/models/kmeans.pkl")

# ── DBSCAN: Anomaly Detection ────────────────────────────────────────────────
db = DBSCAN(eps=1.2, min_samples=5)
db.fit(X_train_sc)
joblib.dump(db, "backend/models/dbscan.pkl")

# Save feature names
joblib.dump(FEATURES, "backend/models/features.pkl")
print("✅ All models saved!")