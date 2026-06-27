"""
CheatShield AI — Flask API Server
===================================
Serves the anti-cheat dashboard and provides ML prediction endpoints.
Uses a 5-model voting ensemble, K-Means archetypes, DBSCAN anomaly
detection, and PCA visualization.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import sqlite3
import json
import numpy as np
import os

app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")
CORS(app)

# ── Load Models ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

scaler   = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
ensemble = joblib.load(os.path.join(BASE_DIR, "models", "ensemble.pkl"))
kmeans   = joblib.load(os.path.join(BASE_DIR, "models", "kmeans.pkl"))
pca      = joblib.load(os.path.join(BASE_DIR, "models", "pca.pkl"))
dbscan   = joblib.load(os.path.join(BASE_DIR, "models", "dbscan.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

# Load training metrics if available
metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
model_metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        model_metrics = json.load(f)

ARCHETYPES = {
    0: "🐢 Casual Noob",
    1: "⚔️ Aggressive Rusher",
    2: "🎯 Precision Pro",
    3: "👻 Suspicious Player"
}

# ── Database Setup ───────────────────────────────────────────────────────────

db_path = os.path.join(BASE_DIR, "sessions.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    features TEXT, prediction INTEGER,
    confidence REAL, archetype TEXT,
    pca_x REAL, pca_y REAL,
    is_anomaly INTEGER DEFAULT 0,
    risk_score REAL DEFAULT 0
)""")
conn.commit()


def _dbscan_is_anomaly(x_scaled: np.ndarray) -> bool:
    """
    Check if a sample is an anomaly using the trained DBSCAN model.
    A point is anomalous if its nearest core sample distance exceeds eps,
    meaning DBSCAN would label it as noise (-1).
    """
    if not hasattr(dbscan, 'components_') or len(dbscan.components_) == 0:
        return False
    distances = np.linalg.norm(dbscan.components_ - x_scaled, axis=1)
    min_dist = distances.min()
    return bool(min_dist > dbscan.eps)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    x_raw = np.array([[data[f] for f in features]])
    x_sc  = scaler.transform(x_raw)

    pred       = int(ensemble.predict(x_sc)[0])
    confidence = float(ensemble.predict_proba(x_sc)[0][pred])
    archetype  = ARCHETYPES[int(kmeans.predict(x_sc)[0])]
    pca_coords = pca.transform(x_sc)[0].tolist()
    is_anomaly = _dbscan_is_anomaly(x_sc[0])

    # Risk score: combine ensemble confidence with anomaly flag
    base_risk = confidence * 100 if pred == 1 else (1 - confidence) * 100
    risk_score = min(100, base_risk + (15 if is_anomaly else 0))

    conn.execute(
        """INSERT INTO sessions
           (features, prediction, confidence, archetype, pca_x, pca_y, is_anomaly, risk_score)
           VALUES (?,?,?,?,?,?,?,?)""",
        (json.dumps(data), pred, confidence, archetype,
         pca_coords[0], pca_coords[1], int(is_anomaly), round(risk_score, 1))
    )
    conn.commit()

    return jsonify({
        "prediction":  "CHEATER 🚨" if pred == 1 else "LEGIT ✅",
        "label":       pred,
        "confidence":  round(confidence * 100, 2),
        "archetype":   archetype,
        "pca":         pca_coords,
        "risk_score":  round(risk_score, 1),
        "is_anomaly":  is_anomaly,
    })


@app.route("/history", methods=["GET"])
def history():
    rows = conn.execute(
        """SELECT timestamp, prediction, confidence, archetype, pca_x, pca_y, is_anomaly
           FROM sessions ORDER BY id DESC LIMIT 50"""
    ).fetchall()
    return jsonify([{
        "timestamp": r[0], "prediction": r[1], "confidence": round(r[2]*100, 1),
        "archetype": r[3], "pca_x": r[4], "pca_y": r[5],
        "is_anomaly": bool(r[6]) if r[6] is not None else False
    } for r in rows])


@app.route("/stats", methods=["GET"])
def stats():
    total    = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    cheaters = conn.execute("SELECT COUNT(*) FROM sessions WHERE prediction=1").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM sessions").fetchone()[0] or 0
    anomalies = conn.execute("SELECT COUNT(*) FROM sessions WHERE is_anomaly=1").fetchone()[0] or 0
    return jsonify({
        "total": total,
        "cheaters": cheaters,
        "legit": total - cheaters,
        "avg_confidence": round(avg_conf * 100, 1),
        "cheat_rate": round(cheaters / total * 100, 1) if total else 0,
        "anomalies": anomalies,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)