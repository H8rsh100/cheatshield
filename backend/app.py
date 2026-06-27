"""
CheatShield AI — Flask API Server
===================================
Serves the anti-cheat dashboard and provides ML prediction endpoints.
Uses a 5-model voting ensemble, K-Means archetypes, DBSCAN anomaly
detection, and PCA visualization.

Database: SQLAlchemy ORM with connection pooling and thread-safe sessions.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import json
import numpy as np
import os

from backend.models_db import (
    init_db, get_session, close_session,
    ScanSession, ModelMetadata
)

app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")
CORS(app)

# ── Initialize Database ─────────────────────────────────────────────────────

init_db()

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


def _dbscan_is_anomaly(x_scaled: np.ndarray) -> bool:
    """
    Check if a sample is an anomaly using the trained DBSCAN model.
    A point is anomalous if its nearest core sample distance exceeds eps.
    """
    if not hasattr(dbscan, 'components_') or len(dbscan.components_) == 0:
        return False
    distances = np.linalg.norm(dbscan.components_ - x_scaled, axis=1)
    min_dist = distances.min()
    return bool(min_dist > dbscan.eps)


# ── Teardown: clean up DB sessions after each request ────────────────────────

@app.teardown_appcontext
def shutdown_session(exception=None):
    close_session()


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

    # Persist via SQLAlchemy ORM
    db = get_session()
    scan = ScanSession(
        features=json.dumps(data),
        prediction=pred,
        confidence=confidence,
        archetype=archetype,
        pca_x=pca_coords[0],
        pca_y=pca_coords[1],
        is_anomaly=is_anomaly,
        risk_score=round(risk_score, 1),
    )
    db.add(scan)
    db.commit()

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
    db = get_session()
    rows = (db.query(ScanSession)
              .order_by(ScanSession.id.desc())
              .limit(50)
              .all())
    return jsonify([r.to_dict() for r in rows])


@app.route("/stats", methods=["GET"])
def stats():
    db = get_session()
    total     = db.query(ScanSession).count()
    cheaters  = db.query(ScanSession).filter(ScanSession.prediction == 1).count()
    anomalies = db.query(ScanSession).filter(ScanSession.is_anomaly == True).count()

    from sqlalchemy import func as sqlfunc
    avg_conf_row = db.query(sqlfunc.avg(ScanSession.confidence)).scalar() or 0

    return jsonify({
        "total":          total,
        "cheaters":       cheaters,
        "legit":          total - cheaters,
        "avg_confidence": round(float(avg_conf_row) * 100, 1),
        "cheat_rate":     round(cheaters / total * 100, 1) if total else 0,
        "anomalies":      anomalies,
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return active model metadata and training metrics."""
    db = get_session()
    active_model = (db.query(ModelMetadata)
                      .filter(ModelMetadata.is_active == True)
                      .first())

    response = {
        "model_loaded": True,
        "features": features,
        "metrics": model_metrics,
    }
    if active_model:
        response["active_model"] = active_model.to_dict()

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)