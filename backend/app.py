from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib, sqlite3, json
import numpy as np
import os

app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")
CORS(app)

# Load everything
scaler   = joblib.load("backend/models/scaler.pkl")
ensemble = joblib.load("backend/models/ensemble.pkl")
kmeans   = joblib.load("backend/models/kmeans.pkl")
pca      = joblib.load("backend/models/pca.pkl")
features = joblib.load("backend/models/features.pkl")

ARCHETYPES = {0: "🐢 Casual Noob", 1: "⚔️ Aggressive Rusher", 2: "🎯 Precision Pro", 3: "👻 Suspicious Player"}

# DB setup
conn = sqlite3.connect("backend/sessions.db", check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    features TEXT, prediction INTEGER,
    confidence REAL, archetype TEXT, pca_x REAL, pca_y REAL
)""")
conn.commit()

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    x_raw = np.array([[data[f] for f in features]])
    x_sc  = scaler.transform(x_raw)

    pred       = int(ensemble.predict(x_sc)[0])
    confidence = float(ensemble.predict_proba(x_sc)[0][pred])
    archetype  = ARCHETYPES[int(kmeans.predict(x_sc)[0])]
    pca_coords = pca.transform(x_sc)[0].tolist()

    conn.execute(
        "INSERT INTO sessions (features, prediction, confidence, archetype, pca_x, pca_y) VALUES (?,?,?,?,?,?)",
        (json.dumps(data), pred, confidence, archetype, pca_coords[0], pca_coords[1])
    )
    conn.commit()

    return jsonify({
        "prediction":  "CHEATER 🚨" if pred == 1 else "LEGIT ✅",
        "label":       pred,
        "confidence":  round(confidence * 100, 2),
        "archetype":   archetype,
        "pca":         pca_coords,
        "risk_score":  round(confidence * 100 if pred == 1 else (1 - confidence) * 100, 1)
    })

@app.route("/history", methods=["GET"])
def history():
    rows = conn.execute(
        "SELECT timestamp, prediction, confidence, archetype, pca_x, pca_y FROM sessions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return jsonify([{
        "timestamp": r[0], "prediction": r[1], "confidence": round(r[2]*100,1),
        "archetype": r[3], "pca_x": r[4], "pca_y": r[5]
    } for r in rows])

@app.route("/stats", methods=["GET"])
def stats():
    total    = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    cheaters = conn.execute("SELECT COUNT(*) FROM sessions WHERE prediction=1").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM sessions").fetchone()[0] or 0
    return jsonify({
        "total": total, "cheaters": cheaters,
        "legit": total - cheaters,
        "avg_confidence": round(avg_conf * 100, 1),
        "cheat_rate": round(cheaters / total * 100, 1) if total else 0
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)