# 🛡️ CheatShield AI

> **ML-Powered Anti-Cheat Detection Dashboard** — Real-time behavioral analysis for competitive gaming using a 5-model ensemble and advanced visualization.

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.0+-ff6384?logo=chartdotjs&logoColor=white)](https://www.chartjs.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 🎯 What is CheatShield?

CheatShield AI is a **cybersecurity-grade anti-cheat dashboard** that uses machine learning to detect suspicious player behavior in real-time. It analyzes 12 behavioral metrics from gaming sessions and classifies players as **Legit** or **Cheater** with confidence scores, player archetypes, and risk levels.

Built with a **glassmorphism UI**, **animated particle effects**, **3 switchable neon themes**, and **interactive 3D profile cards** — this isn't your average ML demo.

---

## ✨ Features

### 🧠 Machine Learning Pipeline
- **5-Model Voting Ensemble**: Logistic Regression, SVM (RBF), Random Forest, MLP Neural Network, Linear Discriminant Analysis
- **K-Means Clustering**: 4 player archetypes (Casual Noob, Aggressive Rusher, Precision Pro, Suspicious Player)
- **DBSCAN Anomaly Detection**: Unsupervised outlier identification
- **PCA Visualization**: 2D player space projection

### 🎮 12 Behavioral Metrics
| Metric | Range | Description |
|---|---|---|
| Avg Reaction (ms) | 30–500 | Average reaction time |
| Reaction StdDev | 1–100 | Consistency of reactions |
| Click Accuracy | 0–1 | Hit accuracy ratio |
| Actions/Min | 50–700 | APM (actions per minute) |
| Mouse Speed (px/s) | 100–2000 | Average cursor velocity |
| Mouse Speed StdDev | 5–400 | Cursor speed variation |
| Snap Events | 0–80 | Sudden aim snaps |
| Win Rate | 0–1 | Session win percentage |
| K/D Ratio | 0.1–15 | Kill/death ratio |
| Headshot Ratio | 0–1 | Headshot percentage |
| Session Hours | 0.5–12 | Play session duration |
| Movement Entropy | 0–1 | Movement randomness |

### 🎨 Premium Frontend
- **Glassmorphism Design** — Frosted-glass cards with `backdrop-filter` blur
- **Animated Dot Grid** — CSS-animated background with scan-line overlay
- **Particle System** — Canvas-based floating dots with dynamic connections
- **3 Switchable Themes** — Midnight (purple), Cyber Neon (green), Blood Red
- **Radial Threat Gauge** — SVG arc with dynamic color transitions
- **3D Player Profile Cards** — CSS `perspective` tilt on hover + radar charts
- **Toast Notifications** — Slide-in alerts with auto-dismiss progress bars
- **Keyboard Shortcuts** — `Space` (scan), `R` (randomize), `T` (theme)
- **Animated Counters** — Smooth number roll-up animations
- **Button Ripple Effects** — Material-inspired click feedback

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/H8rsh100/cheatshield.git
cd cheatshield

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python backend/generate_data.py

# Train models
python backend/train.py

# Start the server
python backend/app.py
```

Open `http://localhost:5000` in your browser.

---

## 📁 Project Structure

```
cheatshield/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train.py            # ML training pipeline
│   ├── generate_data.py    # Synthetic data generator
│   ├── db.py               # Database module
│   ├── data.csv            # Training dataset (1000 sessions)
│   ├── sessions.db         # SQLite session history
│   └── models/
│       ├── ensemble.pkl    # 5-model voting classifier
│       ├── scaler.pkl      # StandardScaler
│       ├── pca.pkl         # PCA (2D projection)
│       ├── kmeans.pkl      # K-Means (4 archetypes)
│       ├── dbscan.pkl      # DBSCAN (anomaly detection)
│       └── features.pkl    # Feature name list
├── frontend/
│   └── index.html          # Single-page dashboard (HTML/CSS/JS)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the dashboard |
| `POST` | `/predict` | Analyze a player session |
| `GET` | `/history` | Get recent session history |
| `GET` | `/stats` | Get aggregate statistics |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "avg_reaction_ms": 80,
    "reaction_std": 5,
    "click_accuracy": 0.98,
    "actions_per_min": 190,
    "mouse_speed_avg": 1800,
    "mouse_speed_std": 15,
    "snap_events": 45,
    "win_rate": 0.95,
    "kill_death_ratio": 12.5,
    "headshot_ratio": 0.94,
    "session_hours": 6,
    "movement_entropy": 0.12
  }'
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Run analysis on current slider values |
| `R` | Randomize all slider inputs |
| `T` | Cycle through themes |

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Flask-CORS, SQLite
- **ML**: scikit-learn (Ensemble, SVM, RF, MLP, LDA, K-Means, DBSCAN, PCA)
- **Frontend**: Vanilla HTML/CSS/JS, Chart.js, Google Fonts (Inter, JetBrains Mono)
- **Design**: Glassmorphism, CSS Custom Properties, Canvas Particles, SVG Animations

---

## 📊 Data Generation

The synthetic dataset contains **1,000 player sessions**:
- **800 Legit Players** — Normal distributions matching real human gameplay
- **200 Cheaters** — Distributed across 3 cheat types:
  - **Aimbot**: Inhuman reaction times, near-perfect accuracy, high snap events
  - **Wallhack**: Abnormally high win rates and K/D ratios
  - **Speedhack**: Extreme APM, extended sessions

---

## 📝 License

MIT License — feel free to use, modify, and distribute.

---

<p align="center">
  Built with 🛡️ by <strong>CheatShield AI</strong>
</p>
