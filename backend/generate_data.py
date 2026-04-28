import numpy as np
import pandas as pd

np.random.seed(42)
n_legit = 800
n_cheat = 200

def legit_player():
    return {
        "avg_reaction_ms":      np.random.normal(220, 40),       # human reaction ~200-300ms
        "reaction_std":         np.random.normal(45, 10),        # humans are inconsistent
        "click_accuracy":       np.random.normal(0.72, 0.1),     # ~72% headshot accuracy
        "actions_per_min":      np.random.normal(180, 30),       # normal APM
        "mouse_speed_avg":      np.random.normal(500, 100),      # pixels/sec
        "mouse_speed_std":      np.random.normal(200, 50),       # natural variation
        "snap_events":          np.random.poisson(2),            # rare snapping to target
        "win_rate":             np.random.beta(3, 4),            # realistic win rate
        "kill_death_ratio":     np.random.normal(1.1, 0.4),
        "headshot_ratio":       np.random.normal(0.35, 0.1),
        "session_hours":        np.random.normal(2.5, 1),
        "movement_entropy":     np.random.normal(0.78, 0.08),    # chaotic = human
        "label": 0
    }

def cheater():
    cheat_type = np.random.choice(["aimbot", "wallhack", "speedhack"])
    base = {
        "avg_reaction_ms":   np.random.normal(80, 15)   if cheat_type=="aimbot" else np.random.normal(210, 40),
        "reaction_std":      np.random.normal(8, 3)     if cheat_type=="aimbot" else np.random.normal(42, 10),
        "click_accuracy":    np.random.normal(0.97, 0.02) if cheat_type=="aimbot" else np.random.normal(0.74, 0.1),
        "actions_per_min":   np.random.normal(600, 50)  if cheat_type=="speedhack" else np.random.normal(185, 30),
        "mouse_speed_avg":   np.random.normal(1800, 100) if cheat_type=="aimbot" else np.random.normal(510, 100),
        "mouse_speed_std":   np.random.normal(20, 5)    if cheat_type=="aimbot" else np.random.normal(195, 50),
        "snap_events":       np.random.poisson(40)      if cheat_type=="aimbot" else np.random.poisson(3),
        "win_rate":          np.random.beta(9, 1),
        "kill_death_ratio":  np.random.normal(8.5, 1.5) if cheat_type=="aimbot" else np.random.normal(3.2, 0.8),
        "headshot_ratio":    np.random.normal(0.92, 0.04) if cheat_type=="aimbot" else np.random.normal(0.38, 0.1),
        "session_hours":     np.random.normal(6, 1.5),
        "movement_entropy":  np.random.normal(0.15, 0.05) if cheat_type=="aimbot" else np.random.normal(0.75, 0.08),
        "label": 1
    }
    return base

rows = [legit_player() for _ in range(n_legit)] + [cheater() for _ in range(n_cheat)]
df = pd.DataFrame(rows)

# Clip to realistic bounds
df["click_accuracy"]   = df["click_accuracy"].clip(0, 1)
df["headshot_ratio"]   = df["headshot_ratio"].clip(0, 1)
df["win_rate"]         = df["win_rate"].clip(0, 1)
df["avg_reaction_ms"]  = df["avg_reaction_ms"].clip(30, 500)
df["movement_entropy"] = df["movement_entropy"].clip(0, 1)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("backend/data.csv", index=False)
print(f"✅ Dataset generated: {len(df)} players | {df['label'].sum()} cheaters")