"""
CheatShield AI — Data Generation Entry Point
==============================================
Supports two modes:
  --mode pubg      (default) ETL from real PUBG Kaggle dataset
  --mode synthetic           Generate synthetic training data (fallback)
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys


# ── Synthetic Generator (Fallback) ───────────────────────────────────────────

def generate_synthetic(output_path: str, n_legit: int = 800, n_cheat: int = 200):
    """Generate synthetic player sessions with realistic distributions."""
    np.random.seed(42)

    def legit_player():
        return {
            "avg_reaction_ms":      np.random.normal(220, 40),
            "reaction_std":         np.random.normal(45, 10),
            "click_accuracy":       np.random.normal(0.72, 0.1),
            "actions_per_min":      np.random.normal(180, 30),
            "mouse_speed_avg":      np.random.normal(500, 100),
            "mouse_speed_std":      np.random.normal(200, 50),
            "snap_events":          np.random.poisson(2),
            "win_rate":             np.random.beta(3, 4),
            "kill_death_ratio":     np.random.normal(1.1, 0.4),
            "headshot_ratio":       np.random.normal(0.35, 0.1),
            "session_hours":        np.random.normal(2.5, 1),
            "movement_entropy":     np.random.normal(0.78, 0.08),
            "label": 0
        }

    def cheater():
        cheat_type = np.random.choice(["aimbot", "wallhack", "speedhack"])
        return {
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

    rows = [legit_player() for _ in range(n_legit)] + [cheater() for _ in range(n_cheat)]
    df = pd.DataFrame(rows)

    # Clip to realistic bounds
    df["click_accuracy"]   = df["click_accuracy"].clip(0, 1)
    df["headshot_ratio"]   = df["headshot_ratio"].clip(0, 1)
    df["win_rate"]         = df["win_rate"].clip(0, 1)
    df["avg_reaction_ms"]  = df["avg_reaction_ms"].clip(30, 500)
    df["movement_entropy"] = df["movement_entropy"].clip(0, 1)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic dataset generated: {len(df)} players | {df['label'].sum()} cheaters")
    return df


# ── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CheatShield Data Generator")
    parser.add_argument("--mode", choices=["pubg", "synthetic"], default="pubg",
                        help="Data generation mode (default: pubg)")
    parser.add_argument("--size", type=int, default=5000,
                        help="Target dataset size (default: 5000)")
    parser.add_argument("--cheat-ratio", type=float, default=0.20,
                        help="Fraction of cheaters (default: 0.20)")
    parser.add_argument("--sample-raw", type=int, default=200000,
                        help="[pubg mode] Sample N rows from raw CSV (default: 200000)")

    args = parser.parse_args()
    output_path = os.path.join(os.path.dirname(__file__), "data.csv")

    if args.mode == "pubg":
        from etl_pubg import run_etl
        run_etl(output_path=output_path, target_size=args.size,
                cheat_ratio=args.cheat_ratio, sample_raw=args.sample_raw)
    else:
        n_cheat = int(args.size * args.cheat_ratio)
        n_legit = args.size - n_cheat
        generate_synthetic(output_path, n_legit=n_legit, n_cheat=n_cheat)