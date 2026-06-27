"""
CheatShield AI — PUBG Kaggle Dataset ETL Pipeline
===================================================
Transforms raw PUBG match statistics (train_V2.csv) from the Kaggle
"PUBG Finish Placement Prediction" competition into CheatShield's
12-feature behavioral analysis format.

Source: https://www.kaggle.com/c/pubg-finish-placement-prediction
Input:  backend/raw_data/train_V2.csv  (~4.4M rows, 29 columns)
Output: backend/data.csv              (N rows, 12 features + label)
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

# ── Configuration ────────────────────────────────────────────────────────────

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "raw_data", "train_V2.csv")
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "data.csv")

# Cheater detection thresholds (based on PUBG community analysis)
CHEAT_RULES = {
    "kill_threshold":       30,    # > 30 kills in a single match
    "longest_kill_m":       500,   # > 500m kill distance (sniper max ~800m, most cheats snap > 500)
    "headshot_ratio_min":   0.90,  # > 90% headshot ratio with decent kills
    "headshot_kills_min":   10,    # minimum kills to flag headshot ratio
    "teleport_walk_max":    10,    # walked < 10m but got kills (teleport hack)
    "teleport_kills_min":   5,     # minimum kills for teleport flag
    "speed_hack_walk_min":  10000, # walked > 10km in a short match
    "speed_hack_duration":  1200,  # match duration < 20 min for speed flag
}

# Final output feature ranges (for clipping)
FEATURE_BOUNDS = {
    "avg_reaction_ms":   (30,   500),
    "reaction_std":      (1,    100),
    "click_accuracy":    (0,    1),
    "actions_per_min":   (50,   700),
    "mouse_speed_avg":   (100,  2000),
    "mouse_speed_std":   (5,    400),
    "snap_events":       (0,    80),
    "win_rate":          (0,    1),
    "kill_death_ratio":  (0.1,  15),
    "headshot_ratio":    (0,    1),
    "session_hours":     (0.5,  12),
    "movement_entropy":  (0,    1),
}


def load_raw_data(path: str, sample_size: int = None) -> pd.DataFrame:
    """Load raw PUBG CSV, optionally sampling for memory efficiency."""
    print(f"📂 Loading raw data from: {path}")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        print("   Download train_V2.csv from:")
        print("   https://www.kaggle.com/c/pubg-finish-placement-prediction/data")
        sys.exit(1)

    # Read in chunks for the massive file
    if sample_size:
        # Count lines first for proportional sampling
        total_lines = sum(1 for _ in open(path, encoding="utf-8")) - 1
        skip_ratio = max(1, total_lines // (sample_size * 3))  # oversample 3x
        df = pd.read_csv(
            path,
            skiprows=lambda i: i > 0 and i % skip_ratio != 0,
            low_memory=False
        )
        print(f"   Sampled {len(df):,} rows from {total_lines:,} total")
    else:
        df = pd.read_csv(path, low_memory=False)
        print(f"   Loaded {len(df):,} rows")

    # Drop rows with NaN in critical columns
    critical_cols = ["kills", "damageDealt", "walkDistance", "winPlacePerc",
                     "headshotKills", "matchDuration"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    if len(df) < before:
        print(f"   Dropped {before - len(df)} rows with NaN in critical columns")

    return df


def flag_cheaters(df: pd.DataFrame) -> pd.Series:
    """
    Apply anomaly-based cheater detection rules.
    Returns a binary Series: 1 = cheater, 0 = legit.
    """
    rules = CHEAT_RULES

    is_cheater = (
        # Rule 1: Absurd kill count
        (df["kills"] > rules["kill_threshold"]) |

        # Rule 2: Impossible kill distance
        (df["longestKill"] > rules["longest_kill_m"]) |

        # Rule 3: Inhuman headshot accuracy with significant kills
        ((df["headshotKills"] / df["kills"].clip(lower=1)) > rules["headshot_ratio_min"]) &
        (df["kills"] >= rules["headshot_kills_min"]) |

        # Rule 4: Teleport hack — barely moved but got kills
        (df["walkDistance"] < rules["teleport_walk_max"]) &
        (df["kills"] >= rules["teleport_kills_min"]) |

        # Rule 5: Speed hack — absurd distance in short match
        (df["walkDistance"] > rules["speed_hack_walk_min"]) &
        (df["matchDuration"] < rules["speed_hack_duration"])
    )

    return is_cheater.astype(int)


def shannon_entropy(walk: float, ride: float, swim: float) -> float:
    """Compute Shannon entropy of movement distribution."""
    total = walk + ride + swim
    if total == 0:
        return 0.0
    probs = np.array([walk, ride, swim]) / total
    probs = probs[probs > 0]  # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)) / np.log2(3))  # normalized to [0, 1]


def _scale_to_range(series: pd.Series, target_min: float, target_max: float,
                     clip_percentile: float = 99) -> pd.Series:
    """Scale a series to a target range using percentile-based normalization."""
    lo = series.quantile(1 - clip_percentile / 100)
    hi = series.quantile(clip_percentile / 100)
    if hi == lo:
        return pd.Series(np.full(len(series), (target_min + target_max) / 2))
    normalized = (series - lo) / (hi - lo)
    return target_min + normalized.clip(0, 1) * (target_max - target_min)


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw PUBG columns into CheatShield's 12 behavioral metrics.
    Uses percentile-based scaling so features fill their intended ranges.
    """
    print("🔄 Transforming features...")
    np.random.seed(42)

    # Avoid division by zero
    kills_safe     = df["kills"].clip(lower=1)
    time_safe_s    = df["matchDuration"].clip(lower=60)   # seconds
    time_safe_min  = time_safe_s / 60

    out = pd.DataFrame()

    # 1. avg_reaction_ms — proxy: time per kill action (inverted — more kills = faster)
    raw_reaction = time_safe_s / (df["kills"] + df["assists"] + 1)
    out["avg_reaction_ms"] = _scale_to_range(raw_reaction, 30, 500)

    # 2. reaction_std — kill streak consistency + noise
    raw_std = (1 / df["killStreaks"].clip(lower=1)) + np.random.normal(0, 0.05, len(df))
    out["reaction_std"] = _scale_to_range(raw_std, 1, 100)

    # 3. click_accuracy — damage efficiency: damageDealt / (kills * 100)
    raw_accuracy = df["damageDealt"] / (kills_safe * 150).clip(lower=1)
    out["click_accuracy"] = raw_accuracy.clip(0, 1)

    # 4. actions_per_min — composite APM (scaled up to fill range)
    total_actions = (df["kills"] + df["assists"] + df["revives"] +
                     df["boosts"] + df["heals"] + df["weaponsAcquired"])
    raw_apm = total_actions / time_safe_min
    out["actions_per_min"] = _scale_to_range(raw_apm, 50, 700)

    # 5. mouse_speed_avg — movement velocity proxy
    total_distance = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    raw_speed = total_distance / time_safe_s
    out["mouse_speed_avg"] = _scale_to_range(raw_speed, 100, 2000)

    # 6. mouse_speed_std — movement variance + noise
    walk_frac = df["walkDistance"] / total_distance.clip(lower=1)
    raw_speed_std = walk_frac + np.random.normal(0, 0.05, len(df))
    out["mouse_speed_std"] = _scale_to_range(raw_speed_std, 5, 400)

    # 7. snap_events — sudden aim snaps
    q99_longest = max(df["longestKill"].quantile(0.99), 1.0)
    longest_kill_norm = df["longestKill"] / q99_longest
    raw_snaps = df["killStreaks"] * 3 + longest_kill_norm * 15
    out["snap_events"] = _scale_to_range(raw_snaps, 0, 80)

    # 8. win_rate — direct from winPlacePerc
    out["win_rate"] = df["winPlacePerc"].clip(0, 1)

    # 9. kill_death_ratio — kills / DBNOs
    raw_kd = df["kills"] / df["DBNOs"].clip(lower=1)
    out["kill_death_ratio"] = _scale_to_range(raw_kd, 0.1, 15)

    # 10. headshot_ratio — direct
    out["headshot_ratio"] = (df["headshotKills"] / kills_safe).clip(0, 1)

    # 11. session_hours — match duration in hours (scaled to fill range)
    out["session_hours"] = _scale_to_range(time_safe_s / 3600, 0.5, 12)

    # 12. movement_entropy — Shannon entropy of movement distribution
    out["movement_entropy"] = [
        shannon_entropy(w, r, s)
        for w, r, s in zip(df["walkDistance"], df["rideDistance"], df["swimDistance"])
    ]

    return out


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip all features to their defined realistic bounds."""
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def balance_dataset(df: pd.DataFrame, target_size: int = 5000,
                    cheat_ratio: float = 0.20) -> pd.DataFrame:
    """
    Balance the dataset to a target size with specified cheat ratio.
    Undersamples the majority class, oversamples minority if needed.
    """
    n_cheaters = int(target_size * cheat_ratio)
    n_legit    = target_size - n_cheaters

    cheaters = df[df["label"] == 1]
    legits   = df[df["label"] == 0]

    print(f"   Raw distribution: {len(legits):,} legit, {len(cheaters):,} cheaters "
          f"({len(cheaters)/len(df)*100:.1f}% cheat rate)")

    # Sample with replacement if needed
    cheater_sample = cheaters.sample(n=min(n_cheaters, len(cheaters)),
                                     random_state=42, replace=len(cheaters) < n_cheaters)
    legit_sample   = legits.sample(n=min(n_legit, len(legits)),
                                   random_state=42, replace=len(legits) < n_legit)

    balanced = pd.concat([legit_sample, cheater_sample], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"   Balanced to: {len(balanced)} rows "
          f"({len(cheater_sample)} cheaters / {len(legit_sample)} legit)")

    return balanced


def run_etl(raw_path: str = RAW_DATA_PATH, output_path: str = OUTPUT_PATH,
            target_size: int = 5000, cheat_ratio: float = 0.20,
            sample_raw: int = None):
    """Execute the full ETL pipeline."""
    print("=" * 65)
    print("  CheatShield AI — PUBG Dataset ETL Pipeline")
    print("=" * 65)

    # 1. Load raw data
    df = load_raw_data(raw_path, sample_size=sample_raw)

    # 2. Flag cheaters
    print("\n🔍 Applying cheater detection rules...")
    df["label"] = flag_cheaters(df)
    cheat_count = df["label"].sum()
    print(f"   Flagged {cheat_count:,} cheaters out of {len(df):,} "
          f"({cheat_count/len(df)*100:.2f}%)")

    # 3. Transform features
    features = transform_features(df)
    features["label"] = df["label"].values

    # 4. Clip to realistic bounds
    features = clip_features(features)

    # 5. Drop any remaining NaN/inf
    before = len(features)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < before:
        print(f"   Cleaned {before - len(features)} rows with inf/NaN after transform")

    # 6. Balance dataset
    print(f"\n⚖️  Balancing dataset to {target_size} rows...")
    features = balance_dataset(features, target_size=target_size,
                                cheat_ratio=cheat_ratio)

    # 7. Save
    features.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Shape: {features.shape}")
    print(f"   Features: {[c for c in features.columns if c != 'label']}")

    # 8. Validation report
    print("\n📊 Validation Report:")
    print(f"   Rows: {len(features)}")
    print(f"   Cheaters: {features['label'].sum()} ({features['label'].mean()*100:.1f}%)")
    print(f"   Legit: {(features['label'] == 0).sum()}")
    print(f"   NaN count: {features.isna().sum().sum()}")
    print(f"   Inf count: {np.isinf(features.select_dtypes(include=[np.number])).sum().sum()}")
    print("\n   Feature Ranges:")
    for col in features.columns:
        if col != "label":
            print(f"     {col:25s}  [{features[col].min():.2f}, {features[col].max():.2f}]  "
                  f"μ={features[col].mean():.2f}")

    print("\n" + "=" * 65)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CheatShield PUBG ETL Pipeline")
    parser.add_argument("--input", default=RAW_DATA_PATH,
                        help="Path to raw train_V2.csv")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help="Path to output data.csv")
    parser.add_argument("--size", type=int, default=5000,
                        help="Target dataset size (default: 5000)")
    parser.add_argument("--cheat-ratio", type=float, default=0.20,
                        help="Fraction of cheaters (default: 0.20)")
    parser.add_argument("--sample-raw", type=int, default=200000,
                        help="Sample N rows from raw CSV for speed (default: 200000)")

    args = parser.parse_args()
    run_etl(args.input, args.output, args.size, args.cheat_ratio, args.sample_raw)
