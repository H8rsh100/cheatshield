# CheatShield — Data Directory

## Source Dataset

**PUBG Finish Placement Prediction** — Kaggle  
https://www.kaggle.com/c/pubg-finish-placement-prediction

- ~4.4 million match records from PlayerUnknown's Battlegrounds
- 29 columns of real player match statistics
- License: Kaggle Competition (non-commercial research)

## Setup

1. Download `train_V2.csv` from the Kaggle competition page
2. Place it in `backend/raw_data/train_V2.csv`
3. Run the ETL pipeline:

```bash
python backend/generate_data.py --mode pubg
```

## ETL Feature Mapping

| CheatShield Metric   | PUBG Source                              | Transform                                 |
|-----------------------|------------------------------------------|--------------------------------------------|
| `avg_reaction_ms`     | `matchDuration`, `kills`, `assists`      | Time per kill action × 10                  |
| `reaction_std`        | `killStreaks`                             | Inverse streak consistency                 |
| `click_accuracy`      | `headshotKills`, `kills`                 | Headshot ratio                             |
| `actions_per_min`     | `kills + assists + revives + boosts + heals + weaponsAcquired` | Composite APM    |
| `mouse_speed_avg`     | `walkDistance + rideDistance + swimDistance` | Movement velocity × 100                 |
| `mouse_speed_std`     | `walkDistance` fraction                  | Walk-dominant variance                     |
| `snap_events`         | `killStreaks`, `longestKill`              | Combined snap heuristic                    |
| `win_rate`            | `winPlacePerc`                           | Direct                                     |
| `kill_death_ratio`    | `kills`, `DBNOs`                         | Direct ratio                               |
| `headshot_ratio`      | `headshotKills`, `kills`                 | Direct ratio                               |
| `session_hours`       | `matchDuration`                          | Seconds → hours                            |
| `movement_entropy`    | `walkDistance`, `rideDistance`, `swimDistance` | Shannon entropy (normalized)          |

## Cheater Labeling Rules

Players are flagged as cheaters if any of the following anomalies are detected:

| Rule           | Condition                                          |
|----------------|----------------------------------------------------|
| Kill count     | `kills > 30` in a single match                    |
| Kill distance  | `longestKill > 500m`                               |
| Headshot ratio | `headshotKills/kills > 0.90` AND `kills >= 10`     |
| Teleport hack  | `walkDistance < 10m` AND `kills >= 5`              |
| Speed hack     | `walkDistance > 10km` AND `matchDuration < 20min`  |

## Fallback: Synthetic Data

If you don't have the PUBG dataset, use synthetic mode:

```bash
python backend/generate_data.py --mode synthetic
```
