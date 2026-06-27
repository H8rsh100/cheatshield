"""
CheatShield AI — Database Migration Script
============================================
Initializes the database schema and seeds model metadata
from the latest training metrics.
"""

import json
import os
import sys
from datetime import datetime

from models_db import init_db, get_session, close_session, ModelMetadata


def migrate():
    """Run database migrations and seed initial data."""
    print("🔄 Running database migrations...")

    # Create all tables
    init_db()
    print("   ✅ Tables created (scan_sessions, model_metadata)")

    # Seed model metadata from metrics.json if available
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        session = get_session()
        try:
            # Check if this model version already exists
            existing = session.query(ModelMetadata).filter_by(
                dataset_hash=metrics.get("dataset_hash")
            ).first()

            if existing:
                print(f"   ⏭️  Model version {metrics['dataset_hash']} already registered")
            else:
                # Deactivate previous models
                session.query(ModelMetadata).update({"is_active": False})

                # Insert new model metadata
                model = ModelMetadata(
                    dataset_hash=metrics.get("dataset_hash"),
                    dataset_rows=metrics.get("dataset_rows"),
                    accuracy=metrics.get("test_accuracy"),
                    precision=metrics.get("test_precision"),
                    recall=metrics.get("test_recall"),
                    f1_score=metrics.get("test_f1"),
                    is_active=True,
                    notes=f"Trained on PUBG dataset ({metrics.get('dataset_rows', '?')} rows)"
                )
                session.add(model)
                session.commit()
                print(f"   ✅ Model metadata seeded: accuracy={model.accuracy:.4f}, "
                      f"F1={model.f1_score:.4f}")

        except Exception as e:
            session.rollback()
            print(f"   ❌ Error seeding metadata: {e}")
            raise
        finally:
            close_session()
    else:
        print("   ⚠️  No metrics.json found — skipping model metadata seed")

    print("✅ Migration complete!")


if __name__ == "__main__":
    migrate()
