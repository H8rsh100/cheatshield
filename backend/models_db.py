"""
CheatShield AI — SQLAlchemy Database Models
=============================================
Production-grade ORM models with connection pooling, thread-safe
session management, and proper schema definitions.

Replaces raw sqlite3 calls with typed, validated ORM queries.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, Boolean,
    DateTime, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import os

# ── Engine & Session Factory ─────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "sessions.db")
DB_URL   = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,           # verify connections before use
    pool_recycle=3600,             # recycle connections after 1 hour
    connect_args={"check_same_thread": False},  # SQLite threading
    echo=False,
)

# Thread-safe scoped session
SessionFactory = sessionmaker(bind=engine)
ScopedSession  = scoped_session(SessionFactory)

Base = declarative_base()


# ── ORM Models ───────────────────────────────────────────────────────────────

class ScanSession(Base):
    """Records each player analysis scan with prediction results."""
    __tablename__ = "scan_sessions"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, server_default=func.now(), nullable=False)
    features    = Column(Text, nullable=False, doc="JSON-encoded input features")
    prediction  = Column(Integer, nullable=False, doc="0=Legit, 1=Cheater")
    confidence  = Column(Float, nullable=False, doc="Model confidence [0, 1]")
    archetype   = Column(String(64), nullable=True, doc="K-Means player archetype")
    pca_x       = Column(Float, nullable=True, doc="PCA component 1")
    pca_y       = Column(Float, nullable=True, doc="PCA component 2")
    is_anomaly  = Column(Boolean, default=False, doc="DBSCAN anomaly flag")
    risk_score  = Column(Float, default=0.0, doc="Composite risk score [0, 100]")

    def to_dict(self):
        return {
            "id":          self.id,
            "timestamp":   self.timestamp.isoformat() if self.timestamp else None,
            "prediction":  self.prediction,
            "confidence":  round(self.confidence * 100, 1) if self.confidence else 0,
            "archetype":   self.archetype,
            "pca_x":       self.pca_x,
            "pca_y":       self.pca_y,
            "is_anomaly":  bool(self.is_anomaly),
            "risk_score":  self.risk_score,
        }

    def __repr__(self):
        label = "CHEAT" if self.prediction == 1 else "LEGIT"
        return f"<ScanSession #{self.id} {label} conf={self.confidence:.2f}>"


class ModelMetadata(Base):
    """Tracks trained model versions and their performance metrics."""
    __tablename__ = "model_metadata"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trained_at      = Column(DateTime, server_default=func.now(), nullable=False)
    dataset_hash    = Column(String(32), nullable=True, doc="MD5 hash of training data")
    dataset_rows    = Column(Integer, nullable=True)
    accuracy        = Column(Float, nullable=True)
    precision       = Column(Float, nullable=True)
    recall          = Column(Float, nullable=True)
    f1_score        = Column(Float, nullable=True)
    is_active       = Column(Boolean, default=True, doc="Currently deployed model")
    notes           = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id":           self.id,
            "trained_at":   self.trained_at.isoformat() if self.trained_at else None,
            "dataset_hash": self.dataset_hash,
            "dataset_rows": self.dataset_rows,
            "accuracy":     self.accuracy,
            "precision":    self.precision,
            "recall":       self.recall,
            "f1_score":     self.f1_score,
            "is_active":    self.is_active,
        }

    def __repr__(self):
        return f"<ModelMetadata #{self.id} acc={self.accuracy:.4f} active={self.is_active}>"


# ── Table Creation ───────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


def get_session():
    """Get a thread-safe database session."""
    return ScopedSession()


def close_session():
    """Remove the current thread's session."""
    ScopedSession.remove()
