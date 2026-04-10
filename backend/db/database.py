"""
Database configuration — SQLAlchemy engine, session, and base model.

Supports SQLite (dev) and PostgreSQL (production) via DATABASE_URL
environment variable.
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///vaaksetu.db",  # Default: SQLite for local dev
)

# Fix for SQLAlchemy + SQLite relative paths
if DATABASE_URL.startswith("sqlite:///") and not DATABASE_URL.startswith("sqlite:////"):
    import pathlib
    db_path = pathlib.Path(__file__).resolve().parent.parent / DATABASE_URL.replace("sqlite:///", "")
    DATABASE_URL = f"sqlite:///{db_path}"

# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    # SQLite-specific: enable WAL mode for better concurrency
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

# Enable WAL mode for SQLite
if "sqlite" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models."""
    pass


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def get_db() -> Session:
    """
    Get a database session. Use as a dependency or context manager.

    Usage (Flask route)::

        db = get_db()
        try:
            patients = db.query(Patient).all()
        finally:
            db.close()
    """
    db = SessionLocal()
    return db


@contextmanager
def get_db_session():
    """
    Context manager for database sessions with auto-commit/rollback.

    Usage::

        with get_db_session() as db:
            patient = Patient(name="Ravi", age=45)
            db.add(patient)
            # auto-commits on exit, auto-rollbacks on exception
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Create all database tables.

    Call this once at app startup or from a migration script.
    """
    # Import all models so they're registered with Base
    import models.patient  # noqa: F401
    import models.session  # noqa: F401
    import models.clinical_record  # noqa: F401

    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables. USE WITH CAUTION — only for dev/testing."""
    Base.metadata.drop_all(bind=engine)
