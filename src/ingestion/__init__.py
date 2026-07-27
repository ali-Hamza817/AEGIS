"""
src/ingestion/__init__.py
"""
from .duckdb_schema import initialise_db, get_conn
from .synthetic_generator import SyntheticGenerator

__all__ = ["initialise_db", "get_conn", "SyntheticGenerator"]
