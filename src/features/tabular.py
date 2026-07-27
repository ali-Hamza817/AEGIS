"""
src/features/tabular.py
========================
Tabular feature engineering for AEGIS.

Constructs feature matrices from DuckDB queries for agent training
and baseline model training. Handles:
    - ERA5 climate features (rolling sums, anomalies)
    - Static land cover + topography features
    - Cross-features (e.g., impervious × precip)

Returns numpy arrays ready for LightGBM/XGBoost.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ERA5_COLUMNS = ["tp_mm", "precip_7d_sum", "precip_30d_anom", "t2m_c", "ssrd_mj"]
STATIC_COLUMNS = ["land_cover", "slope_deg", "impervious_frac", "elevation_m"]
SAR_COLUMNS = ["sar_vv_db", "sar_vh_db", "water_index_sar", "ndwi", "ndvi"]
AQ_COLUMNS = ["pm25_ug", "no2_ppb", "rh_pct", "rain_gauge_mm"]


def build_feature_matrix(
    conn: Any,
    cell_ids: list[int] | None = None,
    target_date: date | None = None,
    feature_groups: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a concatenated feature matrix from DuckDB for training.

    Args:
        conn           : DuckDB connection.
        cell_ids       : Optional subset of cell IDs.
        target_date    : If specified, only this date. Otherwise all dates.
        feature_groups : Subset of ['era5', 'sar', 'static', 'aq'].
                         Default: all.

    Returns:
        X      : np.ndarray of shape (n_samples, n_features)
        columns: list of feature names
    """
    groups = feature_groups or ["era5", "sar", "static", "aq"]
    columns: list[str] = []
    feature_arrays: list[np.ndarray] = []

    # Query base: cell_id + date combos from era5_daily
    where_parts = []
    params: list[Any] = []
    if cell_ids:
        placeholders = ",".join(["?" for _ in cell_ids])
        where_parts.append(f"e.cell_id IN ({placeholders})")
        params.extend(cell_ids)
    if target_date:
        where_parts.append("e.date = ?")
        params.append(target_date)

    where_clause = " AND ".join(where_parts) if where_parts else "1=1"

    if "era5" in groups:
        query = f"""
            SELECT e.cell_id, e.date, {', '.join(f'e.{c}' for c in ERA5_COLUMNS)}
            FROM era5_daily e WHERE {where_clause}
            ORDER BY e.cell_id, e.date
        """
        rows = conn.execute(query, params).fetchall()
        if rows:
            era5_data = np.array([[float(v or 0.0) for v in row[2:]] for row in rows])
            feature_arrays.append(era5_data)
            columns.extend(ERA5_COLUMNS)

    if "static" in groups:
        query = f"""
            SELECT DISTINCT e.cell_id, g.land_cover, g.slope_deg, g.impervious_frac, g.elevation_m
            FROM era5_daily e
            JOIN site_grid g ON e.cell_id = g.cell_id
            WHERE {where_clause}
            ORDER BY e.cell_id
        """
        rows = conn.execute(query, params).fetchall()
        if rows:
            # Repeat static features for each date
            n_per_cell = len(feature_arrays[0]) // len(rows) if feature_arrays else 1
            static_data = np.array([[float(v or 0.0) for v in row[1:]] for row in rows])
            static_data = np.repeat(static_data, n_per_cell, axis=0)
            if feature_arrays and len(static_data) == len(feature_arrays[0]):
                feature_arrays.append(static_data)
                columns.extend(STATIC_COLUMNS)

    if feature_arrays:
        X = np.hstack(feature_arrays)
    else:
        X = np.empty((0, 0))

    return X, columns


def build_cross_features(
    base_features: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Add cross-features (interaction terms) to the feature matrix.

    Cross-features:
        - impervious × precip_7d_sum  (runoff proxy)
        - slope × precip_7d_sum       (slope-drainage interaction)
    """
    cross_cols = []
    cross_data = []

    imp_idx = feature_names.index("impervious_frac") if "impervious_frac" in feature_names else None
    p7d_idx = feature_names.index("precip_7d_sum") if "precip_7d_sum" in feature_names else None
    slope_idx = feature_names.index("slope_deg") if "slope_deg" in feature_names else None

    if imp_idx is not None and p7d_idx is not None:
        cross_data.append(base_features[:, imp_idx] * base_features[:, p7d_idx])
        cross_cols.append("impervious_x_precip7d")

    if slope_idx is not None and p7d_idx is not None:
        cross_data.append(base_features[:, slope_idx] * base_features[:, p7d_idx])
        cross_cols.append("slope_x_precip7d")

    if cross_data:
        cross_arr = np.column_stack(cross_data)
        X = np.hstack([base_features, cross_arr])
        return X, feature_names + cross_cols

    return base_features, feature_names
