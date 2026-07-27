"""
src/tiles/tiling.py
====================
AOI Tiling — generates a regular grid of 100m × 100m cells
over a bounding box for the Brisbane study area.

Grid cell IDs are sequential integers. Each cell has:
    - cell_id  (int)
    - centroid (lon, lat)
    - bbox WKT polygon

Reference CRS: EPSG:4326 (WGS84). 100m at Brisbane latitude (~27.5°S)
≈ 0.0009° lon, 0.0009° lat.
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

# Approximate degrees per 100m at Brisbane latitude
DEG_PER_100M_LON = 0.00099
DEG_PER_100M_LAT = 0.00090


def generate_grid(
    bbox: tuple[float, float, float, float],
    cell_size_deg: float | None = None,
    max_cells: int = 2500,
) -> Generator[dict, None, None]:
    """
    Generate a regular grid of rectangular cells over a bounding box.

    Args:
        bbox          : (min_lon, min_lat, max_lon, max_lat) in WGS84.
        cell_size_deg : Override cell size in degrees. Default uses 100m.
        max_cells     : Cap total cell count (memory safety).

    Yields:
        dict with keys: cell_id, centroid_lon, centroid_lat, bbox_wkt
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    dx = cell_size_deg or DEG_PER_100M_LON
    dy = cell_size_deg or DEG_PER_100M_LAT

    n_cols = int((max_lon - min_lon) / dx)
    n_rows = int((max_lat - min_lat) / dy)

    total = n_cols * n_rows
    if total > max_cells:
        # Sub-sample: increase cell size to fit within max_cells
        factor = np.sqrt(total / max_cells)
        dx *= factor
        dy *= factor
        n_cols = int((max_lon - min_lon) / dx)
        n_rows = int((max_lat - min_lat) / dy)
        logger.warning(
            "Grid exceeded max_cells=%d (%d cells). Resampled to %d × %d = %d.",
            max_cells, total, n_cols, n_rows, n_cols * n_rows,
        )

    cell_id = 0
    for row in range(n_rows):
        for col in range(n_cols):
            lon0 = min_lon + col * dx
            lat0 = min_lat + row * dy
            lon1 = lon0 + dx
            lat1 = lat0 + dy
            yield {
                "cell_id": cell_id,
                "centroid_lon": (lon0 + lon1) / 2,
                "centroid_lat": (lat0 + lat1) / 2,
                "bbox_wkt": (
                    f"POLYGON(({lon0:.6f} {lat0:.6f}, {lon1:.6f} {lat0:.6f}, "
                    f"{lon1:.6f} {lat1:.6f}, {lon0:.6f} {lat1:.6f}, "
                    f"{lon0:.6f} {lat0:.6f}))"
                ),
            }
            cell_id += 1


def brisbane_grid(max_cells: int = 500) -> list[dict]:
    """Convenience: generate the Brisbane AOI grid."""
    bbox = (152.5, -28.0, 153.5, -27.0)
    return list(generate_grid(bbox, max_cells=max_cells))
