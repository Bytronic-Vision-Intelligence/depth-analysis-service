"""Steak thickness and wedge from a line profiler's height map (Gocator part mode, Dunbia).

The camera sends one height map per steak: uint16 PNG, value = round(mm * 100) + 32768,
0 = no data, the belt at 0 mm (aligned sensor). Its metadata carries the pixel size
(geometry.x_resolution_mm across the belt, y_resolution_mm along it) and when the part
passed the laser relative to the trigger (part.start_after_trigger_ms, scan_start_time).

What is measured (all in mm, on a square grid of grid_mm):
- the steak: the largest patch higher than min_height_mm, small no-data holes closed;
- the interior: the steak less a rim_mm (3) edge, where the top slopes down to the belt and
  the edge pixels are least reliable;
- thickness: M&S measures "along the back fat", the thick long edge (the lean side tapers
  naturally). The map has no colour, so both long edges are measured: a band edge_band_mm (5)
  deep just inside the rim (3-8 mm in), over the middle edge_middle_fraction (60%) of the
  steak's length, the edge_percentile (p90) of each side. The thicker side is taken as the fat
  edge: thickness_mm = its p90, thin_edge_mm = the other's. Fillet has no back fat (M&S: ruler at
  the middle), so a SKU can set thickness_method: centre, judged on centre_mm = the median of
  the steak's core;
- whole-steak stats: median, min, max, p10, p90 of the interior;
- wedge: median of each end (end_fraction of the length along the steak's long axis) and their
  difference, wedge_mm. Always reported; no spec gives a number ("wedge steaks not permitted"),
  so the limit is per SKU and off unless set.
No length, width or area: along the belt they depend on the belt speed, not known well enough.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import cv2
import numpy as np

HEIGHT_OFFSET = 32768        # uint16 value of 0 mm
HEIGHT_SCALE = 100.0         # uint16 steps per mm
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"   # verdict-service's time stamp format

# Defaults for the `steak_thickness` config section; every one can be set there.
DEFAULTS = {
    "grid_mm": 1.0,             # the map is resampled to square pixels of this size
    "min_height_mm": 3.0,       # higher than this above the belt is steak
    "min_area_mm2": 2000.0,     # a smaller patch is not a steak (debris, a finger)
    "close_mm": 3.0,            # no-data specks up to about this size inside the steak are closed
    "rim_mm": 3.0,              # the edge left out of every reading (slope, rounded fat, shadows)
    "edge_band_mm": 5.0,        # depth of the long-edge bands, inside the rim (3-8 mm in)
    "edge_percentile": 90.0,    # each edge band's reading (90 = the fat's height, not the dip beside it)
    "edge_middle_fraction": 0.6,  # the edge bands cover this middle share of the length
    "end_fraction": 0.2,        # each end is this share of the length
    "centre_fraction": 0.5,     # the centre: farther from the outline than this share of the deepest point
    "min_points": 20,           # fewest grid points for a reading (an edge or an end)
}


def settings_with_defaults(settings: Optional[dict]) -> dict:
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in (settings or {}).items() if v is not None})
    return merged


def decode_height_mm(raw: np.ndarray) -> np.ndarray:
    """uint16 height map -> float32 mm, NaN where there is no data."""
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    if raw.dtype != np.uint16:
        raise ValueError(f"a height map must be uint16, got {raw.dtype}")
    height = (raw.astype(np.float32) - HEIGHT_OFFSET) / HEIGHT_SCALE
    height[raw == 0] = np.nan
    return height


def to_square_grid(height: np.ndarray, x_res_mm: float, y_res_mm: float, grid_mm: float) -> np.ndarray:
    """Resample so one pixel is grid_mm x grid_mm. Each new pixel is the mean of the valid
    heights it covers; NaN when less than half of it had data."""
    rows, cols = height.shape
    new_cols = max(1, int(round(cols * x_res_mm / grid_mm)))
    new_rows = max(1, int(round(rows * y_res_mm / grid_mm)))
    valid = ~np.isnan(height)
    total = cv2.resize(np.where(valid, height, 0).astype(np.float32), (new_cols, new_rows), interpolation=cv2.INTER_AREA)
    share = cv2.resize(valid.astype(np.float32), (new_cols, new_rows), interpolation=cv2.INTER_AREA)
    grid = np.full(total.shape, np.nan, dtype=np.float32)
    enough = share >= 0.5
    grid[enough] = total[enough] / share[enough]
    return grid


def steak_mask(grid: np.ndarray, s: dict) -> Optional[np.ndarray]:
    """The largest patch above min_height_mm, specks closed and holes filled; None if there
    is no patch of min_area_mm2."""
    above = np.nan_to_num(grid, nan=0.0) > float(s["min_height_mm"])
    if not above.any():
        return None
    k = max(1, int(round(float(s["close_mm"]) / float(s["grid_mm"]))))
    above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area_mm2 = float(stats[largest, cv2.CC_STAT_AREA]) * float(s["grid_mm"]) ** 2
    if area_mm2 < float(s["min_area_mm2"]):
        return None
    mask = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _reading(values: np.ndarray, min_points: int, percentile: float = 50.0) -> Optional[float]:
    values = values[~np.isnan(values)]
    if values.size < min_points:
        return None
    reading = float(np.percentile(values, percentile))
    return reading


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def measure_steak(height: np.ndarray, x_res_mm: float, y_res_mm: float, settings: Optional[dict] = None,
                  thickness_method: str = "edge") -> Dict[str, Any]:
    """Measure one steak's height map (mm, NaN = no data). thickness_method (per SKU): "edge"
    (the thicker long edge, default) or "centre" (the core, for fillet).

    Returns {"measurable": False, "reason": ...} when there is no steak to measure, otherwise
    {"measurable": True, "data": {...}} with every reading in mm (None when too few points).
    """
    s = settings_with_defaults(settings)
    g = float(s["grid_mm"])
    min_points = int(s["min_points"])
    if not x_res_mm or not y_res_mm:
        result = {"measurable": False, "reason": "no pixel size in the metadata"}
        return result
    grid = to_square_grid(height, float(x_res_mm), float(y_res_mm), g)
    mask = steak_mask(grid, s)
    if mask is None:
        result = {"measurable": False, "reason": "no steak in the 3D map"}
        return result

    # distance to the outside, in mm: the rim and the edge bands
    dist_mm = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1] * g
    rim = float(s["rim_mm"])
    interior = mask & (dist_mm > rim) & ~np.isnan(grid)
    if int(interior.sum()) < min_points:
        result = {"measurable": False, "reason": "steak too small to measure"}
        return result
    values = grid[interior]

    # the steak's long axis (principal axis of its outline, in mm)
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64) * g
    centre = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    major, minor = vectors[:, 1], vectors[:, 0]
    along_all = (points - centre) @ major
    length_mm = float(along_all.max() - along_all.min())   # only to place the edge bands and ends

    iy, ix = np.nonzero(interior)
    ipoints = np.column_stack((ix, iy)).astype(np.float64) * g - centre
    along = ipoints @ major
    across = ipoints @ minor
    heights = grid[iy, ix]

    # the two long edges: a band edge_band_mm deep just inside the rim (3-8 mm in by default:
    # past the rounded edge / laser-shadow pixels, but still on the fat cap, often only 5-10 mm
    # wide), over the middle edge_middle_fraction of the length; each side's edge_percentile (90):
    # as long as part of the band is on the fat, the fat's height is read, not the dip next to it
    # or the rounded edge (office dummy steak, 2026-11-19: median read 15.9 on one face and 13.7
    # on the other; p90 17.5-18.3 on both, hand-measured 17-18)
    ey, ex = np.nonzero(interior & (dist_mm <= rim + float(s["edge_band_mm"])))
    epoints = np.column_stack((ex, ey)).astype(np.float64) * g - centre
    e_along, e_across, e_heights = epoints @ major, epoints @ minor, grid[ey, ex]
    middle = np.abs(e_along - (along_all.min() + along_all.max()) / 2) <= float(s["edge_middle_fraction"]) * length_mm / 2
    edge_q = float(s["edge_percentile"])
    edge_a = _reading(e_heights[middle & (e_across < 0)], min_points, edge_q)
    edge_b = _reading(e_heights[middle & (e_across > 0)], min_points, edge_q)
    edges = [e for e in (edge_a, edge_b) if e is not None]
    thin_edge = min(edges) if len(edges) == 2 else None

    # the centre: the core farther from the outline than centre_fraction of the deepest point
    # (a disc in a round steak, a strip along the middle of a long one)
    core = interior & (dist_mm >= float(s["centre_fraction"]) * float(dist_mm.max()))
    centre_reading = _reading(grid[core], min_points)

    # the judged thickness: "edge" = the thicker long edge, taken as the back fat (sirloin,
    # ribeye, rump, picanha: "measured along the back fat"); "centre" = the core (fillet: ruler
    # at the middle, no back fat). Falls back to the whole steak's median.
    if str(thickness_method).lower() == "centre" and centre_reading is not None:
        thickness, source = centre_reading, "centre"
    elif edges:
        thickness, source = max(edges), "edge"
    else:
        thickness, source = float(np.median(values)), "median"

    # the two ends along the long axis
    end_len = float(s["end_fraction"]) * length_mm
    end_1 = _reading(heights[along <= along_all.min() + end_len], min_points)
    end_2 = _reading(heights[along >= along_all.max() - end_len], min_points)
    wedge = abs(end_1 - end_2) if end_1 is not None and end_2 is not None else None

    data = {
        "thickness_mm": _round(thickness),                 # the judged one, see thickness_source
        "thickness_source": source,                        # edge / centre / median (fallback)
        "thin_edge_mm": _round(thin_edge),
        "centre_mm": _round(centre_reading),
        "median_mm": _round(np.median(values)),
        "min_mm": _round(np.min(values)),
        "max_mm": _round(np.max(values)),
        "p10_mm": _round(np.percentile(values, 10)),
        "p90_mm": _round(np.percentile(values, 90)),
        "wedge_mm": _round(wedge),
        "thick_end_mm": _round(max(end_1, end_2)) if wedge is not None else None,
        "thin_end_mm": _round(min(end_1, end_2)) if wedge is not None else None,
        # no length / width / area: along the belt they scale with travel_speed_mm_s, which is
        # not known well enough (no encoder), so they are left out until it is
        "valid_share": _round(float((~np.isnan(grid[mask])).mean()), 3),
    }
    result = {"measurable": True, "data": data}
    return result


def spec_range(value) -> Optional[tuple]:
    """A SKU spec limit as (low, high): [15, 25], (15, 25) or "15-25"; None when not set or
    unreadable. A missing bound is open (e.g. [null, 5] = at most 5)."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("to", "-").split("-")]
        value = parts if len(parts) == 2 else None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        low = float("-inf") if value[0] in (None, "") else float(value[0])
        high = float("inf") if value[1] in (None, "") else float(value[1])
    except (TypeError, ValueError):
        return None
    limits = (low, high)
    return limits


def judge(data: Dict[str, Any], sku_spec: dict) -> Dict[str, Any]:
    """pass/fail of the measurements against the SKU's limits: `thickness` [low, high] on
    thickness_mm and `wedge` [low, high] (normally [0, max]) on wedge_mm. A limit that is not
    set is not judged; a reading that is None against a set limit fails (it could not be
    checked)."""
    checks = {}
    reasons = []
    for name, key in (("thickness", "thickness_mm"), ("wedge", "wedge_mm")):
        limits = spec_range((sku_spec or {}).get(name))
        if limits is None:
            continue
        value = data.get(key)
        ok = value is not None and limits[0] <= value <= limits[1]
        checks[name] = {"value": value, "limits": [None if np.isinf(v) else v for v in limits], "in_range": ok}
        if not ok:
            reasons.append(name)
    verdict = {"verdict": "fail" if reasons else "pass", "reason_failed": reasons, "checks": checks}
    return verdict


def trigger_time(metadata: Optional[dict]) -> Optional[str]:
    """When camera-service took the trigger this map answers, for verdict-service's time match:
    scan_start_time (when the part reached the laser) less part.start_after_trigger_ms. None when
    the metadata does not have both."""
    metadata = metadata or {}
    start = metadata.get("scan_start_time")
    part = metadata.get("part") or {}
    offset_ms = part.get("start_after_trigger_ms")
    if start is None or offset_ms is None:
        return None
    try:
        started = datetime.fromisoformat(str(start))
    except ValueError:
        return None
    stamp = (started - timedelta(milliseconds=float(offset_ms))).strftime(TIME_FORMAT)
    return stamp
