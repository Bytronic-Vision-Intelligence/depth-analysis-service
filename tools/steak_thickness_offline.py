"""Run the steak thickness / wedge measurement on stored Gocator height maps (no MQTT).

    python tools/steak_thickness_offline.py <png or folder> [count] [--since HHMMSS] [--view]

Each map needs its archive-service sidecar (<camera>_metadata/<name>.json, archive_metadata:
true) for the pixel size. Prints one line per map; --view writes <name>_thickness.png beside
it: square pixels, colour = height, the thicker long edge's band in white, the ends in grey.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from dependencies import steak_thickness as st  # noqa: E402

args = sys.argv[1:]
view = "--view" in args
args = [a for a in args if a != "--view"]
since = ""
if "--since" in args:
    k = args.index("--since")
    since = args[k + 1]
    del args[k:k + 2]
if not args:
    sys.exit(__doc__)
target = Path(args[0])
count = int(args[1]) if len(args) > 1 else 1000

pngs = [target] if target.is_file() else sorted(
    p for p in target.rglob("*.png") if not p.stem.endswith(("_view", "_thickness")) and "_metadata" not in str(p.parent))
pngs = [p for p in pngs if not since or p.stem.split("__")[0].split("_")[-1] >= since][-count:]

print(f"{'map':16} {'thick':>6} {'thin edge':>9} {'median':>6} {'min':>5} {'max':>5} {'p90':>5} "
      f"{'wedge':>5} {'ends':>11} {'':11}  note")
for png in pngs:
    raw = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    meta_path = png.parent.parent / f"{png.parent.name}_metadata" / f"{png.stem}.json"
    metadata = {}
    if meta_path.is_file():
        sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = sidecar.get("metadata", sidecar) or {}
    geometry = metadata.get("geometry") or {}
    name = png.stem.split("__")[0][-15:]
    if raw is None or raw.dtype != np.uint16:
        print(f"{name:16} not a uint16 height map")
        continue
    height = st.decode_height_mm(raw)
    result = st.measure_steak(height, geometry.get("x_resolution_mm"), geometry.get("y_resolution_mm"))
    if not result["measurable"]:
        print(f"{name:16} {result['reason']}")
        continue
    d = result["data"]
    f = lambda v, w=5: f"{'-':>{w}}" if v is None else f"{v:{w}.1f}"  # noqa: E731
    ends = "-" if d["wedge_mm"] is None else f"{d['thick_end_mm']:.1f}/{d['thin_end_mm']:.1f}"
    print(f"{name:16} {f(d['thickness_mm'], 6)} {f(d['thin_edge_mm'], 9)} {f(d['median_mm'], 6)} {f(d['min_mm'])} "
          f"{f(d['max_mm'])} {f(d['p90_mm'])} {f(d['wedge_mm'])} {ends:>11} "
          f"{'':11}  {d['thickness_source']}, trigger {st.trigger_time(metadata)}")
    if view:
        s = st.settings_with_defaults(None)
        grid = st.to_square_grid(height, geometry["x_resolution_mm"], geometry["y_resolution_mm"], s["grid_mm"])
        mask = st.steak_mask(grid, s)
        lo, hi = 0.0, max(float(np.nanmax(np.where(mask, grid, np.nan))), 1.0)
        scaled = np.clip(np.nan_to_num(grid, nan=0) / hi * 255, 0, 255).astype(np.uint8)
        out = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        out[~mask] = 0
        out = cv2.resize(out, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(png.with_name(png.stem + "_thickness.png")), out)
