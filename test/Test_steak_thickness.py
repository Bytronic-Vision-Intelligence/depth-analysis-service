import base64
import json
import queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dependencies import steak_thickness as st
from dependencies.state_machine import steak_state

X_RES, Y_RES = 0.3, 1.8     # the 2150's pixel size across / along the belt at ~330 mm/s


def height_map(length_mm=170, width_mm=80, thick=(18.0, 18.0), across=(0.0, 0.0), margin_mm=10):
    """A synthetic steak: a rectangle (long axis along the belt) whose height runs from
    thick[0] at the first end to thick[1] at the other, plus across[0]..across[1] from the left
    long edge to the right one. NaN = no data around it, the belt at 0 mm."""
    rows = int((length_mm + 2 * margin_mm) / Y_RES)
    cols = int((width_mm + 2 * margin_mm) / X_RES)
    y = (np.arange(rows) * Y_RES - margin_mm)[:, None]
    x = (np.arange(cols) * X_RES - margin_mm)[None, :]
    inside = (y >= 0) & (y <= length_mm) & (x >= 0) & (x <= width_mm)
    h = thick[0] + (thick[1] - thick[0]) * y / length_mm + across[0] + (across[1] - across[0]) * x / width_mm
    height = np.where(inside, h, 0.0).astype(np.float32)
    return height


def encode(height):
    raw = np.where(np.isnan(height), 0, np.round(height * 100) + 32768).astype(np.uint16)
    ok, png = cv2.imencode(".png", raw)
    assert ok
    return base64.b64encode(png.tobytes()).decode("ascii")


def test_decode_height_mm_round_trips_and_marks_no_data():
    raw = np.array([[0, 32768, 32768 + 1850]], dtype=np.uint16)
    height = st.decode_height_mm(raw)
    assert np.isnan(height[0, 0])
    assert height[0, 1:].tolist() == pytest.approx([0.0, 18.5])


def test_flat_steak_reads_its_height_with_no_wedge():
    result = st.measure_steak(height_map(), X_RES, Y_RES)
    d = result["data"]
    assert result["measurable"]
    assert d["thickness_mm"] == pytest.approx(18.0, abs=0.05)
    assert d["median_mm"] == pytest.approx(18.0, abs=0.05)
    assert d["wedge_mm"] == pytest.approx(0.0, abs=0.05)
    # left out until the belt speed is known (they scale with it along the belt)
    assert not {"length_mm", "width_mm", "area_mm2"} & set(d)


def test_thickness_is_the_thicker_long_edge_not_the_median():
    # fat edge 22 mm, lean edge 14 mm: M&S measures along the back fat
    d = st.measure_steak(height_map(thick=(0, 0), across=(22.0, 14.0)), X_RES, Y_RES)["data"]
    assert d["thickness_mm"] == pytest.approx(21.45, abs=0.3)   # the band 3-8 mm in from the fat edge
    assert d["thin_edge_mm"] == pytest.approx(14.55, abs=0.3)
    assert d["median_mm"] == pytest.approx(18.0, abs=0.3)
    # a slope across the steak is not a wedge: the ends are the same
    assert d["wedge_mm"] == pytest.approx(0.0, abs=0.1)


def round_steak(diameter_mm=70, centre_mm=30.0, edge_mm=20.0, margin_mm=10):
    """A domed round steak (fillet): centre_mm in the middle, edge_mm at the outline."""
    rows = int((diameter_mm + 2 * margin_mm) / Y_RES)
    cols = int((diameter_mm + 2 * margin_mm) / X_RES)
    r = diameter_mm / 2
    y = (np.arange(rows) * Y_RES - margin_mm - r)[:, None]
    x = (np.arange(cols) * X_RES - margin_mm - r)[None, :]
    d = np.sqrt(x * x + y * y)
    height = np.where(d <= r, centre_mm - (centre_mm - edge_mm) * (d / r) ** 2, 0.0).astype(np.float32)
    return height


def test_centre_method_reads_the_middle_of_a_round_steak():
    edge = st.measure_steak(round_steak(), X_RES, Y_RES)["data"]
    centre = st.measure_steak(round_steak(), X_RES, Y_RES, thickness_method="centre")["data"]
    assert centre["thickness_source"] == "centre"
    assert centre["thickness_mm"] == pytest.approx(29.0, abs=0.7)    # the core, near the 30 mm top
    assert edge["thickness_source"] == "edge" and edge["thickness_mm"] < 25   # the edge band is lower
    assert edge["centre_mm"] == centre["thickness_mm"]                       # always reported


def test_steak_result_uses_the_skus_thickness_method():
    state_machine = make_state_machine(sku="fillet_standard",
                                       defaults={"fillet_standard": {"thickness": [20, 40], "thickness_method": "centre"}})
    payload = steak_state.steak_result(packet(round_steak()), state_machine)
    assert payload["data"]["thickness_source"] == "centre"
    assert payload["verdict"] == {"verdict": "pass", "reason_failed": []}


def test_wedge_is_the_difference_between_the_ends():
    d = st.measure_steak(height_map(thick=(24.0, 12.0)), X_RES, Y_RES)["data"]
    # each end's median sits a tenth of the length in: 24 - 1.2 vs 12 + 1.2
    assert d["wedge_mm"] == pytest.approx(9.6, abs=0.4)
    assert d["thick_end_mm"] > d["thin_end_mm"]


def test_no_steak_and_no_pixel_size_are_not_measurable():
    assert st.measure_steak(np.full((1, 1), np.nan, np.float32), X_RES, Y_RES)["reason"] == "no steak in the 3D map"
    assert st.measure_steak(height_map(), None, Y_RES)["measurable"] is False
    # a small patch (a finger) is not a steak
    assert st.measure_steak(height_map(length_mm=20, width_mm=20), X_RES, Y_RES)["measurable"] is False


def test_no_data_specks_inside_the_steak_are_ignored():
    height = height_map()
    rng = np.random.default_rng(0)
    holes = rng.random(height.shape) < 0.05
    height[holes & (height > 0)] = np.nan
    d = st.measure_steak(height, X_RES, Y_RES)["data"]
    assert d["thickness_mm"] == pytest.approx(18.0, abs=0.05)
    assert d["valid_share"] > 0.9


@pytest.mark.parametrize("value, expected", [
    ([15, 25], (15.0, 25.0)), ((5, 15), (5.0, 15.0)), ("20-35", (20.0, 35.0)),
    ([None, 5], (float("-inf"), 5.0)), (None, None), ("x", None), ([1], None),
])
def test_spec_range(value, expected):
    assert st.spec_range(value) == expected


def test_judge_thickness_and_wedge():
    data = {"thickness_mm": 26.0, "wedge_mm": 3.0}
    assert st.judge(data, {"thickness": [15, 25]})["reason_failed"] == ["thickness"]
    assert st.judge(data, {"thickness": [15, 30], "wedge": [0, 2]})["reason_failed"] == ["wedge"]
    passed = st.judge(data, {"thickness": [15, 30]})
    assert passed["verdict"] == "pass" and set(passed["checks"]) == {"thickness"}
    # no limits set: measured only
    assert st.judge(data, {}) == {"verdict": "pass", "reason_failed": [], "checks": {}}
    # a limit with no reading fails
    assert st.judge({"thickness_mm": 20, "wedge_mm": None}, {"wedge": [0, 5]})["reason_failed"] == ["wedge"]


def test_trigger_time_is_scan_start_less_its_offset():
    metadata = {"scan_start_time": "2026-11-19 16:02:26.333000", "part": {"start_after_trigger_ms": 638.0}}
    assert st.trigger_time(metadata) == "2026-11-19 16:02:25.695000"
    assert st.trigger_time({"scan_start_time": "2026-11-19 16:02:26", "part": None}) is None
    assert st.trigger_time(None) is None


def make_state_machine(sku="sirloin_standard", defaults=None, ui_spec=None):
    state_machine = SimpleNamespace(
        sku=sku, sku_name=None, sku_spec=ui_spec or {},
        steak_settings={"sku_spec_defaults": defaults or {"sirloin_standard": {"thickness": [15, 25]}}},
    )
    return state_machine


def packet(height, part=True):
    metadata = {"geometry": {"x_resolution_mm": X_RES, "y_resolution_mm": Y_RES}}
    if part:
        metadata["scan_start_time"] = "2026-11-19 16:02:26.333000"
        metadata["part"] = {"rows": height.shape[0], "start_after_trigger_ms": 600.0}
    message = {"image": encode(height), "date_time": "2026-11-19 16:02:27.000000", "image_id": "raw",
               "camera_details": {"camera_id": "depth_1"}, "metadata": metadata}
    return message


def test_steak_result_passes_a_steak_in_spec():
    payload = steak_state.steak_result(packet(height_map()), make_state_machine())
    assert payload["verdict"] == {"verdict": "pass", "reason_failed": []}
    assert payload["trigger_time"] == "2026-11-19 16:02:25.733000"
    assert payload["data"]["thickness_mm"] == pytest.approx(18.0, abs=0.05)
    assert payload["checks"]["thickness"]["limits"] == [15.0, 25.0]
    assert payload["camera_id"] == "depth_1" and payload["sku"] == "sirloin_standard"


def test_steak_result_ui_spec_wins_over_the_defaults():
    state_machine = make_state_machine(ui_spec={"thickness": [20, 25]})
    payload = steak_state.steak_result(packet(height_map()), state_machine)
    assert payload["verdict"] == {"verdict": "fail", "reason_failed": ["thickness"]}


def test_steak_result_empty_map_is_not_measurable():
    payload = steak_state.steak_result(packet(np.full((1, 1), np.nan, np.float32), part=False), make_state_machine())
    assert payload["not_measurable"] == "no steak in the 3D map"
    assert payload["verdict"] == {"verdict": "fail", "reason_failed": ["not_measurable"]}
    assert payload["trigger_time"] is None and payload["data"] is None
    state_machine = make_state_machine()
    state_machine.steak_settings["not_measurable_verdict"] = "pass"
    passed = steak_state.steak_result(packet(np.full((1, 1), np.nan, np.float32)), state_machine)
    assert passed["verdict"] == {"verdict": "pass", "reason_failed": []}


def test_steak_result_survives_a_bad_image():
    message = packet(height_map())
    message["image"] = "not an image"
    payload = steak_state.steak_result(message, make_state_machine())
    assert payload["not_measurable"].startswith("height map error")


def test_apply_sku_updates_takes_the_last_selection():
    topic = SimpleNamespace(is_subscribe=True, queue=queue.Queue())
    for message in ({"sku": "ribeye_standard", "sku_spec": {"thickness": [15, 25]}}, "not json",
                    {"sku_id": "ribeye_thick_cut", "sku_name": "Ribeye thick", "sku_spec": {"thickness": [30, 40]}}):
        topic.queue.put(message if isinstance(message, str) else json.dumps(message))
    state_machine = SimpleNamespace(sku_update=topic, sku=None, sku_name=None, sku_spec={})
    steak_state.apply_sku_updates(state_machine)
    assert (state_machine.sku, state_machine.sku_name) == ("ribeye_thick_cut", "Ribeye thick")
    assert state_machine.sku_spec == {"thickness": [30, 40]}
