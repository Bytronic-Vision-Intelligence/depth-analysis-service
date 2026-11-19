"""analysis: steak_thickness. One steak's height map in, one result out.

The result goes to send_depth_analysis (verdict-service's 3D input), with:
- verdict: {"verdict": pass/fail, "reason_failed": [thickness, wedge or not_measurable]}
- trigger_time: when camera-service took the trigger this map answers (verdict-service matches
  it to the line trigger)
- data: every reading in mm (thickness_mm is the judged one); checks: each judged limit
- sku, sku_name: the SKU it was judged against
- not_measurable: why nothing was measured (no steak in the map, e.g. the part timed out)
"""
import json
from json import dumps
from logging import warning

from dependencies import steak_thickness
from dependencies.cycle_timing import mark, report
from dependencies.image_functions import decode_image_text
from .base_functions.state import State


def apply_sku_updates(state_machine) -> None:
    """Take every waiting SKU selection (the UI's sku_selection: {"sku", "sku_name",
    "sku_spec"}) without blocking; the last one wins."""
    topic = getattr(state_machine, "sku_update", None)
    if topic is None or not topic.is_subscribe:
        return
    while not topic.queue.empty():
        raw = topic.queue.get_nowait()
        try:
            message = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
        except ValueError:
            warning("Discarding a malformed sku_update")
            continue
        if not isinstance(message, dict):
            continue
        sku = message.get("sku", message.get("sku_id"))
        if sku is None:
            warning("sku_update without sku / sku_id: ignored")
            continue
        state_machine.sku = str(sku)
        state_machine.sku_name = message.get("sku_name")
        spec = message.get("sku_spec")
        state_machine.sku_spec = spec if isinstance(spec, dict) else {}
        print(f"Info : SKU is now {state_machine.sku} ({len(state_machine.sku_spec)} spec fields)", flush=True)


def sku_limits(state_machine) -> dict:
    """The selected SKU's limits: the UI's sku_spec over steak_thickness.sku_spec_defaults[sku]."""
    settings = getattr(state_machine, "steak_settings", {}) or {}
    defaults = (settings.get("sku_spec_defaults") or {}).get(str(state_machine.sku)) or {}
    limits = {**defaults, **(state_machine.sku_spec or {})}
    return limits


def steak_result(message: dict, state_machine) -> dict:
    """Measure and judge one camera packet; the payload to publish."""
    settings = getattr(state_machine, "steak_settings", {}) or {}
    metadata = message.get("metadata") or {}
    geometry = metadata.get("geometry") or {}
    payload = {
        "date_time": message.get("date_time"),
        "trigger_time": steak_thickness.trigger_time(metadata),
        "source_image_id": message.get("image_id"),
        "camera_id": (message.get("camera_details") or {}).get("camera_id"),
        "sku": state_machine.sku,
        "sku_name": state_machine.sku_name,
        "part": metadata.get("part"),
    }
    try:
        raw = decode_image_text(message.get("image"))
        height = steak_thickness.decode_height_mm(raw)
        mark(state_machine, "decode")
        measured = steak_thickness.measure_steak(
            height, geometry.get("x_resolution_mm"), geometry.get("y_resolution_mm"), settings)
    except Exception as exc:   # one bad packet loses one result, not the service
        warning(f"Steak height map could not be measured: {exc}")
        measured = {"measurable": False, "reason": f"height map error: {exc}"}
    mark(state_machine, "measure")

    if measured["measurable"]:
        payload["data"] = measured["data"]
        decided = steak_thickness.judge(measured["data"], sku_limits(state_machine))
        payload["checks"] = decided.pop("checks")
        payload["verdict"] = decided
    else:
        verdict = str(settings.get("not_measurable_verdict") or "fail").strip().lower()
        payload["data"] = None
        payload["not_measurable"] = measured["reason"]
        payload["verdict"] = {"verdict": verdict,
                              "reason_failed": ["not_measurable"] if verdict == "fail" else []}
    return payload


class SteakState(State):
    '''measures one steak height map, judges it against the selected SKU and publishes it'''
    def __init__(self, state_machine, message, state_id = "Steak State"):
        super().__init__(state_machine, state_id)
        self.message = message

    def tick(self):
        state_machine = self.my_state_machine
        payload = steak_result(self.message, state_machine)
        payload["message_id"] = state_machine.message_id
        if payload["trigger_time"] is None:
            warning("Steak height map without part timing (scan_start_time / part): verdict-service "
                    "cannot match it to a trigger")
        if not state_machine.mqtt_client.is_connected:
            state_machine.mqtt_client.connect()
        state_machine.mqtt_client.publish(state_machine.service_output.topic, dumps(payload))
        mark(state_machine, "publish")
        data = payload.get("data") or {}
        # print, not logging: this service has no log setup, the orchestrator forwards stdout
        print(f"Info : steak {payload['verdict']['verdict']} {payload['verdict']['reason_failed']} "
              f"thickness {data.get('thickness_mm')} wedge {data.get('wedge_mm')} "
              f"{payload.get('not_measurable') or ''} (sku {state_machine.sku}, trigger {payload['trigger_time']})",
              flush=True)
        report(state_machine, payload["verdict"]["verdict"])
        self._return_idle()
