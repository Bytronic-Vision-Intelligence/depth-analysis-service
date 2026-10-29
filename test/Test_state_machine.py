from types import SimpleNamespace
import pytest
from dependencies.state_machine import depth_analysis_sm
from dependencies.state_machine.analysis_state import AnalysisState


class FakeMQTTClient:
    def __init__(self, config):
        self.config = config
        self.is_connected = False

    def connect(self):
        self.is_connected = True


def make_config(image_options):
    return {
        "mqtt": {"mqtt_ip": "localhost", "mqtt_port": 1883},
        # nothing subscribes, so no listener threads or broker are needed
        "topics": [
            {"name": "receive_depth_image", "topic": "test/depth", "is_subscribe": False, "is_trigger": True},
            {"name": "send_depth_analysis", "topic": "test/out", "is_subscribe": False, "is_trigger": False},
        ],
        "image_options": image_options,
    }


def test_state_machine_reads_pixel_pitch_from_config(monkeypatch):
    monkeypatch.setattr(depth_analysis_sm, "MQTTClient", FakeMQTTClient)
    state_machine = depth_analysis_sm.DepthAnalysisSM(make_config(
        {"region_of_interest": [700, 100, 2500, 2100], "trim_value": 0.0325, "pixel_pitch_mm": [0.2, 0.05]}
    ))
    assert state_machine.pixel_pitch_mm == [0.2, 0.05]
    assert state_machine.region_of_interest == [700, 100, 2500, 2100]


def test_state_machine_requires_pixel_pitch(monkeypatch):
    monkeypatch.setattr(depth_analysis_sm, "MQTTClient", FakeMQTTClient)
    with pytest.raises(ValueError):
        depth_analysis_sm.DepthAnalysisSM(make_config({"region_of_interest": [0, 0, 10, 10], "trim_value": 0.0325}))


def test_analysis_state_measures_with_the_configured_pitch():
    state = AnalysisState(SimpleNamespace(pixel_pitch_mm=[0.2, 0.05]), image=None)
    assert state.extractor.pixel_pitch_mm.tolist() == pytest.approx([0.2, 0.05])


def test_analysis_state_passes_the_glare_mask_to_the_measurement():
    calls = {}
    state = AnalysisState(SimpleNamespace(pixel_pitch_mm=[0.2, 0.05]), image="image", glare_mask="mask")
    state.extractor.get_subject_details = lambda image, glare_mask: calls.update(image=image, glare_mask=glare_mask) or False
    state._return_idle = lambda: calls.update(idle=True)
    state.tick()
    assert calls == {"image": "image", "glare_mask": "mask", "idle": True}


def main():
    pytest.main()


if __name__ == "__main__":
    pytest.main()
