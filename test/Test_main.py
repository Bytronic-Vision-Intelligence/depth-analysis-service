import importlib.util
import pathlib
import uuid
import pytest

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"


def load_main_module():
    spec = importlib.util.spec_from_file_location(f"main_mod_{uuid.uuid4().hex}", APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StopLoop(Exception):
    pass


class FakeStateMachine:
    '''stands in for DepthAnalysisSM so main can run without a broker'''
    instances = []

    def __init__(self, config):
        self.config = config
        self.ticks = 0
        FakeStateMachine.instances.append(self)

    def runtime(self):
        self.ticks += 1
        if self.ticks == 3:
            raise StopLoop


def test_main_builds_state_machine_from_config_and_runs_it(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("image_options:\n  pixel_pitch_mm: [0.2, 0.05]\n", encoding="utf-8")
    module = load_main_module()
    FakeStateMachine.instances.clear()
    monkeypatch.setattr(module, "DepthAnalysisSM", FakeStateMachine)

    with pytest.raises(StopLoop):
        module.main(str(config_path))

    state_machine = FakeStateMachine.instances[0]
    assert state_machine.config["image_options"]["pixel_pitch_mm"] == [0.2, 0.05]
    assert state_machine.ticks == 3


def main():
    pytest.main()


if __name__ == "__main__":
    pytest.main()
