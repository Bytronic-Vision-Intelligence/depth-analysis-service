import re
from types import SimpleNamespace

from dependencies.cycle_timing import CycleTiming, mark, report
from dependencies.state_machine.analysis_state import AnalysisState


def test_a_cycle_reports_each_step_and_the_total_in_one_line(capsys):
    machine = SimpleNamespace(timing=CycleTiming("uid-1"))
    for step in ("decode", "crop+spikes", "measure", "publish"):
        mark(machine, step)
    report(machine, "published")

    line = capsys.readouterr().out.strip()
    assert re.fullmatch(r"Timing : uid-1 received \d\d:\d\d:\d\d\.\d{3}; decode \d+, crop\+spikes \d+, "
                        r"measure \d+, publish \d+; total \d+ ms \(published\)", line)
    assert machine.timing is None                      # one line per cycle


def test_marks_without_a_timed_cycle_are_ignored(capsys):
    machine = SimpleNamespace()                        # states built directly, as in other tests
    mark(machine, "decode")
    report(machine, "published")
    assert capsys.readouterr().out == ""


def test_a_cycle_with_no_plate_is_reported_as_such(capsys):
    machine = SimpleNamespace(pixel_pitch_mm=[0.2, 0.05], timing=CycleTiming("uid-2"))
    state = AnalysisState(machine, image="image")
    state.extractor.get_subject_details = lambda image, glare_mask: False
    state._return_idle = lambda: None
    state.tick()
    line = capsys.readouterr().out
    assert "Timing : uid-2" in line and "measure" in line and "(no plate)" in line
    assert machine.timing is None
