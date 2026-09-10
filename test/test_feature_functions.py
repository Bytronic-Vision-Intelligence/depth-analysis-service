"""The measurement stage: what is published about the subject."""

import json

import cv2
import numpy as np
import pytest

from dependencies.feature_functions import FeatureExtraction


def subject(size=64, side=24, value=200):
    """A single bright square on black, which thresholds to one contour."""
    image = np.zeros((size, size), dtype=np.uint8)
    start = (size - side) // 2
    image[start:start + side, start:start + side] = value
    return image


def test_it_measures_depth_perimeter_and_radius():
    details = FeatureExtraction([7, 7]).get_subject_details(subject())

    assert set(details) == {"depth", "perimeter", "radius"}
    assert details["depth"] == 200.0
    assert details["radius"] > 0
    assert details["perimeter"] > 0


def test_every_measurement_is_json_serialisable():
    """They are published as JSON. A numpy scalar raises "Object of type
    int64 is not JSON serializable" on every frame measured successfully."""
    details = FeatureExtraction([7, 7]).get_subject_details(subject())

    json.dumps(details)
    assert all(type(value) is float for value in details.values())


def test_a_frame_with_nothing_in_it_returns_none(caplog):
    """It returned 0, which cv2.arcLength then rejected -- so a blank frame
    raised out of the measurement rather than being skipped."""
    blank = np.zeros((64, 64), dtype=np.uint8)

    with caplog.at_level("INFO"):
        assert FeatureExtraction([7, 7]).get_subject_details(blank) is None
    assert "No contour found" in caplog.text


def test_the_contour_is_found_once_per_measurement(monkeypatch):
    """Perimeter and radius each used to find it again, so every frame ran
    findContours twice for two answers about the same shape."""
    calls = []
    real = cv2.findContours

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2, "findContours", counting)
    FeatureExtraction([7, 7]).get_subject_details(subject())

    assert len(calls) == 1


def test_a_three_channel_image_is_measured_on_one_channel():
    flat = subject()
    stacked = np.dstack((flat, flat, flat))

    assert (FeatureExtraction([7, 7]).get_subject_details(stacked)["depth"]
            == FeatureExtraction([7, 7]).get_subject_details(flat)["depth"])


def test_an_even_blur_kernel_is_refused_in_either_dimension():
    """The check used `and`, so a kernel had to be even in BOTH dimensions to
    be refused -- [2, 3] passed the guard that exists to stop it."""
    for kernel in ([2, 3], [3, 2], [2, 2]):
        with pytest.raises(ValueError, match="must be odd"):
            FeatureExtraction(kernel)


def test_the_default_kernel_is_not_shared_between_instances():
    first = FeatureExtraction()
    second = FeatureExtraction()

    assert first.blur_value == [3, 3]
    assert first.blur_value is not second.blur_value


def test_no_scratch_state_is_kept_between_frames():
    """blur_image, binary_image and three_channel_binary were held on the
    instance; the last was written and never read at all."""
    extractor = FeatureExtraction([7, 7])
    extractor.get_subject_details(subject())

    for attribute in ("blur_image", "binary_image", "three_channel_binary"):
        assert not hasattr(extractor, attribute)
