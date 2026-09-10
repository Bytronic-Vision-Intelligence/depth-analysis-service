"""The image preparation stage: decode, trim the background, crop."""

import base64

import cv2
import numpy as np
import pytest

from dependencies.image_functions import Image, extract_image


def encode(image):
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def depth_frame(background=20, subject=200):
    """Four distinct values, which _trim_min_max needs to keep the subject.

    It clips to the second-lowest and second-highest values present, so a
    frame of only background and subject clips the subject away.
    """
    image = np.full((64, 64, 3), background, dtype=np.uint8)
    image[0, 0] = 5
    image[0, 1] = 255
    image[20:44, 20:44] = subject
    return image


ROI = [0, 0, 2500, 3000]
TRIM = 0.0325


def test_extract_image_decodes_a_base64_png():
    decoded = extract_image(encode(depth_frame()))

    assert decoded.shape == (64, 64, 3)


def test_extract_image_strips_a_data_uri_prefix():
    encoded = "data:image/png;base64," + encode(depth_frame())

    assert extract_image(encoded).shape == (64, 64, 3)


def test_extract_image_refuses_something_that_is_not_an_image():
    with pytest.raises(ValueError, match="could not be decoded"):
        extract_image(base64.b64encode(b"not an image").decode("ascii"))


def test_extract_image_refuses_a_single_channel_image():
    flat = np.full((64, 64), 20, dtype=np.uint8)

    with pytest.raises(ValueError, match="not valid shape"):
        extract_image(encode(flat))


def test_the_background_is_trimmed_away_and_the_subject_kept():
    prepared = Image(depth_frame(), ROI, TRIM)

    remaining = np.unique(prepared.cropped_image)
    assert 0 in remaining, "the background was not zeroed"
    assert 200 in remaining, "the subject did not survive trimming"


def test_the_image_is_cropped_to_the_region_of_interest():
    prepared = Image(depth_frame(), [10, 10, 40, 30], TRIM)

    # numpy slicing is [top:bottom, left:right]
    assert prepared.cropped_image.shape[:2] == (20, 30)


def test_only_the_cropped_stage_is_kept():
    """Three arrays per frame were held where one is read."""
    prepared = Image(depth_frame(), ROI, TRIM)

    assert not hasattr(prepared, "image")
    assert not hasattr(prepared, "trimmed_image")


@pytest.mark.parametrize("value", [0, 1, -0.5, 1.5])
def test_a_trim_value_outside_zero_to_one_is_refused(value):
    with pytest.raises(ValueError, match="trim value"):
        Image(depth_frame(), ROI, value)


def test_an_image_with_too_few_distinct_values_is_refused():
    """_trim_min_max indexes the second-lowest and second-highest value."""
    flat = np.full((64, 64, 3), 20, dtype=np.uint8)

    with pytest.raises(ValueError, match="must be more than 3"):
        Image(flat, ROI, TRIM)
