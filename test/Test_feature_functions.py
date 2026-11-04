import math
import numpy as np
import pytest
from dependencies.feature_functions import FeatureExtraction
from dependencies.image_functions import Image

PITCH = [0.5, 0.25]
BOWL_RADIUS_MM = 60
RIM_HEIGHT_MM = 24
FLOOR_HEIGHT_MM = 10


def bowl_scene(with_bowl=True, seed=0):
    '''a tilted plate in mm with a bowl that is round in mm, so an ellipse in pixels'''
    height, width = 800, 500
    y, x = np.mgrid[:height, :width]
    image = 45 + 0.005 * x + 0.002 * y
    if with_bowl:
        r = np.hypot((x - 250) * PITCH[0], (y - 400) * PITCH[1]) / BOWL_RADIUS_MM
        inside = r <= 1
        image[inside] += FLOOR_HEIGHT_MM + (RIM_HEIGHT_MM - FLOOR_HEIGHT_MM) * r[inside] ** 2
        # speckle dropouts inside the bowl and a bite out of the rim, like the real captures
        rng = np.random.default_rng(seed)
        image[inside & (rng.random(image.shape) < 0.1)] = 0
        image[(r > 0.9) & (r <= 1.0) & (y < 400) & (np.abs(x - 250) < 40)] = 0
    return np.round(image).astype(np.uint16)


def test_bowl_is_measured_relative_to_the_plate_in_mm():
    details = FeatureExtraction(PITCH).get_subject_details(bowl_scene())
    # the plate tilts ~3mm across the image, which must not be counted
    assert details["depth"] == pytest.approx(RIM_HEIGHT_MM, abs=1)
    assert details["radius"] == pytest.approx(BOWL_RADIUS_MM, rel=0.02)
    assert details["perimeter"] == pytest.approx(2 * math.pi * BOWL_RADIUS_MM, rel=0.03)


def test_glare_inside_the_bowl_does_not_change_the_measurement():
    clean = FeatureExtraction(PITCH).get_subject_details(Image(bowl_scene(), [0, 0, 500, 800]).cropped_image)
    glare = bowl_scene()
    glare[300:450, 240:245] = 200
    details = FeatureExtraction(PITCH).get_subject_details(Image(glare, [0, 0, 500, 800]).cropped_image)
    assert details["depth"] == pytest.approx(clean["depth"], abs=0.5)
    assert details["radius"] == pytest.approx(clean["radius"], rel=0.01)


def test_a_few_stray_glare_pixels_do_not_set_the_depth():
    clean = FeatureExtraction(PITCH).get_subject_details(bowl_scene())
    stray = bowl_scene()
    stray[402:432, 250] = 150
    details = FeatureExtraction(PITCH).get_subject_details(stray)
    assert details["depth"] == pytest.approx(clean["depth"], abs=0.5)


def test_frame_is_rejected_when_too_much_of_the_subject_was_glare():
    scene = bowl_scene()
    glare = np.zeros(scene.shape, bool)
    glare[250:550, 180:320] = scene[250:550, 180:320] > 0
    scene[glare] = 0
    assert FeatureExtraction(PITCH).get_subject_details(scene, glare) is False


def test_frame_with_a_little_glare_is_still_measured():
    scene = bowl_scene()
    glare = np.zeros(scene.shape, bool)
    glare[300:450, 240:245] = True
    scene[glare] = 0
    details = FeatureExtraction(PITCH).get_subject_details(scene, glare)
    assert details["depth"] == pytest.approx(RIM_HEIGHT_MM, abs=1)


def test_empty_plate_has_no_subject():
    assert FeatureExtraction(PITCH).get_subject_details(bowl_scene(with_bowl=False)) is False


def test_image_with_no_valid_pixels_has_no_subject():
    assert FeatureExtraction(PITCH).get_subject_details(np.zeros((100, 100), np.uint16)) is False


@pytest.mark.parametrize("pitch", [None, [0.5], [0.5, 0], [-0.5, 0.25]])
def test_invalid_pixel_pitch_is_rejected(pitch):
    with pytest.raises(ValueError):
        FeatureExtraction(pitch)


def main():
    pytest.main()


if __name__ == "__main__":
    pytest.main()
