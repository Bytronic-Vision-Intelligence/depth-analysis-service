import numpy as np
import cv2
import pytest
from dependencies.image_functions import Image, remove_spikes, decode_image_from_bytes


def tilted_plate(height=400, width=400):
    '''a plate in mm like the LJ-S publishes it, tilted the way the churchill plate is'''
    y, x = np.mgrid[:height, :width]
    return np.round(45 + 0.005 * x + 0.002 * y).astype(np.uint16)


def test_clean_surface_is_unchanged():
    plate = tilted_plate()
    plate[50:60, 50:200] = 0
    assert np.array_equal(remove_spikes(plate), plate)


def test_narrow_spike_with_gradual_ramp_is_set_to_zero():
    image = tilted_plate()
    # glare: a few pixels wide, jumps up then ramps down along the streak
    image[100:300, 200:205] += np.linspace(160, 120, 200).astype(np.uint16)[:, None]
    cleaned = remove_spikes(image)
    assert (cleaned[100:300, 200:205] == 0).all()
    assert np.array_equal(cleaned[:, :190], image[:, :190])
    assert np.array_equal(cleaned[:, 215:], image[:, 215:])


def test_narrow_dip_is_set_to_zero():
    image = tilted_plate()
    image[100:300, 200:204] -= 15
    cleaned = remove_spikes(image)
    assert (cleaned[100:300, 200:204] == 0).all()


def test_tall_wide_subject_keeps_its_height():
    image = tilted_plate()
    y, x = np.mgrid[:400, :400]
    disc = (x - 200) ** 2 + (y - 200) ** 2 < 120 ** 2
    image[disc] += 150
    cleaned = remove_spikes(image)
    assert cleaned.max() == image.max()
    # only slivers at the very edge of the disc are narrower than the window
    assert (cleaned[disc] > 0).mean() > 0.99


def test_stranded_fragment_next_to_spike_is_removed_but_distant_patch_is_kept():
    image = tilted_plate()
    image[100:300, 200:205] += 150
    # a fragment cut off from the plate by dropouts, right next to the spike
    image[150:170, 205:208] = 0
    image[150:170, 212:215] = 0
    image[150:151, 208:212] = 0
    image[169:170, 208:212] = 0
    # the same kind of patch far away from any spike
    image[150:170, 330:333] = 0
    image[150:170, 337:340] = 0
    image[150:151, 333:337] = 0
    image[169:170, 333:337] = 0
    cleaned = remove_spikes(image)
    assert (cleaned[151:169, 208:212] == 0).all()
    assert np.array_equal(cleaned[151:169, 333:337], image[151:169, 333:337])


def test_three_channel_image_is_cleaned_on_every_channel():
    image = tilted_plate()
    image[100:300, 200:205] += 150
    cleaned = remove_spikes(np.dstack([image] * 3))
    assert cleaned.shape == (400, 400, 3)
    assert (cleaned[100:300, 200:205] == 0).all()


def test_all_invalid_image_is_returned_unchanged():
    image = np.zeros((50, 50), np.uint16)
    assert np.array_equal(remove_spikes(image), image)


def test_decode_keeps_uint16_values_above_255():
    image = tilted_plate(50, 50)
    image[10, 10] = 300
    decoded = decode_image_from_bytes(cv2.imencode(".png", image)[1].tobytes())
    assert decoded.dtype == np.uint16
    assert np.array_equal(decoded, image)


def test_decode_rejects_empty_bytes():
    with pytest.raises(ValueError):
        decode_image_from_bytes(b"")


def test_image_keeps_zeros_in_cropped_image_and_trims_a_copy():
    image = tilted_plate()
    image[100:300, 200:205] += 150
    image[20:30, 20:30] = 0
    image_object = Image(image, [10, 10, 390, 390], 0.0325)
    cropped = image_object.cropped_image
    assert cropped.shape == (380, 380)
    assert (cropped[90:290, 190:195] == 0).all()
    assert (cropped[10:20, 10:20] == 0).all()
    assert (image_object.trimmed_image > 0).all()
    assert image_object.trimmed_image is not cropped


def main():
    pytest.main()


if __name__ == "__main__":
    pytest.main()
