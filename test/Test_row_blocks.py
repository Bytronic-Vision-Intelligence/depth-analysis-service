"""Row-block work gives the same result as doing the whole image at once."""
import numpy as np

from dependencies.row_blocks import run_in_row_blocks
from dependencies.image_functions import find_spikes
from dependencies.feature_functions import FeatureExtraction


def test_row_blocks_cover_every_row_once():
    seen = np.zeros(1000, int)

    def work(top, bottom):
        seen[top:bottom] += 1

    run_in_row_blocks(1000, work, blocks=4)
    assert (seen == 1).all()
    run_in_row_blocks(10, work, blocks=4)        # too small to split: one piece
    assert (seen[:10] == 2).all()


def _scene(seed):
    rng = np.random.default_rng(seed)
    image = np.where(rng.random((600, 500)) < 0.1, 0, 50).astype(np.float32)
    yy, xx = np.ogrid[:600, :500]
    image[(yy - 300) ** 2 + (xx - 250) ** 2 < 120 ** 2] = 72
    image[:, 100:104] += 30                                  # a glare streak
    return image.astype(np.uint16)


def test_plane_heights_match_the_full_grid_formula():
    image = _scene(1)
    valid = image > 0
    heights = FeatureExtraction([0.2, 0.2])._get_plate_heights(image, valid)
    ys, xs = np.mgrid[:600, :500]
    # with a flat floor the fitted plane is the floor itself, so heights are the image minus 50
    assert np.allclose(heights[valid], image[valid] - 50.0, atol=1e-3)
    assert (heights[~valid] == 0).all()


def test_spikes_are_found_the_same_in_blocks_as_in_one_piece(monkeypatch):
    import dependencies.image_functions as image_functions
    image = _scene(2)
    split = find_spikes(image)
    monkeypatch.setattr(image_functions, "run_in_row_blocks", lambda rows, work, blocks=4, min_rows=256: work(0, rows))
    whole = find_spikes(image)
    assert split.any() and np.array_equal(split, whole)
