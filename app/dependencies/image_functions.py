from base64 import b64decode
from cv2 import IMREAD_UNCHANGED, imdecode, getStructuringElement, connectedComponentsWithStats, dilate, erode, MORPH_RECT, CC_STAT_AREA
from numpy import frombuffer, uint8, float32, ndarray, where, zeros, ones, isin, arange
from logging import info

class Image():
    '''A class containing image processing functions'''
    def __init__(self, image:ndarray, region_of_interest:list):

        self.region_of_interest = region_of_interest

        self.image = image
        self.cropped_image = self._crop_image(image,self.region_of_interest)
        self.spike_mask = find_spikes(self.cropped_image)
        self.cropped_image = remove_spikes(self.cropped_image, self.spike_mask)

    def _crop_image(self, image:ndarray, region_of_interest:list):
        '''crops an image to a specified region of interest
        Args:
            image: an ndarray representing the image
            region_of_interest: a list containing the left, top, right and bottom positions as floats
        returns:
            cropped_image: an ndarray representing the cropped image
        '''
        cropped_image = image[region_of_interest[1]:region_of_interest[3], region_of_interest[0]: region_of_interest[2]]
        return cropped_image

def find_spikes(image: ndarray, max_width: int = 51, min_jump: float = 5, min_area: int = 200, search_distance: int = 15) -> ndarray:
    '''finds glare spikes and dips in a depth image
    Glare from the LJ-S shows up as streaks a few pixels wide in x, running along y, that
    jump above or below the pixels either side. An opening (closing for dips) max_width
    wide in x gives the surface without anything narrower, so the check depends on width
    not height and a large subject is left alone however high it is. Only x is checked
    because a deep bowl's rim is narrow in y where it runs across the image. Fragments of
    glare left stranded between dropouts have no surface to compare against, so small
    patches next to a spike are included too.
    Args:
        image: an ndarray representing the depth image, 0 marks invalid pixels
        max_width: odd width in pixels, wider than the widest spike but narrower than the subject
        min_jump: how far a pixel must stand above or below the surface to count as a spike
        min_area: patches of valid pixels smaller than this many pixels next to a spike are included
        search_distance: how close in pixels a patch must be to a spike to be included
    Returns:
        spikes: an ndarray of bools marking the spike pixels
    '''
    single_channel = image if image.ndim == 2 else image[:, :, 0]
    valid = single_channel > 0
    if not valid.any():
        return zeros(valid.shape, bool)
    values = single_channel.astype(float32)
    kernel = getStructuringElement(MORPH_RECT, (max_width, 1))
    # invalid pixels must neither pull the surface down nor push it up, otherwise a streak
    # with dropouts beside it looks as wide as the dropouts
    out_of_range = float32(1e6)
    opened = dilate(where(valid, erode(where(valid, values, out_of_range), kernel), -out_of_range), kernel)
    closed = erode(where(valid, dilate(where(valid, values, -out_of_range), kernel), out_of_range), kernel)
    spikes = valid & ((values - opened > min_jump) | (closed - values > min_jump))
    if not spikes.any():
        return spikes

    # only patches near a spike, so isolated specks of real surface such as a rim are kept
    count, labels, stats, _ = connectedComponentsWithStats((valid & ~spikes).astype(uint8), connectivity=8)
    near_spike = dilate(spikes.astype(uint8), ones((search_distance, search_distance), uint8)) > 0
    small = stats[:, CC_STAT_AREA] < min_area
    small &= isin(arange(count), labels[near_spike])
    small[0] = False
    return spikes | small[labels]

def remove_spikes(image: ndarray, spikes: ndarray = None) -> ndarray:
    '''sets glare spikes and dips to 0 so they are treated like any other invalid pixel
    Args:
        image: an ndarray representing the depth image, 0 marks invalid pixels
        spikes: the mask from find_spikes, found from the image when not given
    Returns:
        cleaned: a copy of the image with spikes set to 0
    '''
    if spikes is None:
        spikes = find_spikes(image)
    if spikes.any():
        info(f"INFO : removed {int(spikes.sum())} spike pixels")
    cleaned = image.copy()
    cleaned[spikes] = 0
    return cleaned

def extract_image(encoded_image: str) -> bytes:
    """Decode base64 image bytes from a camera packet."""
    encoded = str(encoded_image).strip()
    if encoded.lower().startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    encoded = "".join(encoded.split())
    padding = (-len(encoded)) % 4
    if padding:
        encoded += "=" * padding
    return b64decode(encoded, validate=False)

def decode_image_from_bytes(data: bytes) -> ndarray:
    """Decode image bytes into an ndarray."""
    if len(data) < 1:
        raise ValueError("Empty image bytes.")

    # keep the native dtype, depth arrives as uint16 mm and casting to uint8 wraps anything over 255
    image = imdecode(frombuffer(data, uint8), IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not decode image bytes.")
    return image
