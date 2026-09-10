from base64 import b64decode

from numpy import ndarray, unique, median, clip, frombuffer, uint8
from cv2 import imdecode, IMREAD_UNCHANGED


class Image():
    '''A class containing image processing functions'''
    def __init__(self, image: ndarray, region_of_interest: list, trim_value: float):
        if trim_value >= 1 or trim_value <= 0:
            raise ValueError(
                f"Error : trim value of {trim_value} is not valid, value must "
                f"be between 1 and 0")
        self.trim_value = trim_value
        self.region_of_interest = region_of_interest

        # Only cropped_image is read. The original and the trimmed stage were
        # kept as attributes too, which held three arrays per frame where one
        # is used and nothing ever looked at the other two.
        trimmed = self._trim_min_max(image)
        self.cropped_image = self._crop_image(trimmed, region_of_interest)

    def _trim_min_max(self, image: ndarray) -> ndarray:
        '''trims the minimum and maximum values of the image provided
        Args:
            image: an ndarray representing the image
        Returns:
            trimmed_image: an ndarray representing the image with values trimmed
        '''
        values = unique(image)
        median_value = median(image)
        if values.size < 3:
            raise ValueError(f"Error : value length of {values.size} is not valid, must be more than 3")

        second_max = values[-2]
        second_min = values[1]

        image = clip(image, second_min, second_max)
        mask = (image < 10)
        image[mask] = second_max

        mask = (image < median_value * (1+self.trim_value)) & (image > median_value * (1-self.trim_value))
        image[mask] = 0

        return image

    def _crop_image(self, image: ndarray, region_of_interest: list):
        '''crops an image to a specified region of interest
        Args:
            image: an ndarray representing the image
            region_of_interest: a list containing the left, top, right and bottom positions as floats
        returns:
            cropped_image: an ndarray representing the cropped image
        '''
        cropped_image = image[region_of_interest[1]:region_of_interest[3], region_of_interest[0]: region_of_interest[2]]
        return cropped_image


# Moved here from mqtt_functions.py, which is byte-identical to
# service-template's in every Bytronic service and pinned by a test.
# Decoding an image is not an MQTT concern and cannot live there.
def extract_image(encoded_image):
    '''extracts the image from the encoded array into a ndarray
    Arga:
        encoded_image: a string containing the image packet encoded as base64
    Returns:
        image: an ndarray containing valid image data
    '''
    if isinstance(encoded_image, str) and "," in encoded_image:
        encoded_image = encoded_image.split(",", 1)[1]

    image_bytes = b64decode(encoded_image)
    depth_image = imdecode(frombuffer(image_bytes, dtype=uint8), IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("Error : The image payload could not be decoded by OpenCV.")
    if not len(depth_image.shape) > 2:
        raise ValueError("Error : image not valid shape")
    return depth_image
