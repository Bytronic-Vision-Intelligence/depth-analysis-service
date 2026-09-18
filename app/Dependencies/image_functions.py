from numpy import ndarray, unique, median, clip
from cv2 import imwrite
from base64 import b64decode
from cv2 import IMREAD_UNCHANGED, imdecode, imwrite, imencode, convertScaleAbs
from numpy import frombuffer, uint8, asarray, nan, float32, ndarray, uint16
RAW_PNG_MM_SCALE=100.00
BIT_SCALE_15=32768.0

class Image():
    '''A class containing image processing functions'''
    def __init__(self, image:ndarray, region_of_interest:list, trim_value:float):

        self.region_of_interest = region_of_interest
        self.set_trim_value(trim_value)

        self.image = image
        self.trimmed_image = self._trim_min_max(image)
        self.cropped_image = self._crop_image(self.trimmed_image,self.region_of_interest)

    def set_trim_value(self,trim_value:float):
        '''sets the trim value variable used for the removal of background depth data
        Args:
            trim_value: a float between 1 and 0
        '''
        if trim_value >=1 or trim_value <= 0: raise ValueError(f"Error : trim value of {trim_value} is not valid, value must be between 1 and 0")
        self.trim_value = trim_value

    def _trim_min_max(self, image:ndarray)->ndarray:
        '''trims the minimum and maximum values of the image provided
        Args:
            image: an ndarray representing the image
        Returns:
            trimmed_image: an ndarray representing the image with values trimmed
        '''
        values = unique(image)
        median_value = median(image)
        status = imwrite('output_grey.png', image)
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

    def _crop_image(self, image:ndarray, region_of_interest:list):
        '''crops an image to a specified region of interest
        Args:
            image: an ndarray representing the image
            region_of_interest: a list containing the left, top, right and bottom positions as floats
        returns:
            cropped_image: an ndarray representing the cropped image
        '''
        cropped_image = image[region_of_interest[1]:region_of_interest[3], region_of_interest[0]: region_of_interest[2]]
        # print(cropped_image)
        return cropped_image

def encode_image_to_bytes(image: ndarray) -> bytes:
    """Encode the image as JPEG (8-bit) or PNG (uint16)."""
    if image.dtype == uint16:
        success, encoded_image = imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG format.")
        return encoded_image.tobytes()

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
    depth_image = imdecode(frombuffer(image_bytes, dtype=uint16), IMREAD_UNCHANGED)
    status = imwrite('message_image.png', depth_image)
    if depth_image is None:
        raise ValueError("Error : The image payload could not be decoded by OpenCV.")
    return depth_image

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
    if not data:
        raise ValueError("Empty image bytes.")

    image = imdecode(frombuffer(data, uint8), IMREAD_UNCHANGED).astype('uint8')
    if image is None:
        raise ValueError("Could not decode image bytes.")
    return image

def decode_raw_height_png(image: ndarray) -> ndarray:
    """Decode a raw height PNG (from :func:`prepare_raw_png` float path) to mm."""
    arr = asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    mm = (arr.astype(float32) - BIT_SCALE_15) / float32(RAW_PNG_MM_SCALE)
    mm[arr == 0] = nan
    return mm
