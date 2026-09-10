from logging import info

import cv2
from numpy import ndarray


class FeatureExtraction():
    '''a simple class handler for feature extraction and image processing'''

    def __init__(self, blur_value: list = None):
        # Was a mutable default. Nothing here writes to it, so it never bit,
        # but a default list is shared by every instance that takes it.
        blur_value = [3, 3] if blur_value is None else blur_value
        # `or`, not `and`. With `and` a kernel had to be even in BOTH
        # dimensions to be refused, so [2, 3] passed the check that exists to
        # stop it. cv2.blur itself accepts even kernels; if this proves
        # inconvenient the check can go, but it should at least enforce what
        # its own message claims.
        if blur_value[0] % 2 == 0 or blur_value[1] % 2 == 0:
            raise ValueError(
                f"Error : blur values must be odd and more than 1 value of "
                f"{blur_value} is not valid")
        self.blur_value = blur_value

    def _largest_contour(self, binary_image: ndarray):
        '''finds the largest contour in a binary image
        Args:
            binary_image: the thresholded image to search
        Returns:
            largest_contour: the largest contour by area, or None when the
                image contains none'''
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            info("No contour found in image")
            return None

        return max(contours, key=cv2.contourArea)

    def get_subject_details(self, image: ndarray):
        '''Extracts the depth, perimeter and radius of the subject in an image
        Args:
            image: a np.ndarray representing an image
        Returns:
            details: a dictionary containing details of the image, or None
                when the image holds nothing to measure'''
        # Reduced here, once. cv2.findContours takes CV_8UC1 only, so a
        # multi-channel frame raised out of it -- and main.py reduced channels
        # itself before calling, which put the same decision in two files and
        # still let a two-channel frame through.
        single_channel = image if image.ndim == 2 else image[:, :, 0]

        blurred = cv2.blur(single_channel, self.blur_value)
        _, binary_image = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Found once. Perimeter and radius each used to call this, so every
        # measurement ran findContours twice over the same image for two
        # answers about the same shape.
        contour = self._largest_contour(binary_image)
        if contour is None:
            # Was `return 0`, which cv2.arcLength then rejected -- a blank
            # frame raised out of the measurement instead of being skipped.
            return None

        _, _, width, height = cv2.boundingRect(contour)

        # float() throughout: these are published as JSON, and json.dumps
        # refuses a numpy scalar with "Object of type int64 is not JSON
        # serializable" -- on every frame that measured successfully.
        return {
            "depth": float(single_channel.max() - single_channel.min()),
            "perimeter": float(cv2.arcLength(contour, True)),
            "radius": float((height / 2 + width / 2) / 2),
        }
