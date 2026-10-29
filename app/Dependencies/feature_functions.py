from numpy import (
    ndarray,
    dstack,
    where,
    float32,
    uint8,
    unique,
    isin,
    median,
    abs
)
import cv2
from logging import info

class FeatureExtraction():
    '''a simple class handler for feature extraction and image processing'''
    def __init__(self, blur_value:int = [7,7]):
        if blur_value[0] % 2 == 0 and blur_value[1] % 2 == 0: 
            raise ValueError(f"Error : blur values must be odd and more than 1 value of {blur_value} is not valid")
        self.blur_value = blur_value
        self.blur_image = None
        self.binary_image = None
        self.three_channel_binary = None

        pass

    def _get_largest_contour(self):
        '''finds the largest contour in an image and returns it
        Returns:
            largetst_contour: the largest contour by area'''
        image = self.binary_image
        contours, hierachy = cv2.findContours(
            image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) ==0:
            info("INFO : No contour found in image")
            return 0
        
        return max(contours, key = cv2.contourArea)

    def _get_perimeter(self) -> float:
        '''returns the perimeter of the largest contour found in an image
        Returns:
            perimeter: a float representing the length of the largest contour arc
        '''
        largest_contour = self._get_largest_contour()
        
        return cv2.arcLength(largest_contour, True)


    def _get_radius(self) -> float:
        '''returns the radius of the largest blob found
        Returns:
            radius: a float representing the radius of the largest blob
        '''
        largest_contour = self._get_largest_contour()
        x,y,w,h = cv2.boundingRect(largest_contour)
        return (h/2+w/2)/2


    def get_subject_details(self,image:ndarray) -> dict:
        '''Extracts the depth, area and perimeter from details found in an image
        Args:
            image: a np.ndarray representing an image
        Returns:
            details: a dictionary containing details of the image'''
        self.blur_image = cv2.blur(image, self.blur_value)
        ret, self.binary_image = cv2.threshold(self.blur_image, 50, 255, cv2.THRESH_BINARY)
        self.three_channel_binary = dstack((self.binary_image, self.binary_image, self.binary_image))
        
        details = dict()
        image = self.__remove_outliers(image)
        single_channel = image if image.ndim == 2 else image[:, :, 0]
        details["depth"] = float(single_channel.max() - single_channel.min())
        cv2.imwrite("outliers removed.png", single_channel)

        details["perimeter"] = self._get_perimeter()
        details["radius"] = self._get_radius()
        return details

    def __remove_outliers(self, image: ndarray, kernel_size: int = 51, min_height: float = 10, low_ratio: float = 0.25) -> ndarray:
        '''removes narrow raised spikes (e.g. glare) including their gradual flanks
        A morphological opening wider than the spike estimates the underlying surface.
        Regions rising above the noise floor are removed if they peak more than
        min_height above the surface, so the whole ramp is removed rather than just the tip.
        Args:
            image: an ndarray representing the depth image
            kernel_size: odd width in pixels, wider than the widest spike but narrower than the subject
            min_height: how far above the surface a peak must rise to count as a spike
            low_ratio: fraction of min_height used to grow the spike region down its flanks
        Returns:
            cleaned: a copy of the image with spikes replaced by the surface estimate
        '''
        single_channel = image if image.ndim == 2 else image[:, :, 0]
        valid = single_channel > 0
        if not valid.any():
            return image
        smoothed = cv2.blur(single_channel.astype(float32), (5, 5))
        # stop invalid zero pixels dragging the surface estimate down
        surface_input = where(valid, smoothed, smoothed.max())
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        surface = cv2.morphologyEx(surface_input, cv2.MORPH_OPEN, kernel)
        height_above = where(valid, smoothed - surface, 0)

        # the opening sits at the bottom of the noise, so measure the noise floor and keep above it
        valid_heights = height_above[valid]
        noise_level = median(valid_heights)
        noise_spread = 1.4826 * median(abs(valid_heights - noise_level))
        low_threshold = max(min_height * low_ratio, noise_level + 4 * noise_spread)
        surface = surface + noise_level

        candidates = (height_above > low_threshold).astype(uint8)
        _, labels = cv2.connectedComponents(candidates)
        spike_labels = unique(labels[height_above > noise_level + min_height])
        spikes = isin(labels, spike_labels[spike_labels != 0])
        if spikes.any():
            info(f"INFO : removed {int(spikes.sum())} spike pixels")

        cleaned = image.copy()
        fill = surface[spikes]
        cleaned[spikes] = (fill if image.ndim == 2 else fill[:, None]).astype(image.dtype)
        return cleaned