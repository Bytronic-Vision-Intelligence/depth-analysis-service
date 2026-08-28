from numpy import (
    ndarray, 
    average, 
    dstack
)
import cv2
from logging import info

class FeatureExtraction():
    '''a simple class handler for feature extraction and image processing'''
    def __init__(self, blur_value:int = [3,3]):
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
        contours, hierachy = cv2.findContours(
            self.binary_image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) ==0:
            info("INFO : No contour found in image")
            print("INFO : No contour found in image")
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
        single_channel = image if image.ndim == 2 else image[:, :, 0]
        details["depth"] = float(single_channel.max() - single_channel.min())

        thresh = {
            "min": average(single_channel) - average(single_channel)/2,
            "max": average(single_channel) + average(single_channel)/2
        }
        
        details["perimeter"] = self._get_perimeter()
        details["radius"] = self._get_radius()
        return details