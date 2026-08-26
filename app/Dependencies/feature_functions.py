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

        pass

    def _get_perimeter(self, image:ndarray, threshold_value:dict) -> float:
        '''returns the perimeter of the largest contour found in an image
        Args:
            image: an ndarray containing the image
        Returns:
            perimeter: a float representing the length of the largest contour arc
        '''
        blur = cv2.blur(image, self.blur_value)
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        three_channel_binary = dstack((thresh, thresh, thresh))

        contours, hierachy = cv2.findContours(
            thresh,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) ==0:
            info("INFO : No contour found in image")
            print("INFO : No contour found in image")
            return 0

        largest_contour = max(contours, key = cv2.contourArea)

        three_channel_binary = cv2.drawContours(
            three_channel_binary,
            [largest_contour],
            0,
            128,
            3
        )

        cv2.imwrite("perimeter.png", three_channel_binary)
        
        return cv2.arcLength(largest_contour, True)


    def _get_radius(self, image:ndarray, threshold:dict) -> float:
        '''returns the radius of the largest blob found
        Args:
            image: an ndarray containing the image
        Returns:
            radius: a float representing the radius of the largest blob
        '''
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.blobColor = 255

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        blur = cv2.blur(image, self.blur_value)
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        if not ret: return 0
        
        keypoints = detector.detect(thresh)
        if len(keypoints) == 0: return 0

        keypoints_sorted = sorted(keypoints, key=lambda k: k.size / 2, reverse=True)
        three_channel_binary = dstack((thresh, thresh, thresh))
        cv2.drawKeypoints(three_channel_binary, keypoints, three_channel_binary, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("radius.png", thresh)
        
        return keypoints_sorted[0].size/2


    def get_subject_details(self,image:ndarray) -> dict:
        '''Extracts the depth, area and perimeter from details found in an image
        Args:
            image: a np.ndarray representing an image
        Returns:
            details: a dictionary containing details of the image'''
        self.blur_image = cv2.blur(image, self.blur_value)
        ret, self.binary_image = cv2.threshold(self.blur_image, 50, 255, cv2.THRESH_BINARY)
        
        details = dict()
        single_channel = image if image.ndim == 2 else image[:, :, 0]
        details["depth"] = float(single_channel.max() - single_channel.min())

        thresh = {
            "min": average(single_channel) - average(single_channel)/2,
            "max": average(single_channel) + average(single_channel)/2
        }
        
        details["perimeter"] = self._get_perimeter(single_channel, thresh)
        details["radius"] = self._get_radius(single_channel, thresh)
        return details