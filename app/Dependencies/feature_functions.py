from numpy import (
    ndarray, 
    average, 
    unique, 
    clip,
    median
)
from cv2 import (
    findContours, 
    RETR_TREE, 
    CHAIN_APPROX_SIMPLE, 
    Canny, 
    contourArea, 
    arcLength,
    SimpleBlobDetector_create,
    SimpleBlobDetector_Params,
    imwrite,
    drawContours,
    drawKeypoints,
    DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

class FeatureExtraction():
    '''a simple class handler for feature extraction and image processing'''

    def _trim_min_max(self, image:ndarray)->ndarray:
        '''trims the minimum and maximum values of the image provided
        Args:
            image: an ndarray representing the image
        Returns:
            trimmed_image: an ndarray representing the image with values trimmed'''
        values = unique(image)
        median_value = median(image)
        if values.size < 3:
            raise ValueError(f"Error : value length of {values.size} is not valid, must be more than 3")
        
        second_max = values[-2]
        second_min = values[1]
        print(values)

        image = clip(image, second_min, second_max)
        mask = (image < 10)
        image[mask] = second_max

        mask = (image < median_value * 1.05) & (image > median_value * 0.95)
        image[mask] = 0

        return image

    def _get_perimeter(self, image:ndarray, threshold:dict) -> float:
        '''returns the perimeter of the largest contour found in an image
        Args:
            image: an ndarray containing the image
        Returns:
            perimeter: a float representing the length of the largest contour arc
        '''
        canny_image = Canny(image,threshold["max"],threshold["min"])

        contours, hierachy = findContours(
            canny_image,
            RETR_TREE,
            CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key = contourArea)

        drawContours(image, largest_contour, -1, 255, 3)
        imwrite("C:/Users/AmyHarrison/pointcloud_matcher/perimeter.png", image)
        
        return arcLength(largest_contour, True)

    def _get_radius(self, image:ndarray, threshold:dict) -> float:
        '''returns the radius of the largest blob found
        Args:
            image: an ndarray containing the image
        Returns:
            radius: a float representing the radius of the largest blob'''
        
        params = SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.1
        detector = SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        keypoints_sorted = sorted(keypoints, key=lambda k: k.size / 2, reverse=True)
        drawKeypoints(image, keypoints, image, (0, 255, 0), DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imwrite("C:/Users/AmyHarrison/pointcloud_matcher/radius.png", image)
        
        return keypoints_sorted[0].size/2


    def get_subject_details(self,image:ndarray) -> dict:
        '''Extracts the depth, area and perimeter from details found in an image
        Args:
            image: a np.ndarray representing an image
        Returns:
            details: a dictionary containing details of the image'''
        
        details = dict()
        single_channel = image if image.ndim == 2 else image[:, :, 0]
        single_channel = self._trim_min_max(single_channel)
        imwrite("C:/Users/AmyHarrison/pointcloud_matcher/img.png", single_channel)
        details["depth"] = float(single_channel.max() - single_channel.min())

        thresh = {
            "min": average(single_channel) - average(single_channel)/2,
            "max": average(single_channel) + average(single_channel)/2
        }
        
        details["perimeter"] = self._get_perimeter(single_channel, thresh)
        details["radius"] = self._get_radius(single_channel, thresh)
        return details