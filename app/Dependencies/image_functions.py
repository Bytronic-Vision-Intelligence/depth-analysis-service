from numpy import ndarray, unique, median, clip
from cv2 import imwrite

class Image():
    '''A class containing image processing functions'''
    def __init__(self, image:ndarray, region_of_interest:list):

        self.region_of_interest = region_of_interest

        self.image = image
        self.trimmed_image = self._trim_min_max(image)
        self.cropped_image = self._crop_image(self.trimmed_image,self.region_of_interest)
        
        imwrite("img.png", self.cropped_image)


    def _trim_min_max(self, image:ndarray)->ndarray:
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

        mask = (image < median_value * 1.05) & (image > median_value * 0.95)
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
        print(cropped_image)
        return cropped_image