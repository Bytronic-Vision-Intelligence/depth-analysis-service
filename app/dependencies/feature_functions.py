from numpy import ndarray

class FeatureExtraction():
    '''a simple class handler for feature extraction and image processing'''

    def get_subject_details(image:ndarray):
        '''extracts the depth, area and perimeter from details found in an image
        Args:
            image: a np.ndarray representing an image
        Returns:
            details: a dictionary containing details of the image'''
        
        details = dict()
        
        details["depth"] = image.max() - image.min()
        
        return details