from numpy import ndarray


class FeatureExtraction():
    '''a simple class handler for feature extraction and image processing'''

    @staticmethod
    def get_subject_details(image: ndarray) -> dict:
        '''extracts the depth, area and perimeter from details found in an image
        Args:
            image: a np.ndarray representing an image
        Returns:
            details: a dictionary containing details of the image'''

        details = dict()

        # float(), not the numpy scalar the subtraction returns. The result is
        # published as JSON, and json.dumps refuses a numpy.int64 with
        # "Object of type int64 is not JSON serializable" -- which would be
        # raised on every frame this service successfully measured.
        details["depth"] = float(image.max() - image.min())

        return details
