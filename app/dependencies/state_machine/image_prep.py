from .base_functions.state import State
from .analysis_state import AnalysisState
from dependencies.image_functions import Image, decode_image_from_bytes, extract_image, decode_raw_height_png
from numpy import unique
from logging import info
from cv2 import imshow, waitKey
class ImagePrepState(State):
    ''''''
    def __init__(self, state_machine,message, state_id = "Image Prep State"):
        super().__init__(state_machine, state_id)
        self.message = message

    def tick(self):
        '''prepares the image for analysis, creating the iamge object as it does'''

        image_encoded = extract_image(self.message.get("image"))

        image_decoded = decode_image_from_bytes(image_encoded)
        pixel_range = len(
            unique(image_decoded)
        )
        if pixel_range == 1: 
            self._return_idle()
            info(f"Info : Pixel range of {pixel_range} is not valid at least 3 values are required ")
            return
        
        image_object = Image(
            image_decoded,
            self.my_state_machine.region_of_interest,
            self.my_state_machine.trim_value
        )
        depth_image = image_object.cropped_image
        depth_image_dimentions = len(depth_image.shape)
        if depth_image_dimentions >= 3:
            depth_image = depth_image[:,:,0]

        self.my_state_machine.change_state(
            AnalysisState(self.my_state_machine, depth_image)
        )