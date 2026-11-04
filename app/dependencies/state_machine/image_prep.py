from .base_functions.state import State
from .analysis_state import AnalysisState
from dependencies.image_functions import Image, decode_image_text
from dependencies.cycle_timing import mark, report
from logging import info
class ImagePrepState(State):
    ''''''
    def __init__(self, state_machine,message, state_id = "Image Prep State"):
        super().__init__(state_machine, state_id)
        self.message = message

    def tick(self):
        '''prepares the image for analysis, creating the iamge object as it does'''

        image_decoded = decode_image_text(self.message.get("image"))
        mark(self.my_state_machine, "decode")
        if image_decoded.min() == image_decoded.max():
            report(self.my_state_machine, "single-value image")
            self._return_idle()
            info("Info : image is a single value, nothing to analyse")
            return

        image_object = Image(
            image_decoded,
            self.my_state_machine.region_of_interest
        )
        mark(self.my_state_machine, "crop+spikes")
        depth_image = image_object.cropped_image
        depth_image_dimentions = len(depth_image.shape)
        if depth_image_dimentions >= 3:
            depth_image = depth_image[:,:,0]

        self.my_state_machine.change_state(
            AnalysisState(self.my_state_machine, depth_image, image_object.spike_mask)
        )
