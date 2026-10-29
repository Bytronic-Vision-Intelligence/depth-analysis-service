from .base_functions.state import State
from .publish_state import PublishState
from dependencies.feature_functions import FeatureExtraction

class AnalysisState(State):
    ''''''
    def __init__(self, state_machine,image, glare_mask = None, state_id = "Analysis State"):
        super().__init__(state_machine, state_id)
        self.image = image
        self.glare_mask = glare_mask
        self.extractor = FeatureExtraction(state_machine.pixel_pitch_mm)
    def tick(self):
        ''''''
        image_details = self.extractor.get_subject_details(
            self.image,
            self.glare_mask
        )
        if image_details == False:
            self._return_idle()
            return

        self.my_state_machine.change_state(
            PublishState(self.my_state_machine, image_details)
        )
