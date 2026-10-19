from .base_functions.state import State
from dependencies.mqtt.mqtt_functions import check_for_messages
from .image_prep import ImagePrepState

class IdleState(State):
    ''''''
    def __init__(self, state_machine, state_id = "Idle State"):
        super().__init__(state_machine, state_id)

    def tick(self):
        '''Waits for a message to be sent from the camera before moving to the image prep state'''
        self.__mqtt_watchdog()
        message = check_for_messages(self.my_state_machine.ui_request)
        instruction = message.get("image")
        if instruction == None: return

        self.my_state_machine.change_state(
            ImagePrepState(self.my_state_machine, message)
        )

    def __mqtt_watchdog(self):
        '''a simple process that checks the state of the client and 
        reconnects if the beoker disconnects due to being idle'''
        if self.my_state_machine.mqtt_client.is_connected: return
        print("Info : mqtt client disconnected, reconnecting")
        self.my_state_machine.mqtt_client.connect()