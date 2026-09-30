from dependencies.state_machine.base_functions.state import State
from json import dumps
class PublishState(State):
    ''''''
    def __init__(self, state_machine,packet, state_id = "Publish State"):
        super().__init__(state_machine, state_id)
        self.packet = packet

    def tick(self):
        ''''''
        mqtt_client = self.my_state_machine.mqtt_client
        mqtt_client.publish(
            self.my_state_machine.service_output.topic,
            dumps(self.packet)
        )
        self._return_idle()