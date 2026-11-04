from dependencies.state_machine.base_functions.state import State
from json import dumps
from dependencies.cycle_timing import mark, report
class PublishState(State):
    ''''''
    def __init__(self, state_machine,packet, state_id = "Publish State"):
        super().__init__(state_machine, state_id)
        self.packet = packet

    def enter(self):
        super().enter()
        if not self.my_state_machine.mqtt_client.is_connected:
            self.my_state_machine.mqtt_client.connect()
        return

    def tick(self):
        ''''''
        mqtt_client = self.my_state_machine.mqtt_client
        self.packet["message_id"] = self.my_state_machine.message_id
        mqtt_client.publish(
            self.my_state_machine.service_output.topic,
            dumps(self.packet)
        )
        mark(self.my_state_machine, "publish")
        report(self.my_state_machine, "published")
        self._return_idle()