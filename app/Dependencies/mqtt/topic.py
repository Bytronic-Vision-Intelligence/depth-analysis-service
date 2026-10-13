from queue import Queue

class Topic():
    '''a class to host topics and their associated information making
    it easier to control multiple topics from a list'''
    def __init__(self,broker:dict, details:dict):
        self.broker = broker
        self.name = details["name"]
        self.topic = details["topic"]
        self.is_subscribe = details["is_subscribe"]
        self.is_trigger = details["is_trigger"]
        self.queue = Queue()
        self.thread = None
        self.timeout = details.get("timout", 10)
        if self.is_subscribe:
            self.__create_listner()

    def __create_listner(self):
        '''creates a listner thread and queue'''
        from .mqtt_functions import start_subscribe_thread
        self.thread = start_subscribe_thread(
            self.broker["mqtt_ip"], 
            self.broker["mqtt_port"], 
            self.topic, 
            self.queue,
            None
        )