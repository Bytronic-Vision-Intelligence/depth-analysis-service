from .base_functions.state_machine import StateMachine
from dependencies.mqtt.mqtt_functions import MQTTClient, MQTTConfig
from dependencies.mqtt.topic import Topic
from .idle_state import IdleState

class DepthAnalysisSM(StateMachine):
    ''''''
    def __init__(self, config:dict):
        super().__init__()
        self.service_id = config.get("service_id", "database_1")
        self.mqtt_client = self.__create_mqtt_client(config)
        self.topics = self.__create_topic_listners(config)
        self.blocking_subscribers = self.__get_blocking_subscribers(self.topics)
        self.ui_request = self.__get_topic(self.topics,"receive_depth_image")
        self.service_output = self.__get_topic(self.topics, "send_depth_analysis")
        
        image_options = config.get("image_options")
        self.region_of_interest = image_options.get("region_of_interest")
        self.trim_value = image_options.get("trim_value")
        self.state = IdleState(self)

    @staticmethod
    def __create_topic_listners(config:dict):
        '''creates a set of listners for the given topics and adds
        qeueue objects to the dictionary as well as reference to the thread
        '''
        broker = config.get('broker_details')
        topics = config.get("topics")
        topic_list = {}
        for topic in topics:
            topic_new = Topic(broker, topic)
            topic_list[f"{topic.get('name')}"] = topic_new
        return topic_list

    @staticmethod
    def __create_mqtt_client(config:dict):
        '''creates an mqtt client object from the config'''
        broker_details = config.get("broker_details")
        
        mqtt_config = MQTTConfig(
            host=broker_details["mqtt_ip"], 
            port=broker_details["mqtt_port"]
        )
        client = MQTTClient(mqtt_config)
        client.connect()
        return client

    @staticmethod
    def __get_topic(topic_dict:dict, topic_name:str)-> Topic:
        '''returns the topic associated with the key value
        Args:
            topic_dict: a dictionary of topics, one value needs to be "name"
            topic_name: the name to search within the topic list
        Returns:
            the topic object accosiated with the name
        '''
        return topic_dict.get(topic_name)

    @staticmethod
    def __get_blocking_subscribers(topics):
        '''returns a list of topics that are not triggers and will block
        functionality until data is received
        Args:
            config: the microservice config containing a correctly formatted topics section
        returns
            a list of topics
        '''
        blocking_subscribers = []
        for topic in topics:
            topic = topics.get(topic)
            if not topic.is_subscribe or topic.is_trigger: continue
            blocking_subscribers.append(topic)
        return blocking_subscribers