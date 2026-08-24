from dependencies.mqtt_functions import *
from dependencies.feature_functions import FeatureExtraction

from dependencies import loadConfig
import time
import threading
from queue import  Queue
from mqtt_client import MQTTClient, MQTTConfig
from json import loads
from logging import info
#
MQTT_BROKERS = loadConfig.get_config("mqtt_options")
TOPICS = loadConfig.get_config("topics")

def _check_for_triggers(triggers:dict):
    '''Checks the queue for each of the trigger topics and returns the message when any of them have received one
    Args:
        triggers: a dictionary of topics
    Returns:
        message: the message received from the trigger as dictionary'''
    message = {"command": None}

    for trigger in triggers:
        try:
            if not trigger["is_trigger"]:continue
        except:
            continue

        try:
            message = loads(trigger["queue"].get_nowait())
        except Exception as e:
            if e != KeyError: info(f"error occured when checking for trigger {e}")
            continue

    return message

def worker_process_function(msg, client:MQTTClient):
    #extract parameters to simplifly search
    depth_image = loads(msg)["depth_image"]
    details = FeatureExtraction.get_subject_details(depth_image)
    print(f"radius of the plate: {details["radius"]}, perimeter of the plate: {details["perimeter"]}, depth of the plate: {details["depth"]}")
    details["command"] = "search_phrase"
    details["destination"] = "sku_table"
    details["database_name"] = "churchill_database"
    
    client.publish(details)
    #create feature list using ORB

    #request a list of skus that match the parameters extracted earlier (seperate subscribe service, 
    # this is a blocking function, cannot continue until response or timeout)

    #compare the list to the features

    #publish list of matches
    
def main():
    config = MQTTConfig(host=MQTT_BROKERS["mqtt_ip"], port=MQTT_BROKERS["mqtt_port"])
    client = MQTTClient(config)
    client.connect()

    event_queue = Queue()
    stop_event = threading.Event()
    for topic in TOPICS:
        if not topic["is_subscribe"]:
            continue
        topic["queue"] = Queue()
        
        topic["thread"] = start_subscribe_thread(
            MQTT_BROKERS["mqtt_ip"], 
            MQTT_BROKERS["mqtt_port"], 
            topic["topic"], 
            topic["queue"],
            stop_event
        )

    try:
        while True:
            time.sleep(0.1)
            message = _check_for_triggers(TOPICS)
            if message is None:
                continue

            worker_process_function(message)

    except KeyboardInterrupt:
        print("Shutting down subscribe listener and exiting.")

if __name__ == "__main__":
    main()