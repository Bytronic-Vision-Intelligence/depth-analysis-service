from dependencies.mqtt_functions import *
from dependencies.feature_functions import FeatureExtraction
from dependencies.image_functions import Image

from dependencies import loadConfig
import time
import threading
from queue import  Queue
from mqtt_client import MQTTClient, MQTTConfig
from json import loads, dumps
from logging import info
#
MQTT_BROKERS = loadConfig.return_config_value("mqtt_options")
TOPICS = MQTT_BROKERS["topics"]
IMAGE_DETAILS = loadConfig.return_config_value("image_options")

def _check_for_triggers(triggers:dict):
    '''Checks the queue for each of the trigger topics and returns the message when any of them have received one
    Args:
        triggers: a dictionary of topics
    Returns:
        message: the message received from the trigger as dictionary'''
    message = {"image": None}

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

def _search_database(details:dict, client:MQTTClient):
    '''sends search command to the broker
    Args: 
        details: a dict containing the search details
        client: the mqtt client used to send the data to the broker
    '''
    details["command"] = "search_phrase"
    details["destination"] = "sku_table"
    details["database_name"] = "churchill_database"

    for topic in TOPICS:
        if not topic["is_subscribe"]: client.publish(topic["topic"], dumps(details))

def worker_process_function(msg, client:MQTTClient):
    #extract parameters to simplifly search
    feature_extractor = FeatureExtraction([7,7])

    image_decoded = extract_image(msg["image"])
    image = Image(image_decoded, IMAGE_DETAILS["region_of_interest"], IMAGE_DETAILS["trim_value"])
    depth_image = image.cropped_image[:,:,0]
    
    details = feature_extractor.get_subject_details(depth_image)
    print(f"radius of the plate: {details['radius']}, perimeter of the plate: {details['perimeter']}, depth of the plate: {details['depth']}")
    _search_database(details, client)
    
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
            if message["image"] is None:
                continue

            worker_process_function(message, client)

    except KeyboardInterrupt:
        print("Shutting down subscribe listener and exiting.")

if __name__ == "__main__":
    main()