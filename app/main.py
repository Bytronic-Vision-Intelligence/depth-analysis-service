from dependencies.mqtt_functions import *
from dependencies.feature_functions import FeatureExtraction

from dependencies import loadConfig
import time
import threading
from queue import  Queue
from mqtt_client import MQTTClient, MQTTConfig
from json import loads, dumps
from logging import info
from base64 import b64decode
from cv2 import IMREAD_UNCHANGED, imdecode, imwrite
from numpy import frombuffer, uint8
#
MQTT_BROKERS = loadConfig.return_config_value("mqtt_options")
TOPICS = MQTT_BROKERS["topics"]

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

def _extract_image(encoded_image):
    '''extracts the image from the encoded array into a ndarray
    Arga:
        encoded_image: a string containing the image packet encoded as base64
    Returns:
        image: an ndarray containing valid image data
    '''
    if isinstance(encoded_image, str) and "," in encoded_image:
        encoded_image = encoded_image.split(",", 1)[1]

    image_bytes = b64decode(encoded_image)
    depth_image = imdecode(frombuffer(image_bytes, dtype=uint8), IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("The image payload could not be decoded by OpenCV.")
    return depth_image

def worker_process_function(msg, client:MQTTClient):
    #extract parameters to simplifly search
    feature_extractor = FeatureExtraction()
    depth_image = _extract_image(msg["image"])
    
    details = feature_extractor.get_subject_details(depth_image)
    print(f"radius of the plate: {details['radius']}, perimeter of the plate: {details['perimeter']}, depth of the plate: {details['depth']}")
    details["command"] = "search_phrase"
    details["destination"] = "sku_table"
    details["database_name"] = "churchill_database"
    for topic in TOPICS:
        if not topic["is_subscribe"]: client.publish(topic["topic"], dumps(details))
    
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