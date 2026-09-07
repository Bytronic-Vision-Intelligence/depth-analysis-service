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
import argparse
#
MQTT_BROKERS = None
TOPICS = None
IMAGE_DETAILS = None
DATABASE_DETAILS = None

def _check_for_triggers(trigger:dict, is_blocking:bool=False, timeout:float = 10):
    '''Checks the queue for each of the trigger topics and returns the message when any of them have received one
    Args:
        triggers: a dictionary of topics
        is_blocking: a boolean value that controls the blocking functionality
        timout: a float that determines the timout in s
    Returns:
        message: the message received from the trigger as dictionary'''

    if not trigger:
        raise ValueError("Error : trigger cannot be empty")
    message = {"image": None}

    if is_blocking:
        message = loads(trigger["queue"].get(timeout=timeout))
        return message
    
    try:
        message = loads(trigger["queue"].get_nowait())
    except Exception as e:
        if e != KeyError: info(f"error occured when checking for trigger {e}")

    return message

def _message_database(is_search:bool, details:dict, client:MQTTClient):
    '''sends command to the broker for the database service
    Args: 
        is_search: a bool that determines if the intended command is search or add
        details: a dict containing the search details
        client: the mqtt client used to send the data to the broker
    '''
    if not details:
        raise ValueError("Error : details cannot be empty")
    if not client.is_connected:
        raise ConnectionError("Error : client is not connected")

    details["command"] = "add_to_database"
    if is_search: details["command"] = "search_database"

    details["destination"] = DATABASE_DETAILS["database_table"]
    details["database_name"] = DATABASE_DETAILS["database_name"]

    target = next((t for t in TOPICS if t.get("name") == "send_depth_analysis"), None)
    if target:
        client.publish(target["topic"], dumps(details))
        print("String published")

def depth_analysis(message:dict):
    '''The depth analysis worker captures depth data from a pointcloud image and extracts the radius, perimeter and depth of the subject
    the data is then sent to a database service over MQTT
    Args:
        message: the MQTT message dictionary
        client: the mqtt client used to send and receive messages
        trigger_name: the name of the trigger.'''
    #extract parameters to simplifly search
    if not message:
        raise ValueError("Error : message cannot be empty")
    
    feature_extractor = FeatureExtraction([7,7])

    image_decoded = extract_image(message["image"])
    image = Image(image_decoded, IMAGE_DETAILS["region_of_interest"], IMAGE_DETAILS["trim_value"])
    depth_image = image.cropped_image
    if depth_image.shape[2] > 2:
        depth_image = depth_image[:,:,0]
    
    details = feature_extractor.get_subject_details(depth_image)
    print(f"radius of the plate: {details['radius']}, perimeter of the plate: {details['perimeter']}, depth of the plate: {details['depth']}")
    return details

def send_details(details:dict, client:MQTTClient, hmi_command:str):
    '''sends extracted details to an mqtt broker with an appropriate command and awaits a response
    Args:
        details: a dict containing the details to be sent to the database, radius:float, perimeter:float, depth:float
        client: an mqtt client object
        hmi_command: a command sent from the HMI containing a string reading "search_database" or "add_to_database
    Returns:
        bool: a pass fail response based on the database result
    "'''
    if not details:
        raise ValueError("Error : details cannot be empty")
    if not client.is_connected:
        raise ConnectionError("Error : client is not connected")
    if hmi_command == "":
        raise ValueError("Error : hmi_command cannot be empty")

    is_search = hmi_command == "search_database"
    _message_database(is_search, details, client)
    database_result = {"waiting_for_result": None}

    target = next((t for t in TOPICS if t.get("name") == "receive_analysis_results"), None)
    while "image" in database_result: 
        database_result = _check_for_triggers(target)
        time.sleep(0.1)

    if not "radius" in database_result:
        info("Error: no match found in database")
        print("Error: no match found in database")
        return False
    return True

def main(config_path: str | None = None):
    global MQTT_BROKERS, TOPICS, DATABASE_DETAILS, IMAGE_DETAILS
    loadConfig.set_config_path(config_path)
    MQTT_BROKERS = loadConfig.return_config_value("mqtt")
    TOPICS = loadConfig.return_config_value("topics")
    IMAGE_DETAILS = loadConfig.return_config_value("image_options")
    DATABASE_DETAILS = loadConfig.return_config_value("database_options")
    config = MQTTConfig(host=MQTT_BROKERS["mqtt_ip"], port=MQTT_BROKERS["mqtt_port"])
    client = MQTTClient(config)
    client.connect()

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
            target = next((t for t in TOPICS if t.get("name") == "receive_depth_image"), None)
            message = _check_for_triggers(target)

            if "image" in message:
                if message["image"] is None:
                    continue
            else:
                continue

            details = depth_analysis(message)

            target = next((t for t in TOPICS if t.get("name") == "receive_hmi_instruction"), None)
            message = _check_for_triggers(target, True)

            send_details(details, client, message["hmi_instruction"])

    except KeyboardInterrupt:
        print("Shutting down subscribe listener and exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection analysis service")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use fallback config (app/configs/config.yaml)",
    )
    args = parser.parse_args()

    if args.test and args.config:
        parser.error("cannot use both --test and --config")
    if not args.test and not args.config:
        parser.error("one of --config or --test is required")

    config_path = None if args.test else args.config
    raise SystemExit(main(config_path=config_path))