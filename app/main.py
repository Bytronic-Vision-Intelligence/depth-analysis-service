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
DATABASE_DETAILS = loadConfig.return_config_value("database_options")

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

def main():
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
            message = check_for_triggers(target)

            if "image" in message:
                if message["image"] is None:
                    continue
            else:
                continue

            details = depth_analysis(message)

            target = next((t for t in TOPICS if t.get("name") == "receive_hmi_instruction"), None)
            message = check_for_triggers(target, True)

            send_details(details, client, message["database_instruction"])

    except KeyboardInterrupt:
        print("Shutting down subscribe listener and exiting.")

if __name__ == "__main__":
    main()