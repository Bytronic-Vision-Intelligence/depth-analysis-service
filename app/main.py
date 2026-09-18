from dependencies.mqtt_functions import *
from dependencies.feature_functions import FeatureExtraction
from dependencies.image_functions import Image
from dependencies.image_functions import decode_image_from_bytes, extract_image

from dependencies import loadConfig
import time
import threading
from mqtt_client import MQTTClient, MQTTConfig
from numpy import unique
from logging import info
#
MQTT_BROKERS = loadConfig.return_config_value("broker_details")
TOPICS = MQTT_BROKERS["topics"]
IMAGE_DETAILS = loadConfig.return_config_value("image_options")

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

    image_decoded = decode_image_from_bytes(extract_image(message["image"]))
    pixel_range = len(unique(image_decoded))
    if pixel_range == 1: return False
    image = Image(image_decoded, IMAGE_DETAILS["region_of_interest"], IMAGE_DETAILS["trim_value"])
    depth_image = image.cropped_image
    if len(depth_image.shape) > 2:
        depth_image = depth_image[:,:,0]
    
    details = feature_extractor.get_subject_details(depth_image)
    return details

def main():
    config = loadConfig.get_config()
    service_id = config.get("service_id", "depth_analysis_1")
    topics = TOPICS

    info(f"INFO : {service_id} starting \n\r")
    config = MQTTConfig(host=MQTT_BROKERS["mqtt_ip"], port=MQTT_BROKERS["mqtt_port"])
    client = MQTTClient(config)
    client.connect()

    topics = create_topic_listners(MQTT_BROKERS, topics)
    target_topic = next((t for t in topics if t.get("name") == "receive_depth_image"), None)
    try:
        while True:
            time.sleep(0.1)
            message = check_for_triggers(target_topic)

            if "image" in message:
                if message["image"] is None:
                    continue
            else:
                continue
            details = depth_analysis(message)
        
            if details == False: continue

            send_details(
                details, 
                client,
                topics
            )

    except KeyboardInterrupt:
        info(f"Info: {service_id} Shutting down subscribe listener and exiting.")

if __name__ == "__main__":
    main()