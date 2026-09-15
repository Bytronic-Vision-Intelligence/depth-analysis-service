from mqtt_client import MQTTClient, MQTTConfig
import threading
from base64 import b64decode
from cv2 import IMREAD_UNCHANGED, imdecode, imwrite
from numpy import frombuffer, uint8, asarray, nan, float32, ndarray
from queue import Queue
from json import loads
from logging import info

RAW_PNG_MM_SCALE=100.00
BIT_SCALE_15=32768.0

def subscribe_listener(ip: str, port: int, trigger_topic: str, result_queue: Queue, stop_event: threading.Event):
    config = MQTTConfig(host=ip, port=port)
    client = MQTTClient(config)
    client.connect()

    def on_message(topic: str, payload: str) -> None:
        # Handler signature used by mqtt_client.MQTTClient.subscribe
        try:
            decoded = payload
        except Exception:
            decoded = payload
        print("Capture request received:", topic)
        result_queue.put(decoded)

    client.subscribe(trigger_topic, on_message)

def start_subscribe_thread(
        ip: str, 
        port: int, 
        topic: str, 
        queue: Queue, 
        stop_event: threading.Event
    ) -> threading.Thread:
    thread = threading.Thread(
        
        target=subscribe_listener,
        args=(ip, port, topic, queue, stop_event),
        daemon=True,
    )
    thread.start()
    return thread

def extract_image(encoded_image):
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
    status = imwrite('message_image.png', depth_image)
    if depth_image is None:
        raise ValueError("Error : The image payload could not be decoded by OpenCV.")
    # if len(depth_image.shape) == 2:
    #     depth_image=decode_raw_height_png(depth_image)
    return depth_image

def decode_raw_height_png(image: ndarray) -> ndarray:
    """Decode a raw height PNG (from :func:`prepare_raw_png` float path) to mm."""
    arr = asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    mm = (arr.astype(float32) - BIT_SCALE_15) / float32(RAW_PNG_MM_SCALE)
    mm[arr == 0] = nan
    return mm

def check_for_triggers(trigger:dict, is_blocking:bool=False, timeout:float = 10):
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
        database_result = check_for_triggers(target)
        time.sleep(0.1)

    if not "radius" in database_result:
        info("Error: no match found in database")
        print("Error: no match found in database")
        return False
    return True

def _message_database(
        is_search:bool, 
        details:dict, 
        client:MQTTClient, 
        database_details:dict, 
        topics:dict
    ):
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

    details["destination"] = database_details["database_table"]
    details["database_name"] = database_details["database_name"]

    target = next((t for t in topics if t.get("name") == "send_depth_analysis"), None)
    if target:
        client.publish(target["topic"], dumps(details))
        print("String published")