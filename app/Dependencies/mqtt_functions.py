from mqtt_client import MQTTClient, MQTTConfig
import threading
from base64 import b64decode
from cv2 import IMREAD_UNCHANGED, imdecode, imwrite, imencode, convertScaleAbs
from numpy import frombuffer, uint8, asarray, nan, float32, ndarray, uint16
from queue import Queue
from json import loads, dumps, JSONDecodeError
from logging import info
from time import sleep

RAW_PNG_MM_SCALE=100.00
BIT_SCALE_15=32768.0

def subscribe_listener(ip: str, port: int, trigger_topic: str, result_queue: Queue, stop_event: threading.Event):
    """Connect to a broker and feed every message on `trigger_topic` into a queue.

    Args:
        ip: broker address.
        port: broker port.
        trigger_topic: the topic to watch.
        result_queue: queue used to hand payloads back to the main thread.
        stop_event: shared shutdown signal (reserved; the client owns its loop).
    """
    config = MQTTConfig(host=ip, port=port)
    client = MQTTClient(config)
    client.connect()

    def _on_message(topic: str, payload: str) -> None:
        """Hand a received payload to the main thread."""
        info("Request received:", topic)
        result_queue.put(payload)

    client.subscribe(trigger_topic, _on_message)


def start_subscribe_thread(ip: str, port: int, topic: str, queue: Queue, stop_event: threading.Event) -> threading.Thread:
    """Run `subscribe_listener` on a daemon thread.

    Args:
        ip: broker address.
        port: broker port.
        topic: the topic to watch.
        queue: queue used to hand payloads back to the main thread.
        stop_event: shared shutdown signal.
    Returns:
        thread: the started daemon thread.
    """
    thread = threading.Thread(
        target=subscribe_listener,
        args=(ip, port, topic, queue, stop_event),
        daemon=True,
    )
    thread.start()
    return thread

def create_topic_listners(broker:dict, topics:dict):
    '''creates a set of listners for the given topics and adds
    qeueue objects to the dictionary as well as reference to the thread
    '''
    for topic in topics:
        if not topic["is_subscribe"]:
            continue
        topic["queue"] = Queue()

        topic["thread"] = start_subscribe_thread(
            broker["mqtt_ip"], 
            broker["mqtt_port"], 
            topic["topic"], 
            topic["queue"],
            None
        )
    return topics

def create_subtopic_listners(
        subtopic_count:int, 
        topic:str, 
        name:str,
        broker_info:dict
    ):
    '''creates listners for a set of subtopics and returns a list of those subtopics
    Args:
        subtopic_count: the number of subtopics to create as an int
        topic: a string containing the main topic
        name: a string containing the topic identifier
        broker_info: a dictionary containing the ip address and the port for the broker
    Returns:
        a list of subtopics
    '''
    subtopic_list = []
    for i in range(0,subtopic_count):
        sub_topic = f"{topic}/{i}"
        sub_name = f"{name}_{i}",
        subtopic={
            **topic,
            "name": sub_topic,
            "topic": sub_name,
            "queue": Queue()
        }

        subtopic["thread"] = start_subscribe_thread(
            broker_info["mqtt_ip"], 
            broker_info["mqtt_port"], 
            subtopic["topic"], 
            subtopic["queue"],
            None
        )
        subtopic_list.append(subtopic)
        subtopic = None
    return subtopic_list

def packetize_results(root_message:any, results:dict, results_map:list[str], topic:str):
    '''splits results into packets and then wraps then appropriately for publication
    Args:
        results: a ditionary of results
        results_map: a list of strings containing relevent headings
    Returns:
        a list of dictionaries containing the topic to publish on and the data to publish
    '''
    packet_list = []
    packet_number = 0
    for candidate in results:
        if candidate.get("database_response") != None: 
            info(f"Error : No matches found, aborting packetization")
            return
        packet = dict()
        packet["topic"] = f"{topic}/{packet_number}"
        packet["payload"] = {
            "packet_number": packet_number,
        }
        for item in results_map:
            packet["payload"][results_map[item]] = results.get([results_map[item]])
        packet_number+=1
        packet_list.append(packet)
    return packet_list

def check_trigger(trigger:dict, is_blocking:bool=False, timeout:float = 10):
    '''Checks the queue for each of the trigger topics and returns the message when any of them have received one

    Args:
        triggers: a dictionary of topics
        is_blocking: a boolean value that controls the blocking functionality
        timout: a float that determines the timout in s
    Returns:
        message: the message received from the trigger as dictionary'''

    if not trigger:
        raise ValueError("Error : trigger cannot be empty")
    
    message = dict()

    if is_blocking:
        message = loads(trigger["queue"].get(timeout=timeout))
        return message
    
    if not "queue" in trigger: return message

    try:
        message = loads(trigger["queue"].get_nowait())
    except (JSONDecodeError, TypeError) as exc:
        info(f"Discarding malformed payload on {trigger['topic']}: {exc}")
        info(f"Discarding malformed payload on {trigger['topic']}: {exc}")
    except:
        return message

    return message

def encode_image_to_bytes(image: ndarray) -> bytes:
    """Encode the image as JPEG (8-bit) or PNG (uint16)."""
    if image.dtype == uint16:
        success, encoded_image = imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG format.")
        return encoded_image.tobytes()

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
    depth_image = imdecode(frombuffer(image_bytes, dtype=uint16), IMREAD_UNCHANGED)
    status = imwrite('message_image.png', depth_image)
    if depth_image is None:
        raise ValueError("Error : The image payload could not be decoded by OpenCV.")
    return depth_image

def extract_image(encoded_image: str) -> bytes:
    """Decode base64 image bytes from a camera packet."""
    encoded = str(encoded_image).strip()
    if encoded.lower().startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    encoded = "".join(encoded.split())
    padding = (-len(encoded)) % 4
    if padding:
        encoded += "=" * padding
    return b64decode(encoded, validate=False)

def decode_image_from_bytes(data: bytes) -> ndarray:
    """Decode image bytes into an ndarray."""
    if not data:
        raise ValueError("Empty image bytes.")

    image = imdecode(frombuffer(data, uint8), IMREAD_UNCHANGED).astype('uint8')
    if image is None:
        raise ValueError("Could not decode image bytes.")
    return image

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

def send_details(
        details:dict, 
        client:MQTTClient, 
        hmi_command:str,
        topics:dict
    ):
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
    _message_database(is_search, details, client, topics)
    database_result = {"waiting_for_result": None}

    target = next((t for t in topics if t.get("name") == "receive_analysis_results"), None)
    while "image" in database_result: 
        database_result = check_for_triggers(target)
        sleep(0.1)

    if not "radius" in database_result:
        info("Error: no match found in database")
        print("Error: no match found in database")
        return False
    return True

def _message_database(
        is_search:bool, 
        details:dict, 
        client:MQTTClient,
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

    target = next((t for t in topics if t.get("name") == "send_depth_analysis"), None)
    if target:
        client.publish(target["topic"], dumps(details))
        print("String published")