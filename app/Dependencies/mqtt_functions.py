from mqtt_client import MQTTClient, MQTTConfig
import threading
from queue import Queue
from json import loads, dumps, JSONDecodeError
from logging import info
from time import sleep

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

    _message_database(details, client, topics)

def _message_database(
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

    target = next((t for t in topics if t.get("name") == "send_depth_analysis"), None)
    if target:
        client.publish(target["topic"], dumps(details))
        print("String published")