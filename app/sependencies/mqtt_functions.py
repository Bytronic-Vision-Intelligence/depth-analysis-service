from mqtt_client import MQTTClient, MQTTConfig
import threading
from base64 import b64decode
from cv2 import IMREAD_UNCHANGED, imdecode
from numpy import frombuffer, uint8
from queue import Queue

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

def start_subscribe_thread(ip: str, port: int, topic: str, queue: Queue, stop_event: threading.Event) -> threading.Thread:
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
    if depth_image is None:
        raise ValueError("Error : The image payload could not be decoded by OpenCV.")
    if not len(depth_image.shape) > 2:
        raise ValueError("Error : image not valid shape")
    return depth_image