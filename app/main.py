from dependencies.mqtt_functions import *
from dependencies.feature_functions import *
from dependencies.pointcloud_functions import *

from dependencies import loadConfig
import time
import threading
from queue import Empty, Queue
from mqtt_client import MQTTClient, MQTTConfig
#
IP = loadConfig.return_config_value("ip")
PORT = loadConfig.return_config_value("port")
TRIGGER_TOPIC = loadConfig.return_config_value("listen_topic")
OUTPUT_TOPIC = loadConfig.return_config_value("publish_topic") #this should be adjusted to suit your worker requirements

def _display_data(pointcloud):
    '''a helper function that displays the pointcloud data so it can be interpreted and debugged easier
    Args: 
        pointcloud: the pointcloud data, a list of touples containing the x, y and z of each point
        '''
    
    # Open3D visualization expects an Open3D geometry object, not a raw numpy array.
    point_cloud_for_display = o3d.geometry.PointCloud()
    point_cloud_for_display.points = o3d.utility.Vector3dVector(np.asarray(pointcloud, dtype=np.float64))
    try:
        o3d.visualization.draw_geometries([point_cloud_for_display],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
    except Exception as draw_error:
        print(f"Visualization skipped: {draw_error}")

def worker_process_function(msg):
    print("insert your program here")

    #process the data, create a noiseless pointcloud
    pointcloud = message_to_pointcloud(msg)
    #pointcloud = denoise_data(pointcloud, 2)

    #extract parameters to simplifly search
    perimeter = get_subject_perimeter(pointcloud)
    radius = get_subject_radius(pointcloud)
    depth = get_subject_depth(pointcloud)
    print(f"radius of the plate: {radius}, perimeter of the plate: {perimeter}, depth of the plate: {depth}")

    #create feature list using ORB

    #request a list of skus that match the parameters extracted earlier

    #compare the list to the features

    #publish matches
    
def main():
    config = MQTTConfig(host=IP, port=PORT)
    client = MQTTClient(config)
    client.connect()

    event_queue = Queue()
    stop_event = threading.Event()
    subscribe_thread = start_subscribe_thread(IP, PORT, TRIGGER_TOPIC, event_queue, stop_event)

    try:
        while True:
            time.sleep(0.1)

            try:
                msg = event_queue.get_nowait()
            except Empty:
                continue

            if msg is None:
                print("Received invalid trigger payload; ignoring.")
                continue

            worker_process_function(msg)

    except KeyboardInterrupt:
        print("Shutting down subscribe listener and exiting.")
    finally:
        stop_event.set()
        if subscribe_thread.is_alive():
            subscribe_thread.join(timeout=2)

if __name__ == "__main__":
    main()