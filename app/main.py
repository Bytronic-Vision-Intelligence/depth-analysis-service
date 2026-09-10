import time
from json import JSONDecodeError, dumps, loads
from logging import info, warning
from queue import Empty, Queue
from threading import Event

from mqtt_client import MQTTClient, MQTTConfig
from mqtt_client.exceptions import MQTTClientError

from dependencies import loadConfig, logging_setup
from dependencies.feature_functions import FeatureExtraction
from dependencies.image_functions import Image, extract_image
from dependencies.mqtt_functions import start_subscribe_thread

#: Blur kernel handed to FeatureExtraction. Both values must be odd.
BLUR_KERNEL = [7, 7]

#: Used when the config does not say. Both are deliberately not required:
#: every config written before these existed must still start.
DEFAULT_INSTRUCTION_TIMEOUT = 10
DEFAULT_DATABASE_TIMEOUT = 10

#: The topics this service reads from, by name. Resolved at startup so a
#: renamed or missing entry refuses there rather than at the first message.
INPUT_TOPICS = (
    "receive_depth_image",
    "receive_hmi_instruction",
    "receive_analysis_results",
)


def require(config: dict, key: str):
    """Return a required top-level config value, or exit describing what is missing.

    Args:
        config: the loaded configuration mapping.
        key: the top-level key the service cannot start without.
    Returns:
        the value stored under `key`.
    Raises:
        SystemExit: when `key` is absent, naming both the key and the file.
    """
    # `is None` as well as absent: a key present but empty is a section
    # somebody meant to fill in, and letting it through moves the failure to
    # whatever first subscripts it.
    if key not in config or config[key] is None:
        # Named only if one has been loaded. require() is also called on
        # nested sections in contexts that never parsed arguments, and
        # config_path() refuses to guess there -- which would replace this
        # message with one about the wrong problem entirely.
        try:
            where = f" in {loadConfig.config_path()}"
        except SystemExit:
            where = ""
        raise SystemExit(f"Missing required config key '{key}'{where}")
    return config[key]


def start_subscribers(mqtt_config: dict, topics: list, stop_event: Event) -> list:
    """Start one listener thread per subscribed topic.

    Each topic entry with `is_subscribe` true gains a `queue` key, which
    `next_trigger` later reads from.

    Args:
        mqtt_config: the `mqtt` section, carrying mqtt_ip and mqtt_port.
        topics: configured topic entries; mutated in place to carry queues.
        stop_event: shared shutdown signal handed to every listener.
    Returns:
        threads: the started listener threads.
    """
    threads = []
    for topic in topics:
        if not topic.get("is_subscribe"):
            continue
        topic["queue"] = Queue()
        threads.append(
            start_subscribe_thread(
                mqtt_config["mqtt_ip"],
                mqtt_config["mqtt_port"],
                topic["topic"],
                topic["queue"],
                stop_event,
            )
        )
    return threads


def topic_named(topics: list, name: str) -> dict:
    """Return the configured topic entry called `name`.

    Topics are matched by their `name`, not their position or their text, so a
    deployment can point this service anywhere without the code knowing which
    entry carries the depth image and which carries the database reply.

    Args:
        topics: the `mqtt.topics` entries.
        name: the `name:` to find.
    Returns:
        the entry itself, so callers can reach both its topic and its queue.
    Raises:
        SystemExit: when no entry carries that name, or it carries no topic.
            Returning None here instead is how the previous version published
            nothing, to nowhere, without reporting anything.
    """
    for topic in topics:
        if topic.get("name") != name:
            continue
        if not topic.get("topic"):
            raise SystemExit(f"Topic '{name}' has no `topic:` value")
        return topic
    raise SystemExit(
        f"No topic named '{name}'. depth-analysis-service looks its topics up "
        f"by name; add `- name: {name}` under mqtt.topics.")


def read(topic: dict, timeout: float = None):
    """Return the next payload waiting on one topic, or None.

    This service reads three named topics in a fixed order -- an image, then
    the operator's instruction, then the database's answer -- so it must take
    from the queue it is waiting on and no other. A poll across every queue
    would consume an instruction as though it were an image.

    Args:
        topic: a subscribed topic entry, after `start_subscribers` has run.
        timeout: seconds to wait. None returns immediately.
    Returns:
        message: the decoded payload, or None when nothing arrived in time or
            the payload was not valid JSON.
    """
    try:
        if timeout:
            payload = topic["queue"].get(timeout=timeout)
        else:
            payload = topic["queue"].get_nowait()
    except Empty:
        return None
    try:
        return loads(payload)
    except (JSONDecodeError, TypeError) as exc:
        warning(f"Discarding malformed payload on {topic['topic']}: {exc}")
        return None


def depth_analysis(message: dict, settings: dict):
    """Measure the subject in a depth image.

    Args:
        message: the decoded payload from the depth image topic.
        settings: the `service` section, carrying the region of interest and
            the trim value.
    Returns:
        details: what was measured, or None when the message carried nothing
            measurable. A frame this service cannot read is a frame lost, not
            a reason to stop reading the ones after it.
    """
    encoded = message.get("image")
    if encoded is None:
        return None

    try:
        image = extract_image(encoded)
    except ValueError as exc:
        warning(f"Unreadable image payload: {exc}")
        return None

    prepared = Image(image, settings["region_of_interest"], settings["trim_value"])
    depth_image = prepared.cropped_image
    if depth_image.ndim > 2 and depth_image.shape[2] > 2:
        depth_image = depth_image[:, :, 0]

    details = FeatureExtraction(BLUR_KERNEL).get_subject_details(depth_image)
    info("Measured radius %s, perimeter %s, depth %s",
         details["radius"], details["perimeter"], details["depth"])
    return details


def send_details(client: MQTTClient, topics: list, settings: dict,
                 details: dict, instruction: str) -> bool:
    """Ask the database to match or store a measurement, and wait for its answer.

    Args:
        client: connected MQTT client.
        topics: the `mqtt.topics` entries.
        settings: the `service` section, naming the database and its table.
        details: what `depth_analysis` measured.
        instruction: what the operator asked for; "search_database" searches,
            anything else adds.
    Returns:
        True when the database answered with a match, False when it answered
        without one or did not answer in time.
    """
    request = topic_named(topics, "send_depth_analysis")
    reply = topic_named(topics, "receive_analysis_results")

    # Copied, not tagged in place: the caller's measurement is its own thing,
    # and the database fields belong to this message only.
    payload = dict(details)
    payload["command"] = (
        "search_database" if instruction == "search_database" else "add_to_database")
    payload["destination"] = settings["database_table"]
    payload["database_name"] = settings["database_name"]

    try:
        client.publish(request["topic"], dumps(payload))
    except MQTTClientError as exc:
        # One failed publish loses this measurement. Letting it out of the
        # loop would lose every measurement after it too.
        warning(f"Could not publish to {request['topic']}: {exc}")
        return False

    timeout = settings.get("database_timeout", DEFAULT_DATABASE_TIMEOUT)
    result = read(reply, timeout=timeout)
    if result is None:
        warning(f"No answer on {reply['topic']} within {timeout}s")
        return False
    if "radius" not in result:
        info("No match found in the database")
        return False
    return True


def main(argv=None):
    args = loadConfig.parse_cli(argv)

    config = loadConfig.get_config(args.config)

    log_settings = config.get("logging") or {}
    logging_setup.configure(log_settings.get("level", logging_setup.DEFAULT_LEVEL))

    mqtt_config = require(config, "mqtt")
    topics = require(mqtt_config, "topics")
    settings = require(config, "service")

    # Every topic this service names, resolved before anything connects. A
    # renamed entry then costs a startup instead of a silent no-op later.
    inputs = {name: topic_named(topics, name) for name in INPUT_TOPICS}
    topic_named(topics, "send_depth_analysis")

    client = MQTTClient(
        MQTTConfig(host=mqtt_config["mqtt_ip"], port=mqtt_config["mqtt_port"]))
    client.connect()

    stop_event = Event()
    threads = start_subscribers(mqtt_config, topics, stop_event)

    for name, topic in inputs.items():
        if "queue" not in topic:
            raise SystemExit(
                f"Topic '{name}' is read by this service but is not subscribed. "
                f"Set `is_subscribe: true` on it under mqtt.topics.")

    instruction_timeout = settings.get(
        "instruction_timeout", DEFAULT_INSTRUCTION_TIMEOUT)

    try:
        while True:
            time.sleep(0.1)

            message = read(inputs["receive_depth_image"])
            if message is None:
                continue

            details = depth_analysis(message, settings)
            if details is None:
                continue

            instruction = read(inputs["receive_hmi_instruction"],
                               timeout=instruction_timeout)
            if instruction is None:
                # Was an uncaught queue.Empty, which unwound past the
                # KeyboardInterrupt handler and stopped the service outright.
                warning(f"No instruction within {instruction_timeout}s; "
                        f"discarding this measurement")
                continue

            requested = instruction.get("database_instruction")
            if requested is None:
                warning("Instruction carried no 'database_instruction'; "
                        "discarding this measurement")
                continue

            send_details(client, topics, settings, details, requested)

    except KeyboardInterrupt:
        info("Shutting down subscribe listener and exiting.")
    finally:
        stop_event.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2)


if __name__ == "__main__":
    main()
