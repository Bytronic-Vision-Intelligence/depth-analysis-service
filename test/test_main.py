import base64
import json
from queue import Queue

import cv2
import numpy as np
import pytest

from fakes import FakeMQTTClient, FakeMQTTConfig, FakeThread

import main


def make_topics():
    """The shape shipped in config.example.yaml."""
    return [
        {"name": "receive_depth_image", "topic": "churchill/camera/depth/image",
         "is_subscribe": True, "is_trigger": True},
        {"name": "receive_hmi_instruction", "topic": "churchill/hmi/instruction",
         "is_subscribe": True, "is_trigger": True},
        {"name": "send_depth_analysis", "topic": "churchill/database/search/sku_data",
         "is_subscribe": False, "is_trigger": False},
        {"name": "receive_analysis_results",
         "topic": "churchill/database/search/matching_sku",
         "is_subscribe": True, "is_trigger": True},
    ]


def subscribed(topics):
    """Give every subscribed entry the queue start_subscribers would give it."""
    for topic in topics:
        if topic.get("is_subscribe"):
            topic["queue"] = Queue()
    return topics


SETTINGS = {
    "region_of_interest": [0, 0, 2500, 3000],
    "trim_value": 0.0325,
    "database_name": "churchill_database",
    "database_table": "sku_table",
    "instruction_timeout": 0.05,
    "database_timeout": 0.05,
}


def depth_image_payload(square=200):
    """A depth image that survives trimming, encoded the way a camera sends one.

    The two stray pixels matter. _trim_min_max clips to the second-lowest and
    second-highest values present, so an image of just background and subject
    clips the subject away and leaves nothing to measure.
    """
    image = np.full((64, 64, 3), 20, dtype=np.uint8)
    image[0, 0] = 5
    image[0, 1] = 255
    image[20:44, 20:44] = square
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def test_the_blur_kernel_cannot_be_mutated_by_a_measurement():
    """FeatureExtraction stores the kernel it is given. A mutable module-level
    default would be one object shared by every instance that takes it -- the
    trap the constructor's own default already had."""
    with pytest.raises(TypeError):
        main.BLUR_KERNEL[0] = 3


# --- the shared contract -----------------------------------------------------

def test_require_returns_the_value_when_present():
    assert main.require({"topics": []}, "topics") == []


def test_require_exits_naming_the_missing_key():
    with pytest.raises(SystemExit) as excinfo:
        main.require({}, "mqtt")

    assert "mqtt" in str(excinfo.value)


def test_a_section_present_but_empty_is_refused():
    """`service:` written and left blank is a section somebody meant to fill
    in. Letting it through moves the failure to whatever first subscripts it,
    a stack frame away from the config that caused it."""
    with pytest.raises(SystemExit, match="service"):
        main.require({"service": None}, "service")


def test_the_shared_helpers_match_the_template_exactly():
    """require and start_subscribers are shared verbatim by every Bytronic
    service. A local edit here is how the copies drift, and the drift shows up
    as a different startup failure in one service than in the rest.

    Only these two. next_trigger and output_topics were dropped: this service
    reads three named topics in a fixed order, and a poll across every queue
    would take an instruction as though it were an image."""
    import ast
    import pathlib

    def source_of(path, name):
        text = pathlib.Path(path).read_text()
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(text, node)
        raise AssertionError(f"{name} not found in {path}")

    here = pathlib.Path(__file__).resolve().parent.parent
    template = here.parent / "service-template" / "app" / "main.py"
    if not template.is_file():
        pytest.skip("service-template is not checked out beside this repository")

    for name in ("require", "start_subscribers"):
        assert source_of(here / "app" / "main.py", name) == source_of(template, name), (
            f"{name} has drifted from service-template")


# --- topic_named -------------------------------------------------------------

def test_topic_named_returns_the_entry_so_callers_reach_its_queue():
    topics = subscribed(make_topics())
    entry = main.topic_named(topics, "receive_depth_image")

    assert entry["topic"] == "churchill/camera/depth/image"
    assert "queue" in entry


def test_topic_named_exits_when_the_name_is_absent():
    """The previous version returned None and the caller published nothing, to
    nowhere, reporting nothing."""
    with pytest.raises(SystemExit, match="receive_depth_image"):
        main.topic_named([], "receive_depth_image")


def test_topic_named_exits_when_the_entry_carries_no_topic():
    with pytest.raises(SystemExit, match="no `topic:` value"):
        main.topic_named([{"name": "receive_depth_image"}], "receive_depth_image")


# --- read --------------------------------------------------------------------

def test_read_returns_none_when_nothing_is_waiting():
    topic = {"topic": "t", "queue": Queue()}
    assert main.read(topic) is None


def test_read_decodes_the_waiting_payload():
    topic = {"topic": "t", "queue": Queue()}
    topic["queue"].put('{"image": "abc"}')

    assert main.read(topic) == {"image": "abc"}


def test_read_discards_a_malformed_payload_without_raising(caplog):
    topic = {"topic": "t", "queue": Queue()}
    topic["queue"].put("not json at all")

    with caplog.at_level("WARNING"):
        assert main.read(topic) is None
    assert "Discarding malformed payload" in caplog.text


def test_read_with_a_timeout_returns_none_instead_of_raising():
    """queue.get(timeout=...) raises Empty. Uncaught, it unwound past the
    KeyboardInterrupt handler and stopped the service -- so a quiet HMI took
    the whole thing down after ten seconds."""
    topic = {"topic": "t", "queue": Queue()}

    assert main.read(topic, timeout=0.01) is None


def test_read_with_a_timeout_returns_a_payload_that_arrives():
    topic = {"topic": "t", "queue": Queue()}
    topic["queue"].put('{"database_instruction": "search_database"}')

    assert main.read(topic, timeout=0.5) == {"database_instruction": "search_database"}


def test_read_takes_from_the_topic_it_is_given_and_no_other():
    """The reason next_trigger is gone: an instruction waiting on its own
    topic must not be consumed by the read that wants an image."""
    topics = subscribed(make_topics())
    instruction = main.topic_named(topics, "receive_hmi_instruction")
    instruction["queue"].put('{"database_instruction": "search_database"}')

    assert main.read(main.topic_named(topics, "receive_depth_image")) is None
    assert instruction["queue"].qsize() == 1


# --- depth_analysis ----------------------------------------------------------

def test_depth_analysis_measures_a_real_image():
    details = main.depth_analysis({"image": depth_image_payload()}, SETTINGS)

    assert set(details) == {"depth", "perimeter", "radius"}
    assert details["radius"] > 0
    assert details["perimeter"] > 0


def test_depth_analysis_returns_json_serialisable_numbers():
    """The measurement is published as JSON. A numpy scalar raises
    "Object of type int64 is not JSON serializable" on every frame that was
    measured successfully -- the path nobody exercises by hand."""
    details = main.depth_analysis({"image": depth_image_payload()}, SETTINGS)

    json.dumps(details)


def test_depth_analysis_ignores_a_message_carrying_no_image():
    """The end-to-end check publishes {"command":"run"} at this service. A
    message it cannot read must cost that message, not the process."""
    assert main.depth_analysis({"command": "run"}, SETTINGS) is None


def test_depth_analysis_survives_an_undecodable_image(caplog):
    with caplog.at_level("WARNING"):
        assert main.depth_analysis({"image": "not-an-image"}, SETTINGS) is None
    assert "Unreadable image payload" in caplog.text


# --- send_details ------------------------------------------------------------

def test_send_details_publishes_to_the_topic_named_in_the_config():
    topics = subscribed(make_topics())
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))
    main.topic_named(topics, "receive_analysis_results")["queue"].put(
        '{"radius": 14.0}')

    main.send_details(client, topics, SETTINGS, {"radius": 14.0},
                      "search_database")

    topic, payload = client.published[0]
    assert topic == "churchill/database/search/sku_data"
    sent = json.loads(payload)
    assert sent["command"] == "search_database"
    assert sent["destination"] == "sku_table"
    assert sent["database_name"] == "churchill_database"


def test_send_details_adds_when_the_instruction_is_not_a_search():
    topics = subscribed(make_topics())
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))
    main.topic_named(topics, "receive_analysis_results")["queue"].put(
        '{"radius": 1.0}')

    main.send_details(client, topics, SETTINGS, {"radius": 1.0}, "add_to_database")

    assert json.loads(client.published[0][1])["command"] == "add_to_database"


def test_send_details_does_not_tag_the_callers_measurement():
    topics = subscribed(make_topics())
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))
    main.topic_named(topics, "receive_analysis_results")["queue"].put(
        '{"radius": 1.0}')
    details = {"radius": 1.0}

    main.send_details(client, topics, SETTINGS, details, "search_database")

    assert details == {"radius": 1.0}


def test_send_details_reports_a_match():
    topics = subscribed(make_topics())
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))
    main.topic_named(topics, "receive_analysis_results")["queue"].put(
        '{"radius": 14.0, "sku": "abc"}')

    assert main.send_details(client, topics, SETTINGS, {"radius": 14.0},
                             "search_database") is True


def test_send_details_reports_no_match():
    topics = subscribed(make_topics())
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))
    main.topic_named(topics, "receive_analysis_results")["queue"].put(
        '{"result": "nothing found"}')

    assert main.send_details(client, topics, SETTINGS, {"radius": 14.0},
                             "search_database") is False


def test_send_details_gives_up_when_the_database_never_answers(caplog):
    """It waited in a `while` loop with no timeout at all, so an unanswered
    search stopped this service processing anything ever again."""
    topics = subscribed(make_topics())
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))

    with caplog.at_level("WARNING"):
        answered = main.send_details(client, topics, SETTINGS, {"radius": 1.0},
                                     "search_database")

    assert answered is False
    assert "No answer on" in caplog.text


def test_send_details_survives_a_failed_publish(caplog):
    """One failed publish loses this measurement. Letting it out of the loop
    would lose every measurement after it too."""
    from mqtt_client.exceptions import MQTTPublishError

    class RefusingClient(FakeMQTTClient):
        def publish(self, topic, message):
            raise MQTTPublishError("broker gone")

    topics = subscribed(make_topics())
    client = RefusingClient(FakeMQTTConfig("127.0.0.1", 1883))

    with caplog.at_level("WARNING"):
        assert main.send_details(client, topics, SETTINGS, {"radius": 1.0},
                                 "search_database") is False
    assert "Could not publish" in caplog.text


# --- main --------------------------------------------------------------------

def base_config(topics=None):
    return {
        "mqtt": {"mqtt_ip": "127.0.0.1", "mqtt_port": 1883,
                 "topics": topics if topics is not None else make_topics()},
        "service": dict(SETTINGS),
    }


def run_main(monkeypatch, config, feed=None, ticks_before_stop=2):
    """Run main() against fakes, stopping after a couple of loop passes."""
    monkeypatch.setattr(main.loadConfig, "get_config", lambda supplied=None: config)
    # configure() replaces the root handlers, pytest's capture handler among
    # them, so caplog goes empty for the rest of the test.
    monkeypatch.setattr(main.logging_setup, "configure", lambda level=None: None)
    monkeypatch.setattr(main, "MQTTClient", FakeMQTTClient)
    monkeypatch.setattr(main, "MQTTConfig", FakeMQTTConfig)

    threads = []
    captured = {}

    def fake_start_subscribe_thread(ip, port, topic, queue, stop_event):
        captured["stop_event"] = stop_event
        if feed and topic in feed:
            queue.put(feed[topic])
        thread = FakeThread()
        threads.append(thread)
        return thread

    monkeypatch.setattr(main, "start_subscribe_thread", fake_start_subscribe_thread)

    ticks = {"count": 0}

    def fake_sleep(_duration):
        ticks["count"] += 1
        if ticks["count"] == ticks_before_stop:
            raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", fake_sleep)

    main.main(["--config", "stub.yaml"])
    return captured, threads


def test_main_runs_one_measurement_all_the_way_through(monkeypatch):
    handled = []
    monkeypatch.setattr(
        main, "send_details",
        lambda client, topics, settings, details, instruction:
            handled.append((details, instruction)) or True)

    captured, threads = run_main(
        monkeypatch, base_config(),
        feed={
            "churchill/camera/depth/image":
                json.dumps({"image": depth_image_payload()}),
            "churchill/hmi/instruction":
                '{"database_instruction": "search_database"}',
        })

    assert len(handled) == 1
    details, instruction = handled[0]
    assert instruction == "search_database"
    assert set(details) == {"depth", "perimeter", "radius"}
    assert captured["stop_event"].is_set()
    assert all(thread.join_called for thread in threads)


def test_main_carries_on_when_no_instruction_arrives(monkeypatch, caplog):
    """A quiet HMI used to stop the service outright."""
    monkeypatch.setattr(main, "send_details",
                        lambda *a, **k: pytest.fail("should not have been reached"))

    with caplog.at_level("WARNING"):
        run_main(monkeypatch, base_config(),
                 feed={"churchill/camera/depth/image":
                       json.dumps({"image": depth_image_payload()})},
                 ticks_before_stop=3)

    assert "No instruction within" in caplog.text


def test_main_discards_an_instruction_that_names_nothing(monkeypatch, caplog):
    monkeypatch.setattr(main, "send_details",
                        lambda *a, **k: pytest.fail("should not have been reached"))

    with caplog.at_level("WARNING"):
        run_main(monkeypatch, base_config(),
                 feed={"churchill/camera/depth/image":
                       json.dumps({"image": depth_image_payload()}),
                       "churchill/hmi/instruction": '{"something_else": "x"}'},
                 ticks_before_stop=3)

    assert "carried no" in caplog.text


def test_main_exits_when_a_topic_it_reads_is_missing(monkeypatch):
    topics = [t for t in make_topics() if t["name"] != "receive_analysis_results"]
    monkeypatch.setattr(main.loadConfig, "get_config",
                        lambda supplied=None: base_config(topics))

    with pytest.raises(SystemExit, match="receive_analysis_results"):
        main.main(["--config", "stub.yaml"])


def test_main_exits_when_a_topic_it_reads_is_not_subscribed(monkeypatch):
    """Its queue would never exist, and the read would fail on the first
    message rather than at startup."""
    topics = make_topics()
    for topic in topics:
        if topic["name"] == "receive_hmi_instruction":
            topic["is_subscribe"] = False

    monkeypatch.setattr(main, "MQTTClient",
                        lambda *a, **k: pytest.fail("connected before refusing"))
    monkeypatch.setattr(main.loadConfig, "get_config",
                        lambda supplied=None: base_config(topics))

    with pytest.raises(SystemExit, match="is_subscribe"):
        main.main(["--config", "stub.yaml"])


def test_main_exits_when_a_required_key_is_missing(monkeypatch):
    monkeypatch.setattr(main.loadConfig, "get_config",
                        lambda supplied=None: {"topics": []})

    with pytest.raises(SystemExit):
        main.main(["--config", "stub.yaml"])


def test_main_configures_logging_before_it_can_fail(monkeypatch):
    """Until configure() runs, the root logger sits at WARNING and every
    info() call is dropped. A service that failed while starting would then
    report nothing about why -- which is precisely when the log matters."""
    import logging as _logging

    from dependencies import logging_setup

    order = []
    monkeypatch.setattr(logging_setup, "configure",
                        lambda level=None: order.append("configured"))
    monkeypatch.setattr(main.loadConfig, "get_config", lambda supplied=None: {})

    with pytest.raises(SystemExit):
        main.main(["--config", "stub.yaml"])
    assert order == ["configured"], "logging was not configured before the first failure"
    _logging.getLogger().handlers[:] = _logging.getLogger().handlers


def test_the_configured_level_comes_from_the_service_config(monkeypatch):
    seen = []
    from dependencies import logging_setup
    monkeypatch.setattr(logging_setup, "configure", lambda level=None: seen.append(level))
    monkeypatch.setattr(main.loadConfig, "get_config",
                        lambda supplied=None: {"logging": {"level": "DEBUG"}})
    with pytest.raises(SystemExit):
        main.main(["--config", "stub.yaml"])
    assert seen == ["DEBUG"]


def test_a_config_without_a_logging_section_still_starts(monkeypatch):
    """Every existing service config predates this setting. A missing section
    must mean the default, not a crash on startup."""
    seen = []
    from dependencies import logging_setup
    monkeypatch.setattr(logging_setup, "configure", lambda level=None: seen.append(level))
    monkeypatch.setattr(main.loadConfig, "get_config", lambda supplied=None: {})
    with pytest.raises(SystemExit):
        main.main(["--config", "stub.yaml"])
    assert seen == [logging_setup.DEFAULT_LEVEL]


def test_a_config_without_timeouts_still_starts(monkeypatch):
    """instruction_timeout and database_timeout arrived with this conversion.
    A config written before them must not stop the service."""
    config = base_config()
    del config["service"]["instruction_timeout"]
    del config["service"]["database_timeout"]

    run_main(monkeypatch, config)


def test_main_answers_help_before_looking_for_a_config(monkeypatch):
    """--help must exit 0 on a binary that has no config beside it, which is
    every binary the release pipeline builds. Reaching get_config() first
    exits 1 for want of a file that only exists once deployed, and no release
    could ever be published."""
    monkeypatch.setattr(
        main.loadConfig, "get_config",
        lambda supplied=None: pytest.fail("looked for a config before --help"))
    with pytest.raises(SystemExit) as exit_info:
        main.main(["--help"])
    assert exit_info.value.code == 0


def test_main_refuses_an_empty_config_path(monkeypatch):
    """`--config ""` reaches main from an unset shell variable or a launcher
    that dropped an argument. With no fallback there is nothing to quietly
    start instead, and the refusal says which flag is at fault."""
    monkeypatch.setattr(main.loadConfig, "_ACTIVE", None)
    with pytest.raises(SystemExit, match="--config is required"):
        main.main(["--config", ""])


def test_main_runs_the_config_it_is_given(monkeypatch, tmp_path):
    """One binary, several instances: the file named on the command line is
    the one that runs, and nothing else is consulted."""
    other = tmp_path / "instance-2.yaml"
    other.write_text("mqtt:\n  mqtt_ip: 10.0.0.2\n")
    monkeypatch.setattr(main.loadConfig, "_ACTIVE", None)

    seen = {}
    monkeypatch.setattr(main, "require",
                        lambda config, key: seen.setdefault(key, config.get(key)) or {})
    # SystemExit derives from BaseException, so `Exception` alone misses it.
    with pytest.raises((Exception, SystemExit)):
        main.main(["--config", str(other)])
    assert seen["mqtt"]["mqtt_ip"] == "10.0.0.2"


# --- what ships --------------------------------------------------------------

def shipped_config():
    from pathlib import Path

    import yaml

    return yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.example.yaml").read_text())


def test_the_config_shipped_in_the_repo_satisfies_what_main_requires():
    """config.example.yaml is what a customer receives beside the binary. If
    it lacks a key main requires, every fresh install fails on first start."""
    config = shipped_config()
    mqtt = main.require(config, "mqtt")
    topics = main.require(mqtt, "topics")
    settings = main.require(config, "service")

    for key in ("mqtt_ip", "mqtt_port"):
        assert key in mqtt, f"{key} missing from the shipped config"
    for key in ("region_of_interest", "trim_value",
                "database_name", "database_table"):
        assert key in settings, f"service.{key} missing from the shipped config"

    for name in main.INPUT_TOPICS + ("send_depth_analysis",):
        main.topic_named(topics, name)


def test_every_topic_the_service_reads_is_subscribed_in_the_shipped_config():
    topics = shipped_config()["mqtt"]["topics"]
    for name in main.INPUT_TOPICS:
        assert main.topic_named(topics, name)["is_subscribe"] is True, (
            f"{name} is read by this service but not subscribed")


def test_the_cloud_topics_are_declared_but_not_subscribed():
    """They are kept as documentation of an exchange nobody has implemented:
    pointcloud_functions.py was deleted in d8340c8 and no code on any branch
    has referenced them since. Subscribing gives them an unbounded queue that
    nothing drains, which grows for the life of the process the moment
    anything starts publishing."""
    topics = shipped_config()["mqtt"]["topics"]
    for name in ("request_cloud_list", "receive_cloud_list"):
        assert main.topic_named(topics, name)["is_subscribe"] is False


def test_the_release_ships_the_config_that_is_tracked():
    """The workflow copies config.example.yaml into the bundle as config.yaml.
    Tracking config.yaml instead means service-orchestrator overwrites a
    tracked file on every sync, and it then shows as modified in a tree
    somebody is about to commit."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    workflow = (root / ".github/workflows/release-pipeline.yml").read_text(encoding="utf-8")

    assert 'cp config.example.yaml "$output_dir/config.yaml"' in workflow
    ignored = (root / ".gitignore").read_text().split()
    assert "/config.yaml" in ignored, "the orchestrator's config.yaml is not ignored"


def test_the_repository_does_not_track_a_config_yaml():
    """A tracked config.yaml in a repository the orchestrator writes into is
    how a customer's brokers and topics reach a remote."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tracked = subprocess.run(["git", "-C", str(root), "ls-files"],
                             capture_output=True, text=True).stdout.split()
    assert "config.yaml" not in tracked
