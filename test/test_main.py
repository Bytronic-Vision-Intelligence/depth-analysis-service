from queue import Queue

import pytest

from fakes import FakeMQTTClient, FakeMQTTConfig, FakeThread

import main


def make_topics():
    """The shape shipped in config.example.yaml."""
    return [
        {
            "name": "receive_depth_image",
            "topic": "project/camera/depth/image",
            "is_subscribe": True,
            "is_trigger": True,
        },
        {
            "name": "request_cloud_list",
            "topic": "project/camera/depth/request_cloud",
            "is_subscribe": False,
            "is_trigger": False,
        },
        {
            "name": "receive_cloud_list",
            "topic": "project/camera/depth/receive_cloud_list",
            "is_subscribe": True,
            "is_trigger": True,
        },
    ]


SETTINGS = {"database_name": "churchill_database", "database_table": "sku_table"}


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


def test_require_matches_the_template_exactly():
    """Every Bytronic service shares this function verbatim. A local edit here
    is how the copies drift, and the drift shows up as a different startup
    failure in one service than in the rest."""
    import ast
    import pathlib

    def source_of(path, name):
        text = pathlib.Path(path).read_text()
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(text, node)
        raise AssertionError(f"{name} not found in {path}")

    here = pathlib.Path(__file__).resolve().parent.parent
    template = here.parent / "service-template" / "app" / "main.py"
    if not template.is_file():
        pytest.skip("service-template is not checked out beside this repository")

    for name in ("require", "start_subscribers", "next_trigger", "output_topics"):
        assert source_of(here / "app" / "main.py", name) == source_of(template, name), (
            f"{name} has drifted from service-template")


def test_output_topics_returns_only_the_unsubscribed_topics():
    assert main.output_topics(make_topics()) == [
        "project/camera/depth/request_cloud"]


def test_next_trigger_returns_none_when_no_queues_exist():
    assert main.next_trigger(make_topics()) is None


def test_next_trigger_returns_none_when_queues_are_empty():
    topics = make_topics()
    for topic in topics:
        topic["queue"] = Queue()

    assert main.next_trigger(topics) is None


def test_next_trigger_decodes_the_waiting_payload():
    topics = make_topics()
    topics[0]["queue"] = Queue()
    topics[0]["queue"].put('{"depth_image": [[1, 2], [3, 4]]}')

    assert main.next_trigger(topics) == {"depth_image": [[1, 2], [3, 4]]}


def test_next_trigger_ignores_queues_that_are_not_triggers():
    topics = make_topics()
    topics[1]["is_subscribe"] = True
    topics[1]["queue"] = Queue()
    topics[1]["queue"].put('{"depth_image": [[1]]}')

    assert main.next_trigger(topics) is None


def test_next_trigger_discards_malformed_payloads_without_raising(caplog):
    topics = make_topics()
    topics[0]["queue"] = Queue()
    topics[0]["queue"].put("not json at all")

    with caplog.at_level("WARNING"):
        assert main.next_trigger(topics) is None
    # WARNING, not INFO: a payload the service cannot read is something being
    # published wrongly, and at INFO it reads as routine chatter.
    assert "Discarding malformed payload" in caplog.text
    assert "WARNING" in caplog.text


def test_start_subscribers_spawns_only_for_subscribed_topics(monkeypatch):
    started = []

    def fake_start_subscribe_thread(ip, port, topic, queue, stop_event):
        started.append(topic)
        return FakeThread()

    monkeypatch.setattr(main, "start_subscribe_thread", fake_start_subscribe_thread)
    topics = make_topics()

    threads = main.start_subscribers(
        {"mqtt_ip": "127.0.0.1", "mqtt_port": 1883}, topics, main.Event()
    )

    assert started == ["project/camera/depth/image",
                       "project/camera/depth/receive_cloud_list"]
    assert len(threads) == 2
    assert "queue" in topics[0] and "queue" in topics[2]
    assert "queue" not in topics[1]


def test_depth_analysis_measures_the_image_and_publishes_the_result():
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))
    message = {"depth_image": [[1, 2], [3, 9]]}

    main.depth_analysis(client, message, ["project/camera/depth/request_cloud"],
                        SETTINGS)

    assert len(client.published) == 1
    topic, payload = client.published[0]
    assert topic == "project/camera/depth/request_cloud"

    import json
    sent = json.loads(payload)
    assert sent["depth"] == 8            # 9 - 1
    assert sent["destination"] == "sku_table"
    assert sent["database_name"] == "churchill_database"


def test_depth_analysis_measures_a_json_payload_not_only_an_array():
    """The image arrives over MQTT as JSON, so it reaches this function as
    nested lists. get_subject_details measures with .max()/.min(), which a
    list does not carry -- calling it with the payload as received raises
    AttributeError and loses the frame."""
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))

    main.depth_analysis(client, {"depth_image": [[4, 11]]},
                        ["project/camera/depth/request_cloud"], SETTINGS)

    import json
    assert json.loads(client.published[0][1])["depth"] == 7


def test_depth_analysis_ignores_a_message_carrying_no_image():
    """Every subscribed trigger feeds this, not just the depth image topic.
    A cloud list arriving on its own trigger is not something to measure, and
    must not take the service down or publish a result."""
    client = FakeMQTTClient(FakeMQTTConfig("127.0.0.1", 1883))

    main.depth_analysis(client, {"cloud_list": ["a", "b"]},
                        ["project/camera/depth/request_cloud"], SETTINGS)

    assert client.published == []


def test_main_processes_one_message_then_shuts_down_cleanly(monkeypatch):
    config = {
        "mqtt": {"mqtt_ip": "127.0.0.1", "mqtt_port": 1883,
                 "topics": make_topics()},
        "service": SETTINGS,
    }
    monkeypatch.setattr(main.loadConfig, "get_config", lambda supplied=None: config)
    monkeypatch.setattr(main, "MQTTClient", FakeMQTTClient)
    monkeypatch.setattr(main, "MQTTConfig", FakeMQTTConfig)

    threads = []
    captured = {}

    def fake_start_subscribe_thread(ip, port, topic, queue, stop_event):
        captured["stop_event"] = stop_event
        if topic == "project/camera/depth/image":
            queue.put('{"depth_image": [[1, 5]]}')
        thread = FakeThread()
        threads.append(thread)
        return thread

    monkeypatch.setattr(main, "start_subscribe_thread", fake_start_subscribe_thread)

    handled = []
    monkeypatch.setattr(
        main,
        "depth_analysis",
        lambda client, message, outputs, settings: handled.append(
            (message, outputs, settings)),
    )

    ticks = {"count": 0}

    def fake_sleep(_duration):
        ticks["count"] += 1
        if ticks["count"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", fake_sleep)

    main.main(["--config", "stub.yaml"])

    assert handled == [({"depth_image": [[1, 5]]},
                        ["project/camera/depth/request_cloud"], SETTINGS)]
    assert captured["stop_event"].is_set()
    assert all(thread.join_called for thread in threads)


def test_main_exits_when_a_required_key_is_missing(monkeypatch):
    monkeypatch.setattr(main.loadConfig, "get_config",
                        lambda supplied=None: {"topics": []})

    with pytest.raises(SystemExit):
        main.main(["--config", "stub.yaml"])


def test_main_configures_logging_before_it_can_fail(monkeypatch):
    """Until configure() runs, the root logger sits at WARNING and every
    info() call is dropped. A service that failed while starting would then
    report nothing about why -- which is precisely when the log matters.

    So it must run before the first thing that can raise: `require`, which
    exits when a config key is missing."""
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
    with pytest.raises(Exception):
        main.main(["--config", str(other)])
    assert seen["mqtt"]["mqtt_ip"] == "10.0.0.2"


def test_the_config_shipped_in_the_repo_satisfies_what_main_requires():
    """config.example.yaml is what a customer receives beside the binary. If
    it lacks a key main requires, every fresh install fails on first start."""
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.example.yaml").read_text())
    mqtt = main.require(config, "mqtt")
    topics = main.require(mqtt, "topics")
    settings = main.require(config, "service")

    for key in ("mqtt_ip", "mqtt_port"):
        assert key in mqtt, f"{key} missing from the shipped config"
    for key in ("database_name", "database_table"):
        assert key in settings, f"service.{key} missing from the shipped config"
    assert main.output_topics(topics), "the shipped config declares nothing to publish to"


def test_the_shipped_config_names_a_topic_for_every_name_main_expects():
    """Topics are matched by name, so a rename in the config silently stops
    the service subscribing rather than failing."""
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.example.yaml").read_text())
    names = {topic["name"] for topic in config["mqtt"]["topics"]}
    assert "receive_depth_image" in names


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
