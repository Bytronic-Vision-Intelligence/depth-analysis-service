# app

`main.py` is the entrypoint. It runs the configuration it is given, connects to
the broker, and measures the subject in every depth image that arrives.

```bash
python app/main.py --config /path/to/config.yaml
```

`--config` is required and there is no fallback. A service that found a config
beside itself would start whenever one happened to be there -- a stale copy, or
another instance's file -- and look healthy while being the wrong service.

## The loop

Three topics are read, by name, in a fixed order. That order is the reason
there is no poll-every-queue helper here: a single `next_trigger` returns
whichever queue fires first, which would consume the operator's instruction as
though it were a depth image.

| Step | Where |
|---|---|
| take the next depth image | `read(inputs["receive_depth_image"])` |
| trim, crop, measure | `depth_analysis` |
| wait for the operator | `read(..., timeout=instruction_timeout)` |
| ask the database and wait for its answer | `send_details` |

## The functions

| | |
|---|---|
| `require`, `start_subscribers` | byte-identical to `service-template`; a test pins them |
| `topic_named` | finds a topic by its `name:`, and exits when there isn't one |
| `read` | the next payload on **one** topic, blocking only if given a timeout |
| `depth_analysis` | decode, prepare, measure |
| `send_details` | publish the measurement, wait for the answer |

## Configuration

| Key | Meaning |
|---|---|
| `mqtt.mqtt_ip`, `mqtt.mqtt_port` | the broker |
| `mqtt.topics` | looked up **by name**; every topic this service reads is resolved at startup |
| `service.region_of_interest`, `service.trim_value` | how the frame is prepared |
| `service.database_name`, `service.database_table` | what the measurement is matched against |
| `service.instruction_timeout`, `service.database_timeout` | how long to wait for each answer |
| `logging.level` | how much this service says |

`config.example.yaml` in the repository root is the documented shape. The real
file is written by service-orchestrator and passed with `--config`.

`dependencies/loadConfig.py`, `dependencies/logging_setup.py` and
`dependencies/mqtt_functions.py` are copied verbatim from `service-template` and
are identical in every Bytronic service, so nothing service-specific belongs in
them. That is why `extract_image` lives in `image_functions.py` and not beside
the subscriber helpers.

## Failure

Every step that can fail loses one measurement and no more: an undecodable
payload, a frame with nothing in it, an operator who never answers, a database
that never replies, a publish the broker refuses. The alternative is losing
every measurement after it as well.

Anything the service refuses to start with -- a missing config key, a topic
name it reads that the config does not declare, a topic it reads that is not
subscribed -- is reported at startup, before it connects.
