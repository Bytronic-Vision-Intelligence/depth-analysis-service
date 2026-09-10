# app

`main.py` is the entrypoint. It runs the configuration it is given, connects to
the broker, and for every message arriving on a trigger topic measures the
subject in the depth image and publishes what it found.

```bash
python app/main.py --config /path/to/config.yaml
```

`--config` is required and there is no fallback. A service that found a config
beside itself would start whenever one happened to be there -- a stale copy, or
another instance's file -- and look healthy while being the wrong service.

## The loop

| Step | Where |
|---|---|
| poll every trigger queue | `next_trigger` |
| measure the subject | `depth_analysis` → `feature_functions` |
| publish the result | `client.publish(topic, ...)` |

## Configuration

| Key | Meaning |
|---|---|
| `mqtt.mqtt_ip`, `mqtt.mqtt_port` | the broker |
| `mqtt.topics` | looked up **by name**; `is_subscribe` decides in or out |
| `service.database_name`, `service.database_table` | what the published result asks to be searched against |
| `logging.level` | how much this service says |

`config.example.yaml` in the repository root is the documented shape. The real
file is written by service-orchestrator and passed with `--config`.

`dependencies/loadConfig.py`, `dependencies/logging_setup.py` and
`dependencies/mqtt_functions.py` are copied verbatim from `service-template` and
are identical in every Bytronic service, so nothing service-specific belongs in
them.

## What this service does not do yet

`feature_functions.get_subject_details` reports `depth` only. Radius and
perimeter, the image decoding, and the database round-trip live on
`churchill-dev` and are not part of this branch.
