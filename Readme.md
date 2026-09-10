# Depth Analysis Service

Measures the subject in a depth image arriving over MQTT and publishes what it
found, for a database service to match against known parts.

## Requirements

- Python 3.10 — the version CI installs and the release build freezes
- An MQTT broker, normally at `localhost:1883`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements-dev.txt
python -m pytest
python app/main.py --config config.example.yaml
```

`--config` is required and there is no fallback. A service that looked for a
config beside itself would start whenever one happened to be there — a stale
copy from a previous deployment, or another instance's file sharing the
directory — and would subscribe, report itself healthy, and be the wrong
service, with nothing saying so.

## What it does

| Step | Where |
|---|---|
| poll every trigger queue | `next_trigger` |
| measure the subject | `depth_analysis` → `FeatureExtraction.get_subject_details` |
| publish the result | to every topic the config marks `is_subscribe: false` |

A trigger carrying no `depth_image` is ignored rather than treated as an error:
every subscribed trigger feeds the same loop, and the point-cloud list arrives
on one of them.

## Configuration

`config.example.yaml` is the documented shape and the copy that ships beside the
binary. The real file is written by
[service-orchestrator](https://github.com/Bytronic-Vision-Intelligence/service-orchestrator)
and passed with `--config`, so one binary can serve several installations.

| Key | Meaning |
|---|---|
| `project` | topic namespace for this installation |
| `mqtt.mqtt_ip`, `mqtt.mqtt_port` | the broker |
| `mqtt.topics` | looked up **by name**; `is_subscribe` decides in or out, `is_trigger` decides what starts work |
| `service.database_name`, `service.database_table` | what the published result asks to be searched against |
| `logging.level` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |

A required key that is missing, or written and left blank, stops the service at
startup naming the key and the file. That costs one restart; discovering it
later means a service that connects, looks healthy, and publishes into nothing.

This service writes no log files. It prints, the orchestrator forwards every
line to `project/logging/depth-analysis-service`, and logging-service is the
only thing that writes to disk.

## Local CI

```bash
./docker-local/run.sh            # what a pull request would run
./docker-local/run.sh --release  # build the binary and smoke-test it
```

Needs Docker; act and the docker CLI live inside the image.

## Releases

Pushing to `prod` builds a signed, per-platform release. GitHub runs a
branch-triggered workflow **as it exists on the branch that was pushed**, so a
`prod` branch that predates `.github/workflows/release-pipeline.yml` produces no
release and no error. Keep `prod` current with `main`.

## Project layout

- `app/` — the service; `main.py` is the entrypoint
- `app/dependencies/` — `loadConfig`, `logging_setup` and `mqtt_functions` are
  byte-identical to `service-template` and to every other Bytronic service
- `test/` — the suite; `python -m pytest`
- `scripts/`, `docker-local/` — release packaging and the local runner
- `docs/` — documentation and licence

## Not implemented on this branch

`FeatureExtraction.get_subject_details` reports `depth` only. Radius and
perimeter, the point-cloud image decoding and the database round-trip are on
`churchill-dev`.
