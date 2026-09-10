# Depth Analysis Service

Measures the subject in a depth image arriving over MQTT, then asks
database-service to match that measurement against known parts or to store it
as a new one.

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

One measurement runs in this order, and each step reads the topic it is
waiting on by name — never whichever queue happens to have something in it.

| Step | Topic | Where |
|---|---|---|
| a depth image arrives | `receive_depth_image` | `read` |
| trim the background and crop | — | `image_functions.Image` |
| measure depth, perimeter, radius | — | `feature_functions.get_subject_details` |
| the operator says what to do with it | `receive_hmi_instruction` | `read` |
| ask the database | `send_depth_analysis` | `send_details` |
| the database answers | `receive_analysis_results` | `send_details` |

The instruction payload carries `database_instruction`: `search_database`
searches, anything else adds.

Nothing here takes the service down. A frame it cannot decode, a frame with
nothing in it, an operator who never answers, a database that never replies, a
publish the broker refuses — each costs that one measurement and the loop
carries on. Losing a frame is losing a frame; stopping loses every frame
after it too.

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
| `service.region_of_interest` | left, top, right, bottom; the frame is cropped to this before anything is measured |
| `service.trim_value` | how far either side of the median counts as background and is zeroed |
| `service.database_name`, `service.database_table` | what the measurement asks to be searched against |
| `service.instruction_timeout` | seconds to wait for the operator |
| `service.database_timeout` | seconds to wait for the database |
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

## Declared but not implemented

`request_cloud_list` and `receive_cloud_list` are in `config.example.yaml` with
`is_subscribe: false`. They were the request/response pair for
`pointcloud_functions.py`, which reconstructed a point cloud from the payload,
downsampled it and generated a mesh. That module was deleted in `d8340c8` and
no code on any branch has referenced the topics since.

They are kept so the intended exchange is not lost: this service was meant to
be able to ask for the list of clouds to compare against. To pick it up, set
`is_subscribe: true` on `receive_cloud_list` and read it by name the way the
other four are read.

`is_subscribe` stays false until then on purpose. A subscribed topic gets an
unbounded queue that only the code reading it drains, so subscribing to
something nothing reads grows without limit for the life of the process — the
moment anything starts publishing, and silently.

## Known: the database topics do not line up

This service publishes to `churchill/database/search/sku_data` and listens on
`churchill/database/search/matching_sku`. database-service subscribes to
`churchill/db/search/sku_data` and publishes to `churchil/db/search/matching_sku`
— `database` against `db`, and `churchil` with one L on its side.

So the round trip currently goes nowhere in both directions. Whichever spelling
wins, both repositories have to agree; this one is a config change on each side,
not a code change.
