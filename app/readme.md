App usage notes

This folder contains application code and dependencies for the worker template.

Key files:
- `Dependencies/config.yaml` — default configuration values.
- `Dependencies/loadConfig.py` — helper to load and parse the config.

Quick example (from project root):

```powershell
python -c "from app.Dependencies import loadConfig; print(loadConfig.load_config())"
```

Customize `config.yaml` and re-run your worker entrypoint.

