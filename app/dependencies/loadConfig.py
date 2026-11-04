import yaml
from pathlib import Path


def _config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def get_config(config_path=None) -> dict:
    """Read and return configuration from `config_path`, or `app/configs/config.yaml` when it is None.

    Returns an empty dict if the file is missing or empty.
    """
    path = _config_path() if config_path is None else Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def return_config_value(key: str, config_path=None) -> str:
    """Return the value for `key` from the loaded config.

    Raises ValueError for empty keys and KeyError when the key is missing.
    """
    if not key:
        raise ValueError("Key cannot be empty.")
    config = get_config(config_path)
    if key not in config:
        raise KeyError(f"Key '{key}' not found in configuration.")
    return config[key]