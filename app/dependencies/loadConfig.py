import yaml
from pathlib import Path


_CONFIG_PATH: Path | None = None
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_FALLBACK_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def set_config_path(path: str | None) -> None:
    """Set the YAML config path used by get_config / return_config_value.
    Args:
        path: path to a config file, or None to use the fallback under app/configs/
    """
    global _CONFIG_PATH
    _CONFIG_PATH = _FALLBACK_CONFIG_PATH if path is None else Path(path)


def _config_path() -> Path:
    return _CONFIG_PATH if _CONFIG_PATH is not None else _DEFAULT_CONFIG_PATH


def get_config() -> dict:
    """Read and return configuration from the local `config.yaml` next to this module.

    Returns an empty dict if the file is missing or empty.
    """
    path = _config_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def return_config_value(key: str) -> str:
    """Return the value for `key` from the loaded config.

    Raises ValueError for empty keys and KeyError when the key is missing.
    """
    if not key:
        raise ValueError("Key cannot be empty.")
    config = get_config()
    if key not in config:
        raise KeyError(f"Key '{key}' not found in configuration.")
    return config[key]