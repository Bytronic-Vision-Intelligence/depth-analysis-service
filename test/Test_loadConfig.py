import pytest
from dependencies.loadConfig import return_config_value, get_config


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("key1: value1\nimage_options:\n  pixel_pitch_mm: [0.2, 0.05]\n", encoding="utf-8")
    return path


def test_get_config_reads_yaml(config_file):
    config = get_config(config_file)
    assert config["key1"] == "value1"
    assert config["image_options"]["pixel_pitch_mm"] == [0.2, 0.05]


def test_get_config_missing_or_empty_file_returns_empty_dict(tmp_path):
    assert get_config(tmp_path / "missing.yaml") == {}
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert get_config(empty) == {}


def test_get_config_defaults_to_app_config():
    assert isinstance(get_config(), dict)


def test_return_config_value(config_file):
    assert return_config_value("key1", config_file) == "value1"

    with pytest.raises(KeyError):
        return_config_value("non_existent_key", config_file)

    with pytest.raises(ValueError):
        return_config_value("", config_file)


def main():
    pytest.main()


if __name__ == "__main__":
    pytest.main()
