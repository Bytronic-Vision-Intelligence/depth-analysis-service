import sys
import pathlib
import pytest

# the service runs from app/ and imports its own modules as `dependencies.*`
APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    '''Image writes cropped_depth.png into the working directory, keep it out of the repo'''
    monkeypatch.chdir(tmp_path)
