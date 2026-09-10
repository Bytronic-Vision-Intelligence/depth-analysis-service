# Tests

Pytest, covering config resolution, logging setup, the MQTT subscriber helpers,
the main loop and the release tooling. No broker is required: `fakes.py`
supplies stand-ins for the client, config and threads.

```bash
python -m pytest
```

`pytest.ini` sets `pythonpath = . app scripts`, so tests import exactly the way
the runtime does (`from main import ...`, `from dependencies import ...`) and
can reach the release scripts without installing them.

Test files must be named `test_*.py`. A capitalised `Test_*.py` is not collected
by pytest's default pattern, and because macOS and Windows are case-insensitive
the file looks fine locally while the Linux runner collects nothing. All three
of this repository's original test files were in that state, so the suite
reported success while running none of them. CI fails the build on any tracked
file matching `Test_*.py`.
