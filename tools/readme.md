# Tools

Reserved for operational scripts specific to this service. Run them from the
repository root so imports and relative paths resolve consistently.

Intentionally empty. Release packaging lives in `scripts/` (`package.sh`,
`sign.py`) and local CI in `docker-local/`.

The service itself takes no arguments beyond `--help` and a required
`--config PATH`:

```bash
python app/main.py --config config.example.yaml
```
