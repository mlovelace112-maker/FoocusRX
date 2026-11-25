# Browser tests

Headless-Chrome tests for the running app (`tests/e2e/test_ui.py`). They are
skipped unless `FOOOCUS_E2E_URL` is set, so `python -m unittest discover tests`
and CI stay fast.

```sh
# 1. Start the app on scratch outputs, with a scratch trash folder
FOOOCUS_TRASH_DIR=$PWD/outputs-e2e-trash venv/bin/python launch.py --port 7866

# 2. Run the tests
FOOOCUS_E2E_URL=http://127.0.0.1:7866/ \
FOOOCUS_E2E_OUTPUTS=$PWD/outputs FOOOCUS_E2E_ALLOW_DELETE=1 \
venv/bin/python -m unittest tests.e2e.test_ui -v
```

| Variable | Effect |
|---|---|
| `FOOOCUS_E2E_URL` | App URL; required |
| `FOOOCUS_E2E_OUTPUTS` + `FOOOCUS_E2E_ALLOW_DELETE=1` | History delete/undo test; scratch outputs only |
| `FOOOCUS_E2E_GENERATE=1` | Real FLUX + SDXL-draft generations (GPU, minutes) |
| `FOOOCUS_E2E_SHOTS=<dir>` | Save screenshots |
| `FOOOCUS_E2E_CHROME` | Chrome binary (default: macOS Google Chrome) |
