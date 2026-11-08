# MOSS-Image test harness

Scripts used to run and judge the real-model tests behind the fork's README. They talk to a
ComfyUI started separately (default `http://127.0.0.1:8189`, see `comfy_api.py`) that has this
package and ComfyUI_UltimateSDUpscale installed and the MOSS-Image files available.

`STEPWISE_TEST_ROOT` is the folder that holds `run/input`, `run/output` and `testdata`
(default: the parent of this folder).

- `workflows.py`: API-format builders (MOSS text-to-image, Stepwise Tile Upscale, chained
  Ultimate SD Upscale, VAE round trip, untiled whole-image refine).
- `run_suite.py`: the comparison suite, results in `testdata/runs.json`.
- `analyze.py`: exact-size check, colour drift, seam-line outliers and 16/8 px grid
  periodicity, plus 100% comparison crops (`--crops`).
