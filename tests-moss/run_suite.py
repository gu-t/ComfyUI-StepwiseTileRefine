"""Runs the MOSS comparison suite through the test ComfyUI and records results to testdata/runs.json."""
import json
import os
import sys
import traceback
from pathlib import Path

import comfy_api
from prompts import PROMPTS

import workflows

ROOT = Path(os.environ.get("STEPWISE_TEST_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "testdata" / "runs.json"

CASES = [
    # name, kind, input, prompt key, factor, steps, denoises
    ("lighthouse_x3_1step", "stepwise", "lighthouse_1280x960.png", "lighthouse", 3.0, 1, (0.35, 0.35)),
    ("lighthouse_x3_3step", "stepwise", "lighthouse_1280x960.png", "lighthouse", 3.0, 3, (0.45, 0.22)),
    ("staircase_x3_3step", "stepwise", "staircase_1280x976.png", "staircase", 3.0, 3, (0.45, 0.22)),
    ("fisherman_x2_auto", "stepwise", "fisherman_2000x2000.png", "fisherman", 2.0, 0, (0.45, 0.22)),
    ("valley_x2_auto", "stepwise", "valley_1344x768.png", "valley", 2.0, 0, (0.45, 0.22)),
    ("lighthouse_x3_3step_model", "stepwise_model", "lighthouse_1280x960.png", "lighthouse", 3.0, 3, (0.45, 0.22)),
    ("usdu_lighthouse_x3_1pass", "usdu", "lighthouse_1280x960.png", "lighthouse", 3.0, 1, (0.35,)),
    ("usdu_lighthouse_x3_3pass", "usdu", "lighthouse_1280x960.png", "lighthouse", 3.0, 3, (0.45, 0.335, 0.22)),
    ("usdu_staircase_x3_3pass", "usdu", "staircase_1280x976.png", "staircase", 3.0, 3, (0.45, 0.335, 0.22)),
    ("usdu_fisherman_x2_2pass", "usdu", "fisherman_2000x2000.png", "fisherman", 2.0, 2, (0.45, 0.22)),
    ("usdu_valley_x2_2pass", "usdu", "valley_1344x768.png", "valley", 2.0, 2, (0.45, 0.22)),
]


def main(only=None):
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, kind, image, key, factor, steps, den in CASES:
        if only and name not in only:
            continue
        prompt = PROMPTS[key]
        if kind.startswith("stepwise"):
            wf = workflows.stepwise(image, prompt, factor, steps, seed=1, denoise_first=den[0], denoise_last=den[1],
                                    upscale_model="4x-UltraSharp.pth" if kind == "stepwise_model" else None,
                                    prefix=f"suite/{name}")
        else:
            wf = workflows.usdu(image, prompt, factor, seed=1, passes=steps, denoises=list(den), prefix=f"suite/{name}")
        print(f"== {name}", flush=True)
        try:
            r = comfy_api.run(wf)
            results[name] = {"kind": kind, "input": image, "factor": factor, "steps": steps, "denoise": den,
                             "elapsed": r["elapsed"], "files": [str(f) for f in r["files"]],
                             "plan": next(iter(r["texts"].values()), [None])[0] if r["texts"] else None}
            print(f"   {r['elapsed']:.1f}s", flush=True)
        except Exception:
            results[name] = {"kind": kind, "error": traceback.format_exc()}
            print(results[name]["error"], flush=True)
        OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
