"""Numeric checks and 100% comparison crops for the suite in testdata/runs.json.

Metrics per output image (all on luma, 0-255):
  size_ok      final size == round(source * factor)
  drift        mean |output area-downscaled to source size - source| per pixel, and the worst
               32x32-block mean difference (colour drift between regions/steps)
  bias         mean signed per-channel difference (global colour shift)
  seam_x/y     the most outstanding column/row: gradient across that line divided by the local
               median, taken as the median over 256 px segments so only a line that runs the
               whole image (a tile seam) scores high. ~1 = nothing, >1.5 = suspicious
  grid16/8     periodicity of the gradient energy folded modulo 16 / 8 px (latent-cell staircase
               or blocking indicator): max/mean of the folded profile, ~1.0 = none
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("STEPWISE_TEST_ROOT", Path(__file__).resolve().parent.parent))
RUNS = ROOT / "testdata" / "runs.json"
COMPARE = ROOT / "testdata" / "compare"
INPUT = ROOT / "run" / "input"


def luma(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def line_outliers(img_l, axis, seg=256):
    # Gradient across each column (axis=1) or row (axis=0), per segment along the line.
    a = img_l if axis == 1 else img_l.T
    g = np.abs(np.diff(a, axis=1))  # [H, W-1]; g[:, x-1] is the step between x-1 and x
    h = g.shape[0]
    segs = [g[i:i + seg].mean(axis=0) for i in range(0, h - seg // 2, seg)] or [g.mean(axis=0)]
    ratios = []
    for prof in segs:
        pad = 24
        local = np.array([np.median(prof[max(0, x - pad):x + pad + 1]) for x in range(prof.size)])
        ratios.append(prof / np.maximum(local, 1e-3))
    r = np.median(np.stack(ratios), axis=0)
    order = np.argsort(r)[::-1][:5]
    return [(int(x + 1), round(float(r[x]), 2)) for x in order]


def grid_periodicity(img_l, period):
    gx = np.abs(np.diff(img_l, axis=1))
    gy = np.abs(np.diff(img_l, axis=0))
    px = np.array([gx[:, k::period].mean() for k in range(period)])
    py = np.array([gy[k::period, :].mean() for k in range(period)])
    return round(float(max(px.max() / px.mean(), py.max() / py.mean())), 3)


def drift(out, src):
    small = np.asarray(Image.fromarray(out.astype(np.uint8)).resize((src.shape[1], src.shape[0]), Image.Resampling.BOX)).astype(np.float32)
    diff = small - src
    b = 32
    h, w = (src.shape[0] // b) * b, (src.shape[1] // b) * b
    blocks = np.abs(diff[:h, :w].reshape(h // b, b, w // b, b, 3).mean(axis=(1, 3))).max()
    return round(float(np.abs(diff).mean()), 2), round(float(blocks), 2), [round(float(c), 2) for c in diff.reshape(-1, 3).mean(axis=0)]


def analyze(path, src_name, factor):
    out = load(path)
    src = load(INPUT / src_name)
    want = (round(src.shape[1] * factor), round(src.shape[0] * factor))
    l = luma(out)
    d_mean, d_block, bias = drift(out, src)
    return {
        "size": [out.shape[1], out.shape[0]], "size_ok": (out.shape[1], out.shape[0]) == want,
        "drift": d_mean, "drift_block_max": d_block, "bias_rgb": bias,
        "seam_x": line_outliers(l, 1)[:3], "seam_y": line_outliers(l, 0)[:3],
        "grid16": grid_periodicity(l, 16), "grid8": grid_periodicity(l, 8),
    }


def crop_triplet(name, src_name, factor, ours, usdu, box, tag):
    # box in SOURCE coordinates (x, y, size); shows lanczos of the source, ours, USDU at 100%.
    src = Image.open(INPUT / src_name).convert("RGB")
    x, y, s = box
    fx = round(x * factor)
    fy = round(y * factor)
    fs = round(s * factor)
    lan = src.crop((x, y, x + s, y + s)).resize((fs, fs), Image.Resampling.LANCZOS)
    tiles = [lan] + [Image.open(p).convert("RGB").crop((fx, fy, fx + fs, fy + fs)) for p in (ours, usdu) if p]
    sheet = Image.new("RGB", (fs * len(tiles) + 8 * (len(tiles) - 1), fs), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (fs + 8), 0))
    COMPARE.mkdir(parents=True, exist_ok=True)
    out = COMPARE / f"{name}_{tag}.png"
    sheet.save(out)
    return out


def final_file(entry):
    files = [f for f in entry.get("files", []) if "_step_" not in Path(f).name]
    return files[-1] if files else None


def main():
    runs = json.loads(RUNS.read_text())
    report = {}
    for name, e in runs.items():
        if "error" in e:
            report[name] = {"error": e["error"].strip().splitlines()[-1]}
            continue
        report[name] = analyze(final_file(e), e["input"], e["factor"])
        report[name]["elapsed"] = round(e["elapsed"], 1)
    (ROOT / "testdata" / "metrics.json").write_text(json.dumps(report, indent=2))
    for name, m in report.items():
        print(name, json.dumps(m))


if __name__ == "__main__":
    main()


# Crops in SOURCE coordinates (x, y, size): tile seams of the final grid plus diagonal-heavy areas.
CROPS = {
    "lighthouse_x3_3step": ("usdu_lighthouse_x3_3pass", [((200, 200, 160), "seam_wires"), ((560, 300, 160), "wires_birds"), ((40, 600, 200), "fence")]),
    "lighthouse_x3_1step": ("usdu_lighthouse_x3_1pass", [((200, 200, 160), "seam_wires"), ((40, 600, 200), "fence")]),
    "staircase_x3_3step": ("usdu_staircase_x3_3pass", [((520, 340, 200), "center"), ((900, 100, 200), "cables")]),
    "fisherman_x2_auto": ("usdu_fisherman_x2_2pass", [((800, 650, 256), "face"), ((300, 300, 200), "net_seam")]),
    "valley_x2_auto": ("usdu_valley_x2_2pass", [((880, 360, 256), "bridge"), ((300, 300, 200), "forest_seam")]),
}


def make_crops():
    runs = json.loads(RUNS.read_text())
    made = []
    for ours, (theirs, boxes) in CROPS.items():
        if ours not in runs or "error" in runs[ours]:
            continue
        e = runs[ours]
        other = final_file(runs[theirs]) if theirs in runs and "error" not in runs[theirs] else None
        for box, tag in boxes:
            made.append(crop_triplet(ours, e["input"], e["factor"], final_file(e), other, box, tag))
    return made


if __name__ == "__main__" and "--crops" in sys.argv:
    for p in make_crops():
        print(p)
