"""Tiny client for the isolated test ComfyUI (127.0.0.1:8189): queue an API-format prompt,
wait for it, and return the history entry plus the saved output files."""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8189"
ROOT = Path(os.environ.get("STEPWISE_TEST_ROOT", Path(__file__).resolve().parent.parent))
OUTPUT_DIR = ROOT / "run" / "output"
INPUT_DIR = ROOT / "run" / "input"


def _req(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read() or b"null")


def queue(prompt):
    try:
        return _req("/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})["prompt_id"]
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"/prompt rejected: {err.read().decode(errors='replace')}") from None


def wait(prompt_id, timeout=7200, poll=2.0):
    start = time.time()
    while time.time() - start < timeout:
        hist = _req(f"/history/{prompt_id}")
        if hist and prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") in ("success", "error"):
                return entry, time.time() - start
        time.sleep(poll)
    raise TimeoutError(prompt_id)


def run(prompt, timeout=7200):
    pid = queue(prompt)
    entry, elapsed = wait(pid, timeout)
    status = entry.get("status", {})
    if status.get("status_str") != "success":
        msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        detail = msgs[0][1] if msgs else status
        if isinstance(detail, dict):
            detail = f"{detail.get('node_type')}: {detail.get('exception_message')}\n{''.join(detail.get('traceback', []))}"
        raise RuntimeError(f"prompt {pid} failed: {detail}")
    files = []
    texts = {}
    for node_id, out in entry.get("outputs", {}).items():
        for img in out.get("images", []):
            if img.get("type") == "output":
                files.append(OUTPUT_DIR / img.get("subfolder", "") / img["filename"])
        if "text" in out:
            texts[node_id] = out["text"]
    return {"prompt_id": pid, "elapsed": elapsed, "files": files, "texts": texts, "entry": entry}


def free_memory():
    _req("/free", {"unload_models": True, "free_memory": True})


if __name__ == "__main__":
    prompt = json.load(open(sys.argv[1], encoding="utf-8"))
    result = run(prompt)
    print(f"done in {result['elapsed']:.1f}s")
    for f in result["files"]:
        print(f)
