"""One progress bar over a whole run of the fork's nodes, with a status line under it.

A run is a list of parts (the upscale model, then each refine pass, in the order they run), each
with a weight fixed before anything samples: for a refine pass the plan's model work, tiles (or
windows) x sampler steps x pictures. A part hands its engine a `Part`, which quacks like
comfy.utils.ProgressBar (update_absolute, update), so an engine reports in its own units and the
part maps that share onto its slice of the one bar. The value never goes down and the total never
changes, so the bar fills once from start to end.

Module scope is comfy-free: comfy and server are imported where they are used."""
import math
import time

# Smallest bar the frontend gets: a run lighter than this is drawn in finer units, so a part with
# few units (a single pass of few steps) still moves every tile step.
MIN_UNITS = 1000
# One comfy tiled_scale step of the upscale model, in tile sampler steps (upstream's weight).
UPSCALE_TILE_WEIGHT = 0.1


def send_status(node_id, text):
    # The status line under the node, as core's PromptServer sends it. Headless (no server
    # instance) or without a node id there is no one to show it to.
    if node_id is None:
        return
    try:
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is not None:
        server.send_progress_text(text, node_id)


class RunProgress:
    """The bar of one node run. `weights` are the parts' shares, in run order (0 for a part that
    does no work); `node_id` is the node's UNIQUE_ID, for the status line. Use it as a context
    manager: leaving it fills the bar and says how the run ended (done, stopped by the user, or failed)."""

    def __init__(self, weights, node_id=None):
        weights = [max(0, math.ceil(w)) for w in weights]
        units = sum(weights)
        self.scale = max(1, math.ceil(MIN_UNITS / units)) if units else 1
        self.total = max(1, units * self.scale)
        self.ends = []
        end = 0
        for w in weights:
            end += w * self.scale
            self.ends.append(end)
        self.node_id = node_id
        self.value = 0
        self.status = None
        self._bar = None
        self._started = None

    def part(self, index, label=None):
        """The ProgressBar stand-in for part `index`. `label` leads its status line."""
        start = self.ends[index - 1] if index else 0
        return Part(self, start, self.ends[index] - start, label)

    def advance(self, value, preview=None):
        # Clamped to the bar and never below where it already is.
        value = min(max(int(value), self.value), self.total)
        if value == self.value and preview is None and self._bar is not None:
            return
        self.value = value
        self._ensure_bar().update_absolute(value, self.total, preview)

    def say(self, text):
        if text and text != self.status:
            self.status = text
            send_status(self.node_id, text)

    def finish(self):
        self.advance(self.total)

    def _ensure_bar(self):
        if self._bar is None:
            import comfy.utils

            self._bar = comfy.utils.ProgressBar(self.total)
        return self._bar

    def __enter__(self):
        self._started = time.perf_counter()
        self._ensure_bar()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.finish()
            self.say(f"done in {time.perf_counter() - self._started:.1f} s")
        else:
            import comfy.model_management

            interrupted = issubclass(exc_type, comfy.model_management.InterruptProcessingException)
            self.say("stopped" if interrupted else "failed")
        return False


class Part:
    """One part of a RunProgress, with core ProgressBar's interface: `update_absolute(value, total,
    preview)` and `update(n)` in the engine's own units. `note(detail)` sets the status line to the
    part's label and the detail. An engine that starts over in a new total (an OOM retry) only
    holds the bar where it was until it passes that point again."""

    def __init__(self, run, start, size, label=None):
        self.run = run
        self.start = start
        self.size = size
        self.label = label
        self.total = 1
        self.current = 0

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        value = min(value, self.total)
        self.current = value
        share = value / self.total if self.total > 0 else 1.0
        self.run.advance(self.start + math.floor(share * self.size), preview)

    def update(self, value):
        self.update_absolute(self.current + value)

    def note(self, detail=None):
        self.run.say(" · ".join(t for t in (self.label, detail) if t))


def picture_prefix(batch_size, batch_index):
    # "image 2/4 · " in a batch, nothing for one picture.
    return f"image {batch_index + 1}/{batch_size} · " if batch_size > 1 else ""
