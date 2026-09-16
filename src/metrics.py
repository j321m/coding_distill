"""Per-step cost metrics: what HF's Trainer does not measure.

Loss, grad norm and learning rate arrive for free -- `Trainer` logs them every
`logging_steps` and `rewrite_logs` prefixes them `train/`. What it never reports
is what a step *costs*: wall-clock time, tokens pushed through, memory headroom.
Those are the numbers that decide batch size and sharding at 7B/32k, and reading
them after an OOM is too late.
"""

import time

import torch
from transformers import TrainerCallback

BYTES_PER_GB = 1024**3
TOKENS_KEY = "num_input_tokens_seen"


class ThroughputCallback(TrainerCallback):
    """Log step time, token throughput and GPU memory alongside HF's metrics.

    This runs *after* `WandbCallback`: `Trainer.__init__` builds
    `callbacks = default_callbacks + callbacks`, so the integration callbacks fire
    first and wandb has already logged by the time `on_log` gets here. Editing
    `logs` would therefore change nothing -- hence the separate `wandb.log` call,
    which opens its own `_step`. Re-logging the token count is what stitches the
    two payloads together: `init_wandb` makes it the x-axis for `train/*`, so a
    shared token value puts these points on the same place in every chart.
    """

    def __init__(self):
        self._last_time = None
        self._last_tokens = None
        # carried here because `reset_peak_memory_stats` clears torch's own max
        self._high_water = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time = time.monotonic()
        self._last_tokens = state.num_input_tokens_seen

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        # Trainer logs an end-of-training summary right after the last step: no
        # steps ran, no tokens consumed. Measured, it reports a ~0s step and 0
        # tokens/s, and wandb's summary keeps the last value -- so that garbage
        # is what the run ends up reporting. Per-step payloads carry `loss`, only
        # the summary carries `train_loss`.
        if logs and "train_loss" in logs:
            return

        import wandb

        if wandb.run is None:
            return

        now = time.monotonic()
        metrics = {}

        # Skipped on the first call, where there is no interval to divide by.
        if self._last_time is not None:
            elapsed = now - self._last_time
            tokens = logs.get(TOKENS_KEY) if logs else None
            if elapsed > 0:
                metrics["train/sec_per_step"] = elapsed / max(args.logging_steps, 1)
                if tokens is not None and self._last_tokens is not None:
                    # This is the rate right now; theirs (train_tokens_per_second)
                    # is the run average.
                    metrics["train/tokens_per_sec_interval"] = (
                        tokens - self._last_tokens
                    ) / elapsed
            if tokens is not None:
                # the x-axis these points are plotted against -- see init_wandb
                metrics[f"train/{TOKENS_KEY}"] = tokens
                self._last_tokens = tokens

        self._last_time = now

        if torch.cuda.is_available():  # absent on the local MPS/CPU path
            # Two readings, two questions: the high-water mark says how close the
            # job is to OOM, the per-interval peak says which phase spikes.
            peak = torch.cuda.max_memory_allocated() / BYTES_PER_GB
            torch.cuda.reset_peak_memory_stats()
            self._high_water = max(self._high_water, peak)
            metrics["train/gpu_mem_peak_gb"] = peak
            metrics["train/gpu_mem_high_water_gb"] = self._high_water

        if metrics:
            wandb.log(metrics)
