"""Per-step cost metrics: step time, throughput, memory. Loss and grad norm
come from Trainer for free; what a step *costs* does not."""

import time

import torch
from transformers import TrainerCallback

BYTES_PER_GB = 1024**3
TOKENS_KEY = "num_input_tokens_seen"


class ThroughputCallback(TrainerCallback):
    """Log step time, token throughput and GPU memory alongside HF's metrics.

    Runs after `WandbCallback`, which has already logged by the time `on_log`
    fires, so editing `logs` would do nothing -- hence a separate `wandb.log`.
    Re-logging the token count stitches the two payloads onto the same x-axis.
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

        # the end-of-training summary: no steps ran, so measuring it would report
        # a ~0s step and 0 tokens/s, which wandb's summary would then keep
        if logs and "train_loss" in logs:
            return

        import wandb

        if wandb.run is None:
            return

        now = time.monotonic()
        metrics = {}

        # skipped on the first call, where there is no interval to divide by
        if self._last_time is not None:
            elapsed = now - self._last_time
            tokens = logs.get(TOKENS_KEY) if logs else None
            if elapsed > 0:
                metrics["train/sec_per_step"] = elapsed / max(args.logging_steps, 1)
                if tokens is not None and self._last_tokens is not None:
                    # the rate now; train_tokens_per_second is the run average
                    metrics["train/tokens_per_sec_interval"] = (
                        tokens - self._last_tokens
                    ) / elapsed
            if tokens is not None:
                # the x-axis these points are plotted against -- see init_wandb
                metrics[f"train/{TOKENS_KEY}"] = tokens
                self._last_tokens = tokens

        self._last_time = now

        if torch.cuda.is_available():  # absent on the local MPS/CPU path
            # high water says how close the job is to OOM, the interval peak
            # says which phase spikes
            peak = torch.cuda.max_memory_allocated() / BYTES_PER_GB
            torch.cuda.reset_peak_memory_stats()
            self._high_water = max(self._high_water, peak)
            metrics["train/gpu_mem_peak_gb"] = peak
            metrics["train/gpu_mem_high_water_gb"] = self._high_water

        if metrics:
            wandb.log(metrics)
