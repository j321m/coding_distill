"""mrunner entry point. Turns mrunner's flat config into CLI flags for main.py,
so mrunner never sees tyro and tyro never sees mrunner's `--config`.

Keys are dotted (`train.learning_rate`) -- tyro's syntax for nested dataclasses.
"""

import os
import subprocess
import sys

from mrunner.helpers.client_helper import get_configuration

if __name__ == "__main__":
    cfg = get_configuration()
    cfg.pop("experiment_id", None)  # added by mrunner, not ours

    flags = []
    for key, value in cfg.items():
        flags.append(f"--{key}")
        if isinstance(value, list):
            flags.extend(str(item) for item in value)
        else:
            # bools too: with no default, tyro parses them as `--x True`
            flags.append(str(value))

    # one process per GPU (DDP); a single process on several GPUs makes
    # Trainer fall back to DataParallel, which gathers all logits on GPU 0.
    # SLURM sets CUDA_VISIBLE_DEVICES from --gres; unset locally -> no GPUs.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_gpus = len(visible.split(",")) if visible else 0
    if n_gpus > 1:
        launcher = ["-m", "torch.distributed.run", f"--nproc-per-node={n_gpus}"]
    else:
        launcher = []
    cmd = [sys.executable, *launcher, "-m", "main"] + flags
    print("Running command:", cmd, flush=True)
    subprocess.run(cmd, check=True)
