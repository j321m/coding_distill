"""mrunner entry point for every job. Turns mrunner's flat config into CLI flags
for the module named by its `entry_point` key, so mrunner never sees tyro and
tyro never sees mrunner's `--config`.

Keys are dotted (`train.learning_rate`) -- tyro's syntax for nested dataclasses.
Each entry point parses them into its schema in src/config.py.
"""

import json
import os
import subprocess
import sys
import tempfile

from mrunner.helpers.client_helper import get_configuration

from src.config import ENTRY_POINTS
from src.provenance import GIT_ENV, GIT_KEYS

if __name__ == "__main__":
    cfg = get_configuration()
    cfg.pop("experiment_id", None)  # added by mrunner, not ours
    # both validated at submit by configs/common.py
    module = cfg.pop("entry_point")
    gpus = cfg.pop("gpus")

    # not flags: no schema has them. And not argv or env either: Linux caps a
    # single string there at 128KB, which a dirty tree's diff can exceed.
    # In the cwd, not /tmp: mrunner cd's into the experiment dir on shared
    # storage, and every node of a multi-node job has to be able to read it.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=os.getcwd(), delete=False
    ) as git_file:
        json.dump({key: cfg.pop(key, None) for key in GIT_KEYS}, git_file)

    flags = []
    for key, value in cfg.items():
        flags.append(f"--{key}")
        if isinstance(value, list):
            flags.extend(str(item) for item in value)
        elif isinstance(value, dict):
            # tyro reads a dict as `--x key value key value`
            for item in value.items():
                flags.extend(str(part) for part in item)
        else:
            # bools too: with no default, tyro parses them as `--x True`
            flags.append(str(value))

    # one process per GPU (DDP); a single process on several GPUs makes
    # Trainer fall back to DataParallel, which gathers all logits on GPU 0.
    # `gpus` counts every node (configs/common.py); torchrun gets the per-node share.
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    prefix = []
    if ENTRY_POINTS[module] and gpus > 1:
        if gpus % nodes:
            raise SystemExit(f"gpus={gpus} does not split over {nodes} nodes")
        # --tee 3: prefix every stdout/stderr line with the rank that wrote it
        launcher = [
            "-m",
            "torch.distributed.run",
            f"--nnodes={nodes}",
            f"--nproc-per-node={gpus // nodes}",
            "--tee",
            "3",
        ]
        if nodes > 1:
            # an sbatch body runs on the first node only; srun runs the launcher
            # once per node (--ntasks-per-node=1 in clusters.yaml), and the c10d
            # rendezvous at MASTER_ADDR:MASTER_PORT (set in prolog_cmd) assigns
            # node ranks, so nothing here depends on per-node variables
            launcher += [
                "--rdzv-backend=c10d",
                f"--rdzv-id={os.environ['SLURM_JOB_ID']}",
                f"--rdzv-endpoint={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            ]
            # --kill-on-bad-exit: one rank dying takes the step down instead of
            # leaving the other nodes waiting at the next collective
            prefix = ["srun", "--ntasks-per-node=1", "--kill-on-bad-exit=1"]
            if "SLURM_CPUS_PER_TASK" in os.environ:
                # srun since Slurm 22.05 no longer inherits it from sbatch
                prefix.append(f"--cpus-per-task={os.environ['SLURM_CPUS_PER_TASK']}")
    else:
        launcher = []
    cmd = [*prefix, sys.executable, *launcher, "-m", module] + flags
    print("Running command:", cmd, flush=True)
    try:
        subprocess.run(cmd, check=True, env={**os.environ, GIT_ENV: git_file.name})
    finally:
        os.remove(git_file.name)
