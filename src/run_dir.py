"""Where a run writes, and what it records about itself."""

import json
import os
import platform
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from munch import unmunchify

# Dumped into args.json and into `wandb.config` under `job`. One list, two
# consumers, so the run dir and the wandb page cannot disagree.
JOB_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_NUM_NODES",
    "SLURM_CPUS_ON_NODE",
    "SLURM_GPUS_ON_NODE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    # which out-of-tree env was activated, and where the model cache came from
    "PIXI_HOME",
    "HF_HOME",
)

VERSIONED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "wandb",
    "liger-kernel",
    "triton",
)

PATCH_FILE = "git.patch"


def resolve_output_dir(output_dir):
    """`output_dir` is a base shared by every run, so give each job its own
    subdirectory. mrunner gives runs a unique cwd for free, but checkpoints live
    outside the copied repo, so uniqueness has to come from the job id."""
    expanded = os.path.expandvars(output_dir)
    if "$" in expanded:
        # The base comes from prolog_cmd. Unexpanded, it would create a literal
        # `$RUNS_DIR` under the code snapshot on $HOME -- the one filesystem this
        # whole out-of-tree arrangement exists to keep runs off.
        raise SystemExit(
            f"unresolved variable in output_dir: {output_dir!r} -> {expanded!r}; "
            "it must be exported by prolog_cmd in clusters.yaml"
        )
    run_id = os.environ.get("SLURM_JOB_ID", "local")  # unique per array task
    return os.path.join(expanded, run_id)


def _versions():
    installed = {}
    for package in VERSIONED_PACKAGES:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    return installed


def write_run_metadata(output_dir, params):
    """Write `args.json` (and `git.patch`, if the submitted tree was dirty).

    Called before the model and the dataset are built: a run that dies while
    loading is precisely the one whose config you want to read afterwards.
    """
    is_rank_zero = os.environ.get("RANK", "0") == "0"
    recorded = unmunchify(params)
    # configs/common.py injects these into the parameters because the slurm
    # backend has no other channel to the job. Lift them back out: provenance is
    # not a hyperparameter, and a diff is a file rather than a config value.
    commit = recorded.pop("git_commit", None)
    dirty = recorded.pop("git_dirty", None)
    diff = recorded.pop("git_diff", None)
    if diff and is_rank_zero:
        with open(os.path.join(output_dir, PATCH_FILE), "w") as patch:
            patch.write(diff)

    metadata = {
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "params": recorded,
        "job": {key: os.environ.get(key) for key in JOB_ENV_KEYS},
        "git": {
            # captured at submit time: .git is excluded from the mrunner
            # snapshot, so none of this is recoverable on the compute node
            "commit": commit,
            "dirty": dirty,
            "patch": PATCH_FILE if diff else None,
        },
        "host": platform.node(),
        "output_dir": output_dir,
        "versions": _versions(),
    }

    if is_rank_zero:
        with open(os.path.join(output_dir, "args.json"), "w") as args_file:
            json.dump(metadata, args_file, indent=2, default=str)
    return metadata
