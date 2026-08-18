"""Shared experiment params, in Python.

`BASE` is the single source of truth for a full run. Local configs and mrunner
specs both start from it and apply `override`, so the two paths cannot drift.

This lives in the installed package on purpose: mrunner loads a spec with a bare
`exec(open(script).read())` and never touches `sys.path`, so an installed module
is the only thing a spec can reliably import. `configs/*.py` are thin files that
import from here.

Nothing here imports mrunner -- local runs must work without it.
"""

from coding_distill.config import config_from_params

# The cluster baseline: SFT-distill a small coder on codeforces-cots traces.
BASE = {
    "data.dataset_name": "open-r1/codeforces-cots",
    "data.dataset_config": "solutions_py",
    "data.split": "train",
    "data.max_seq_length": 16384,
    "data.n_train_samples": None,
    "data.n_eval_samples": 200,
    "data.mask_prompt": True,
    "data.num_proc": 8,
    "model.student_name": "Qwen/Qwen2.5-Coder-1.5B",
    "model.attn_implementation": "flash_attention_2",
    "model.torch_dtype": "bfloat16",
    "model.gradient_checkpointing": True,
    "optim.learning_rate": 1e-5,
    "optim.lr_scheduler_type": "cosine",
    "optim.warmup_ratio": 0.03,
    "optim.weight_decay": 0.0,
    "optim.max_grad_norm": 1.0,
    "optim.n_epochs": 3,
    "optim.per_device_batch_size": 1,
    "optim.grad_accumulation_steps": 16,
    "optim.seed": 42,
    "eval.benchmarks": ["humanevalplus", "livecodebench"],
    "eval.n_samples_per_problem": 1,
    "eval.temperature": 0.0,
    "eval.top_p": 1.0,
    "eval.max_new_tokens": 4096,
    "eval.run_at_end": False,  # flip on once evaluation.py is implemented
    "logging.project": "coding_distill",
    "logging.run_name": "sft_baseline",
    "logging.tags": ["baseline", "codeforces-cots"],
    "logging.output_dir": "./out",
    "logging.save_steps": 500,
    "logging.logging_steps": 10,
    "logging.use_wandb": True,
}


def override(base: dict, changes: dict) -> dict:
    """Return `base` with `changes` applied, rejecting keys that aren't already there.

    `base` is complete by construction, so a key that isn't in it is a typo. Catching
    it here means a bad config file fails on import rather than after the cluster queue.
    """
    unknown = sorted(set(changes) - set(base))
    if unknown:
        raise KeyError(f"override introduces keys absent from base: {unknown}")
    return {**base, **changes}


def _grid_key(key: str) -> str:
    """Strip mrunner's `___` zip-axis marker to get the real param name."""
    return key[:-3] if key.endswith("___") else key


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("/", "-").replace(" ", "")


def label_experiments(experiments, params_grid: dict, *, output_root: str) -> None:
    """Give each grid point its own `run_name` and `output_dir`, in place.

    Without this every point in the sweep shares one output dir and one wandb name:
    the checkpoints overwrite each other and the runs are indistinguishable.
    """
    keys = [_grid_key(k) for k in params_grid]
    for experiment in experiments:
        params = experiment.parameters
        suffix = "-".join(f"{k.split('.')[-1]}{_fmt(params[k])}" for k in keys)
        if suffix:
            params["logging.run_name"] = f"{params['logging.run_name']}-{suffix}"
        params["logging.output_dir"] = f"{output_root}/{params['logging.run_name']}"


def validate_experiments(experiments) -> None:
    """Parse every grid point through the real schema, before anything is submitted.

    mrunner ships params to the cluster as an opaque pickle, so an unvalidated typo
    surfaces as a failed job. This turns that into an exception on the laptop.
    """
    for experiment in experiments:
        config_from_params(dict(experiment.parameters))
