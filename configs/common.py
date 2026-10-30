import re
import subprocess

import click
from munch import Munch
from mrunner.cli.mrunner_cli import register_after_run_callback
from mrunner.experiment import Experiment

from src.config import ENTRY_POINTS


def _print_log_location(sweep, experiments):
    """Called by mrunner right after the job is submitted."""
    log = f"{sweep.grid_logs_dir}/slurm_"
    log += "0.log" if len(experiments) == 1 else "<array_task_id>.log"
    print(f"\nlog: ssh {sweep.slurm_url} tail -f {log}")


register_after_run_callback(_print_log_location)

EXCLUDE = [
    ".git",
    ".gitignore",
    ".gitmodules",
    ".pixi",
    "pixi.toml",
    "pixi.lock",
    ".vscode",
    "__pycache__",
    "out",
    "assets",
    "notebooks",
]


def _git_info():
    """Provenance of the code being submitted"""

    def git(*args):
        return subprocess.check_output(
            ("git",) + args, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        dirty = bool(git("status", "--porcelain"))
        return git("rev-parse", "HEAD"), dirty, git("diff", "HEAD") if dirty else ""
    except (subprocess.CalledProcessError, OSError):
        return None, None, ""  # e.g. a config re-run from inside the snapshot


def _context_gpus():
    """GPUs the mrunner context being submitted to requests, across all of its
    nodes: `--gres=gpu:N` in its sbatch_options times its `nodes` (1 when
    unset). None outside `mrunner run`, e.g. `pixi run debug`."""
    ctx = click.get_current_context(silent=True)
    if ctx is None or not ctx.obj:
        return None
    context = ctx.obj["context"]
    per_node = 0
    for option in context.get("sbatch_options", []):
        match = re.fullmatch(r"--gres=gpu:(\d+)", option)
        if match:
            per_node = int(match.group(1))
    return per_node * int(context.get("nodes", 1))


def make_experiment(name, parameters, entry_point="main", gpus=None, **kwargs):
    """Build an `Experiment` with the invariants filled in.

    Every job runs mrunner_run.py, which starts `entry_point` (a module, e.g.
    `scripts.eval_lcb`) with `parameters` as tyro flags.

    `gpus` is required for training and counts every node: the effective batch
    depends on it, so the submit fails unless the context grants exactly that
    many. mrunner_run.py starts one process per GPU on every node from it.

    `git_*` rides in `parameters` because the slurm backend ignores
    `Experiment.env` (only `backends/k8s.py` reads it), so the pickled parameters
    are the only channel to the job.
    """
    if entry_point not in ENTRY_POINTS:
        raise SystemExit(
            f"unknown entry_point {entry_point!r}; one of {list(ENTRY_POINTS)}"
        )
    if ENTRY_POINTS[entry_point]:
        if gpus is None:
            raise SystemExit(f"{name}: training specs must set `gpus`")
        granted = _context_gpus()
        if granted is not None and granted != gpus:
            raise SystemExit(
                f"{name}: spec expects gpus={gpus}, the context requests {granted} "
                "(--gres in clusters.yaml); fix the spec or the context"
            )

    commit, dirty, diff = _git_info()
    return Experiment(
        name=name,
        # plain `python`, not `pixi run` -- prolog_cmd already activated $PIXI_HOME
        script="python mrunner_run.py",
        exclude=EXCLUDE,
        # flat so mrunner can sweep them; dotted (`train.learning_rate`) for
        # tyro's nesting (src/config.py)
        parameters=Munch(
            entry_point=entry_point,
            gpus=gpus,
            experiment_name=name,
            git_commit=commit,
            git_dirty=dirty,
            git_diff=diff,
            **parameters,
        ),
        **kwargs,
    )
