import subprocess

from munch import Munch
from mrunner.cli.mrunner_cli import register_after_run_callback
from mrunner.experiment import Experiment


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


def data_params(tokenize_params, **extra):
    """Prefix a tokenize config for main.py's `data_` nesting."""
    return {f"data_{key}": value for key, value in {**tokenize_params, **extra}.items()}


def make_experiment(name, parameters, script="python main.py", **kwargs):
    """Build an `Experiment` with the invariants filled in.

    `git_*` rides in `parameters` because the slurm backend ignores
    `Experiment.env` (only `backends/k8s.py` reads it), so the pickled parameters
    are the only channel to the job.
    """
    commit, dirty, diff = _git_info()
    return Experiment(
        name=name,
        # plain `python`, not `pixi run` -- prolog_cmd already activated $PIXI_HOME
        script=script,
        exclude=EXCLUDE,
        # flat so mrunner can sweep them; main.py nests by prefix
        parameters=Munch(
            git_commit=commit,
            git_dirty=dirty,
            git_diff=diff,
            **parameters,
        ),
        **kwargs,
    )
