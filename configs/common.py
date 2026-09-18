"""Submit-side machinery shared by every config.

Configs stay pure parameter dicts; everything that is true of *all* experiments
-- entry point, what gets shipped, code provenance, the log-location print --
lives here, so a new config is a name and its parameters.
"""

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

# mrunner tars up everything in the cwd that is not listed here. Note that
# providing `exclude` REPLACES mrunner's default of [.git, .gitignore,
# .gitmodules], so those have to be repeated.
#
# The pixi bits are excluded on purpose: `.pixi` is a local, platform-specific
# env with a huge file count, and the manifest is kept out so nothing on the
# cluster can accidentally materialize an env inside the copied repo (home has a
# file-count quota). The remote env lives at $PIXI_HOME and is activated by
# `prolog_cmd` in clusters.yaml -- see update_pixi.py.
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
]


def _git_info():
    """Provenance of the code being submitted, read on the laptop because `.git`
    is excluded above -- the compute node cannot recover any of this.

    The diff matters as much as the sha: mrunner ships the *working tree*, so a
    sha recorded next to uncommitted edits is a claim `git checkout` cannot
    honour. With the patch, a dirty run stays reproducible.
    """

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
    """Prefix a tokenize config for main.py's `data_` nesting.

    Training configs derive their data block from the same dict the tokenize job
    used, so the two cannot drift; `build_dataset` still asserts it against the
    artifact's meta.json.
    """
    return {f"data_{key}": value for key, value in {**tokenize_params, **extra}.items()}


def make_experiment(name, parameters, script="python main.py", **kwargs):
    """Build an `Experiment` with the invariants filled in.

    `git_*` rides in `parameters` because the slurm backend ignores
    `Experiment.env` (only `backends/k8s.py` reads it), so the pickled parameters
    are the only channel to the job. `src/run_dir.py` lifts these keys back out
    into the `git` block of `args.json`, keeping them out of the hyperparameters.

    `script` defaults to training; the tokenization job overrides it.
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
