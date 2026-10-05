"""Submit-side helpers shared by every spec in configs/ -- the one non-spec file
here. Runs on the submitting machine only, never on the cluster.

Importing this registers the after-submit callbacks. A spec uses:

    from mrunner.helpers.specification_helper import create_experiments_helper
    from configs._helper_functions import HELPER_KWARGS, stamped

The repo root is on sys.path via the `mrun` task's PYTHONPATH (pixi.toml) --
mrunner exec()s specs from a console script, so the cwd isn't there by default.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

from mrunner.cli.mrunner_cli import register_after_run_callback

# Taken once per submit, shared by the experiment name and the cemetery file
STAMP = datetime.now().strftime("%Y_%m_%d_%H%M")
CEMETERY = Path("configs/cemetery")

# Paths mrunner does NOT upload; the helper appends .git/.gitignore/.gitmodules.
#
# The pixi bits are excluded on purpose: `.pixi` is a local, platform-specific
# env with a huge file count, and the manifest is kept out so nothing on the
# cluster can accidentally materialize an env inside the copied repo (home has a
# file-count quota). The remote env lives at $PIXI_HOME and is activated by
# `prolog_cmd` in clusters.yaml -- see update_pixi.py.
EXCLUDE = [
    ".pixi",
    "pixi.toml",
    "pixi.lock",
    ".vscode",
    "__pycache__",
    "out",
    "configs/cemetery",  # submitted-spec copies; already in git
]

HELPER_KWARGS = dict(
    # plain `python`, not `pixi run` -- prolog_cmd already activated $PIXI_HOME
    script="python mrunner_run.py",
    python_path="",  # required by mrunner, no default
    # only way to turn off mrunner's default neptune logger; the FutureWarning
    # it prints is harmless
    with_neptune=False,
    exclude=EXCLUDE,
)


def stamped(name):
    return f"{STAMP}_{name}"


def _print_log_location(sweep, experiments):
    """Called by mrunner right after the job is submitted."""
    log = f"{sweep.grid_logs_dir}/slurm_"
    log += "0.log" if len(experiments) == 1 else "<array_task_id>.log"
    print(f"\nlog: ssh {sweep.slurm_url} tail -f {log}")


def _bury_spec(_sweep, _experiments):
    """Copy the spec just submitted (e.g. configs/toy.py) to
    configs/cemetery/<STAMP>_toy.py.
    """
    spec = Path(sys.argv[sys.argv.index("run") + 1])
    CEMETERY.mkdir(parents=True, exist_ok=True)
    dst = CEMETERY / f"{STAMP}_{spec.stem}.py"
    shutil.copy(spec, dst)
    print(f"spec: {dst}")


register_after_run_callback(_print_log_location)
register_after_run_callback(_bury_spec)
