"""Resolving the cluster paths that `prolog_cmd` exports."""

import os


def expand_path(value, name):
    """`os.path.expandvars`, but a variable that did not resolve is fatal.

    $DATA_DIR and $RUNS_DIR come from `prolog_cmd` in clusters.yaml. Left
    unexpanded they would silently create a literal `$RUNS_DIR` directory under
    the mrunner code snapshot on $HOME -- the one filesystem this whole
    out-of-tree arrangement exists to keep runs off.
    """
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise SystemExit(
            f"unresolved variable in {name}: {value!r} -> {expanded!r}; "
            "it must be exported by prolog_cmd in clusters.yaml"
        )
    return expanded


def resolve_output_dir(output_dir):
    """Give each job its own subdirectory under the shared base.

    mrunner gives runs a unique cwd for free, but checkpoints live outside the
    copied repo, so uniqueness has to come from the job id.
    """
    run_id = os.environ.get("SLURM_JOB_ID", "local")  # unique per array task
    return os.path.join(expand_path(output_dir, "output_dir"), run_id)
