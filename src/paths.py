"""Resolving the cluster paths that `prolog_cmd` exports."""

import os


def expand_path(value, name):
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise SystemExit(
            f"unresolved variable in {name}: {value!r} -> {expanded!r}; "
            "it must be exported by prolog_cmd in clusters.yaml"
        )
    return expanded


def resolve_output_dir(output_dir):
    run_id = os.environ.get("SLURM_JOB_ID", "local")  # unique per array task
    return os.path.join(expand_path(output_dir, "output_dir"), run_id)
