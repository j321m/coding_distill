#!/usr/bin/env python
"""Pull run metadata and eval outputs off a cluster, for local analysis.

$RUNS_DIR is defined once, in the `prolog_cmd` of clusters.yaml. Rather than
repeat the path here, this resolves it the way update_pixi.py does: by running
that block's export lines on the cluster and asking what it came out as.

    pixi run python sync_runs.py --context helios
    pixi run python sync_runs.py --context helios --dry-run
"""

import argparse
import subprocess
from pathlib import Path

from update_pixi import connect_with_passphrase, env_setup_lines, load_context

# Only what analysis needs: each run's args.json and LiveCodeBench's three
# output files. Everything else is excluded rather than named, which also keeps
# `lcb/<model>/model/` out -- those entries are symlinks into multi-GB
# checkpoints, and an rsync that followed them would pull the weights too.
FILTERS = [
    "--include=*/",
    "--include=args.json",
    "--include=Scenario.*.json",
    "--exclude=*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, help="context name in clusters.yaml")
    parser.add_argument("--config", default="clusters.yaml")
    parser.add_argument("--dest", default="out/runs", help="local destination")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    context = load_context(project_root / args.config, args.context)
    host = context.get("slurm_url", args.context)
    env_setup = env_setup_lines(context.get("prolog_cmd") or "")

    with connect_with_passphrase(host=host, inline_ssh_env=True) as connection:
        runs_dir = connection.run(f"{env_setup}\necho $RUNS_DIR", hide=True).stdout.strip()
    if not runs_dir or "$" in runs_dir:
        raise SystemExit(f"could not resolve RUNS_DIR from {args.context}: {runs_dir!r}")

    dest = project_root / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    # trailing slashes: copy the *contents* of runs/ into dest, not runs/ itself
    command = [
        "rsync",
        "-avz",
        "--prune-empty-dirs",
        *(["--dry-run"] if args.dry_run else []),
        *FILTERS,
        f"{host}:{runs_dir}/",
        f"{dest}/",
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
