#!/usr/bin/env python
"""Maintain the out-of-tree pixi environment on a cluster.

The cluster $HOME has a file-count quota, and a pixi env (plus its rattler/uv
caches) is by far the biggest contributor. So the env is not shipped with the
code: `pixi.toml` / `pixi.lock` / `.pixi` are excluded from the mrunner code
snapshot (see configs/toy.py) and the env lives at $PIXI_HOME on a storage disk,
where this script maintains it.

$PIXI_HOME and friends are defined once per context, in the `prolog_cmd` of
clusters.yaml. This script pulls the `export` / `module load` lines out of that
same block (ignoring the activation line, which would fail before the env
exists), so there is no second place to keep in sync.

Usage:
    pixi run python update_pixi.py --context entropy
    pixi run python update_pixi.py --context entropy --dry-run
    pixi run python update_pixi.py --context entropy --login   # skip srun
    pixi run python update_pixi.py --context entropy --verbose # `pixi install -v`

Every run is timestamped and logged twice: on the cluster in
~/update_pixi/<timestamp>/install.log, and locally in out/update_pixi/.
"""

import argparse
import copy
import getpass
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import paramiko.ssh_exception
import yaml
from fabric import Connection

# sbatch options that make no sense for a CPU-only `pixi install` job
GPU_OPTION_PREFIXES = ("--gres", "--gpus", "--cpus-per-gpu", "--mem-per-gpu")
# options we set ourselves for the install job
OVERRIDDEN_OPTION_PREFIXES = (
    "--cpus-per-task",
    "--mem",
    "--time",
    "--job-name",
    "--ntasks",
    "--output",
)
INSTALL_JOB_OPTIONS = [
    "--job-name=update_pixi",
    # the install is dominated by unpacking the torch/CUDA wheels, which uv does
    # in parallel -- more cores is the cheapest speedup available here
    "--cpus-per-task=16",
    "--mem=32G",
    "--time=1:00:00",
]

_SSH_HOSTS_TO_PASSPHRASES = {}


@contextmanager
def connect_with_passphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host, prompting for a passphrase if the key is encrypted."""
    connection = None
    try:
        connection = Connection(*args, **kwargs)
        connection.run('echo "Connection successful."', hide=True)
        yield connection
    except paramiko.ssh_exception.PasswordRequiredException:
        host = kwargs.get("host", args[0] if args else None)
        if host not in _SSH_HOSTS_TO_PASSPHRASES:
            _SSH_HOSTS_TO_PASSPHRASES[host] = getpass.getpass(
                f"SSH key encrypted, provide the passphrase ({host}): "
            )
        kwargs["connect_kwargs"] = copy.deepcopy(kwargs.get("connect_kwargs", {}))
        kwargs["connect_kwargs"]["passphrase"] = _SSH_HOSTS_TO_PASSPHRASES[host]
        connection = Connection(*args, **kwargs)
        yield connection
    finally:
        if connection is not None:
            connection.close()


def load_context(clusters_path: Path, context_name: str) -> dict:
    with open(clusters_path) as file:
        config = yaml.full_load(file) or {}

    contexts = config.get("contexts") or {}
    if context_name not in contexts:
        raise SystemExit(
            f"Context '{context_name}' not found in {clusters_path}. "
            f"Available: {', '.join(sorted(contexts))}"
        )
    return contexts[context_name]


def env_setup_lines(prolog_cmd: str) -> str:
    """Keep only the env-setup part of prolog_cmd (exports + module loads).

    The activation line (`pixi shell-hook`) is dropped on purpose: it fails until
    the env actually exists, which is what we are about to create.
    """
    kept = []
    for raw in prolog_cmd.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith(("module load", "ml ")):
            kept.append(line)

    if not any(line.startswith("export PIXI_HOME") for line in kept):
        raise SystemExit(
            "prolog_cmd must contain an `export PIXI_HOME=...` line -- that is "
            "where the out-of-tree env is kept."
        )
    return "\n".join(kept)


def install_job_options(sbatch_options: list[str]) -> list[str]:
    """Strip GPU/overridden options from the context's sbatch options."""
    dropped = GPU_OPTION_PREFIXES + OVERRIDDEN_OPTION_PREFIXES
    kept = [
        option
        for option in sbatch_options
        if option.split("=")[0].strip() not in dropped
    ]
    return kept + INSTALL_JOB_OPTIONS


def build_install_script(env_setup: str, staging_dir: str, verbose: bool) -> str:
    """Bash driver for the install, run either under srun or on the login node.

    Everything it prints is timestamped and teed into `{staging_dir}/install.log`
    on the cluster, so a slow/stuck install can be attributed to a phase (solve
    vs. download vs. unpack) instead of guessed at.
    """
    install_flags = " -v" if verbose else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

log_file="{staging_dir}/install.log"
exec > >(tee -a "$log_file") 2>&1

start=$(date +%s)
step() {{
  echo "[$(date +%H:%M:%S) +$(( $(date +%s) - start ))s] $*"
}}
trap 'step "exited with status $?"' EXIT

step "host=$(hostname) log=$log_file"

{env_setup}

if ! command -v pixi > /dev/null; then
  echo "pixi not found on PATH ($PATH). Install it first: curl -fsSL https://pixi.sh/install.sh | bash" >&2
  exit 1
fi

step "pixi $(pixi --version 2>&1 | tail -1), PIXI_HOME=$PIXI_HOME"
mkdir -p "$PIXI_HOME"

step "filesystem holding PIXI_HOME:"
df -hT "$PIXI_HOME" | sed 's/^/    /'

# keep the previous manifests around, so a bad update can be traced/reverted
ts=$(date +%Y_%m_%d_%H_%M_%S)
if [ -f "$PIXI_HOME/pixi.toml" ] || [ -f "$PIXI_HOME/pixi.lock" ]; then
  step "backing up previous manifests"
  backup="$PIXI_HOME/old_pixi_files/obsolete_since_$ts"
  mkdir -p "$backup"
  [ -f "$PIXI_HOME/pixi.toml" ] && mv -f "$PIXI_HOME/pixi.toml" "$backup/"
  [ -f "$PIXI_HOME/pixi.lock" ] && mv -f "$PIXI_HOME/pixi.lock" "$backup/"
fi

step "installing new manifests"
mv -f "{staging_dir}/pixi.toml" "$PIXI_HOME/"
if [ -f "{staging_dir}/pixi.lock" ]; then
  mv -f "{staging_dir}/pixi.lock" "$PIXI_HOME/"
fi

step "running pixi install (this is the slow part: ~GBs of torch/CUDA wheels)"
# stdout is a pipe (see the tee above), and pixi hides its progress bars unless
# it is talking to a terminal -- so hand it a pty via `script`, which still lets
# everything flow through the tee into the log
install_cmd="pixi install{install_flags} --manifest-path $PIXI_HOME/pixi.toml"
if command -v script > /dev/null; then
  script -qefc "$install_cmd" /dev/null
else
  $install_cmd
fi

step "pixi env at $PIXI_HOME is up to date."
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", required=True, help="context name in clusters.yaml"
    )
    parser.add_argument(
        "--config", default="clusters.yaml", help="path to the mrunner cluster config"
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="run `pixi install` on the login node instead of allocating with srun",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="pass -v to `pixi install` (shows the solve/fetch/link phases)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    pixi_toml = project_root / "pixi.toml"
    pixi_lock = project_root / "pixi.lock"
    if not pixi_toml.exists():
        raise SystemExit(f"pixi.toml not found at {pixi_toml}")

    context = load_context(project_root / args.config, args.context)
    host = context.get("slurm_url", args.context)
    env_setup = env_setup_lines(context.get("prolog_cmd") or "")
    options = install_job_options(context.get("sbatch_options") or [])
    srun_cmd = "srun " + " ".join(options)

    print(f"Cluster: {args.context} (ssh host: {host})")
    print(
        f"Local manifests: {pixi_toml}" + ("" if pixi_lock.exists() else " (no lock)")
    )
    print("\nEnv setup taken from prolog_cmd:")
    for line in env_setup.splitlines():
        print(f"  {line}")

    if args.dry_run:
        print("\n[DRY RUN] Would:")
        print(f"  1. connect to {host}")
        print("  2. stage pixi.toml/pixi.lock in ~/update_pixi/<timestamp>")
        if args.login:
            print("  3. run the install script on the login node")
        else:
            print(f"  3. run: {srun_cmd} bash <script>")
        print("  4. move the manifests into $PIXI_HOME and run `pixi install` there")
        return

    with connect_with_passphrase(host=host, inline_ssh_env=True) as connection:
        home = connection.run("cd && pwd", hide=True).stdout.strip()
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # only ever 3 tiny files, so staging in $HOME is harmless
        staging_dir = f"{home}/update_pixi/{timestamp}"
        connection.run(f"mkdir -p {staging_dir}", hide=True)

        print(f"\nStaging manifests in {staging_dir} ...")
        connection.put(str(pixi_toml), remote=f"{staging_dir}/pixi.toml")
        if pixi_lock.exists():
            connection.put(str(pixi_lock), remote=f"{staging_dir}/pixi.lock")

        script = build_install_script(env_setup, staging_dir, args.verbose)
        with tempfile.NamedTemporaryFile("w", suffix=".sh") as local_script:
            local_script.write(script)
            local_script.flush()
            remote_script = f"{staging_dir}/install_pixi.sh"
            connection.put(local_script.name, remote=remote_script)

        command = f"bash {remote_script}"
        if not args.login:
            command = f"{srun_cmd} {command}"
        print(f"\nRunning: {command}")
        print(f"Remote log: {host}:{staging_dir}/install.log (tail -f it to watch)\n")

        result = connection.run(command, pty=True, warn=True)

        local_log = project_root / "out" / "update_pixi" / f"{timestamp}.log"
        local_log.parent.mkdir(parents=True, exist_ok=True)
        local_log.write_text(result.stdout)
        print(f"\nLog saved to {local_log}")

    if result.ok:
        print("✓ pixi environment updated")
    else:
        print("✗ pixi install failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
