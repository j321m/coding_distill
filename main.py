"""Entrypoint. Runs under mrunner on the cluster, or from a Python config locally.

  local:        pixi run python main.py --local-config configs/tiny_local.py
  local (spec): pixi run python main.py --ex mrunner_specs/sft_baseline.py
  cluster:      mrunner --config ~/.mrunner.yaml --context <ctx> run mrunner_specs/sft_baseline.py

The local flag is `--local-config`, not `--config`: mrunner launches the remote job
as `python3 main.py --config config_$SLURM_ARRAY_TASK_ID`, where that file is a
cloudpickle dump it reads itself. Reusing the name would shadow it.
"""

import argparse
import importlib.util
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(name)s] %(message)s",
    stream=sys.stdout,
)


def load_local_config(path: str) -> dict:
    """Load a flat params dict from a `.py` file exposing `params`, or from JSON."""
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)

    spec = importlib.util.spec_from_file_location("_local_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import a config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "params"):
        raise AttributeError(f"{path} defines no `params` dict")
    return dict(module.params)


def load_params() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--local-config", default=None, help="Python or JSON file of flat params"
    )
    args, _ = parser.parse_known_args()

    if args.local_config is not None:
        return load_local_config(args.local_config)

    # No --local-config: mrunner supplies the params, via --config (the remote
    # pickle) or --ex (a spec file, whose first experiment is used).
    from mrunner.helpers.client_helper import get_configuration

    return dict(get_configuration(print_diagnostics=True))


def main():
    from coding_distill.config import config_from_params
    from coding_distill.train import run

    cfg = config_from_params(load_params())
    run(cfg)


if __name__ == "__main__":
    main()
