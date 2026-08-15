"""Entrypoint. Runs under mrunner on the cluster, or from a JSON file locally.

  local:   pixi run python main.py --config configs/tiny_local.json
  cluster: mrunner --config ~/.mrunner.yaml --context <ctx> run mrunner_specs/sft_baseline.py
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s", stream=sys.stdout
)


def load_params() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON file of flat params")
    args, _ = parser.parse_known_args()

    if args.config is not None:
        with open(args.config) as f:
            return json.load(f)

    # no --config: assume mrunner injected the params
    from mrunner.helpers.client_helper import get_configuration

    return dict(get_configuration(print_diagnostics=True, with_neptune=False))


def main():
    from coding_distill.config import config_from_params
    from coding_distill.train import run

    cfg = config_from_params(load_params())
    run(cfg)


if __name__ == "__main__":
    main()
