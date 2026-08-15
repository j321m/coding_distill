"""Execution-based coding eval.

STUB -- not implemented yet. Deliberately separate from training: generation wants
vLLM and a sandbox, which do not belong in the training process.

Intended shape:
  1. spin up vLLM on the saved checkpoint
  2. sample `n_samples_per_problem` completions per problem
  3. extract the code block, run it against the benchmark's tests in a sandbox
  4. return {"eval/<benchmark>/pass@1": ...}

Run the sandbox on a compute node with no network, and treat generated code as
untrusted -- it executes arbitrary programs from a model.
"""

import logging

logger = logging.getLogger(__name__)


def evaluate(checkpoint_path: str, cfg) -> dict:
    raise NotImplementedError(
        f"eval not implemented: would score {checkpoint_path} on {cfg.benchmarks}"
    )
