"""LiveCodeBench evaluation: one job per entry in MODELS.

    pixi run mrun --context entropy-eval run configs/eval/eval_entropy.py

All entries share COMMON, so the models are scored on the same problem set with
the same decoding.
"""

from configs.common import make_experiment

COMMON = dict(
    scenario="codegeneration",
    release_version="release_v6",
    start_date="2024-08-01",
    end_date="2024-08-15",
    n=1,
    # DeepSeek's documented settings for R1-style distills; LiveCodeBench's 0.2
    # default sends long CoT into repetition loops
    temperature=0.6,
    top_p=0.95,
    # vllm rejects prompt + max_tokens above max_position_embeddings (32768),
    # so this leaves ~2k for the prompt
    max_tokens=30720,
    dtype="bfloat16",
    tensor_parallel_size=1,
    report_to="wandb",
)

# `model_path` loads from the hub; without it the checkpoint is read from
# $RUNS_DIR/<train_job_id>/<checkpoint>. `train_job_id` also decides where
# results land, at $RUNS_DIR/<train_job_id>/lcb/<model_name>/ -- hub models
# belong to no training run, so they are parked under BASELINES instead.
BASELINES = "baselines"
MODELS = [
    (
        "eval_lcb_math7b",
        dict(
            train_job_id="29047",
            checkpoint="checkpoint-5",
            model_path=None,
            model_name="sft-5step",
        ),
    )
]

experiments_list = [
    make_experiment(name, {**COMMON, **params}, entry_point="scripts.eval_lcb")
    for name, params in MODELS
]
