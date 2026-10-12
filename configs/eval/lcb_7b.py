"""7B on LiveCodeBench: the SFT checkpoint against the base model it started from.

Two jobs from one submit. Both use the same prompt wrapper and the same decoding
settings, so the difference is the fine-tuning and nothing else -- and the base
model answers the question the 0.5B run could not: whether a near-zero score
means "this model is weak" or "this harness is broken".

Expect them to fail differently. The base Instruct model was never trained to
emit <think>, so it should answer short and fenced; the SFT model reasons at
length first. `lcb/tokens_p50` and `lcb/pct_with_fence` show which is happening.

    pixi run mrun --context helios-eval run configs/eval/lcb_7b.py
"""

from configs.common import make_experiment

TRAIN_JOB_ID = "24321228"  # the finished 7B run; results for both land under it
CHECKPOINT = "checkpoint-290"  # e.g. checkpoint-300 -- `ls $RUNS_DIR/<id>` to confirm

COMMON = dict(
    scenario="codegeneration",
    release_version="release_v6",
    start_date="2024-08-01",
    end_date="2024-08-15",
    n=1,
    temperature=0.6,
    top_p=0.95,
    # vllm rejects prompt + max_tokens > the model's 32768 max_model_len
    max_tokens=30720,
    dtype="bfloat16",
    tensor_parallel_size=1,
    report_to="wandb",
)

finetuned = make_experiment(
    "eval_lcb_7b_sft",
    dict(train_job_id=TRAIN_JOB_ID, checkpoint=CHECKPOINT, **COMMON),
    script="python -m scripts.eval_lcb",
)

# `model_path` loads from the hub instead of $RUNS_DIR; `model_name` keeps its
# output/ directory separate from the fine-tuned one under the same job id
base = make_experiment(
    "eval_lcb_7b_base",
    dict(
        train_job_id=TRAIN_JOB_ID,
        model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        model_name="qwen2.5-coder-7b-instruct-base",
        **COMMON,
    ),
    script="python -m scripts.eval_lcb",
)

experiments_list = [finetuned, base]
