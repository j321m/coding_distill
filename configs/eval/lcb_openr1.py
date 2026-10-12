"""Calibrate the harness against open-r1's own models.

Two reference points for the same 27 problems the local runs were scored on:

  Qwen2.5-Math-7B-RoPE-300k -- the base they started from (stock Qwen2.5-Math-7B
    with max_position_embeddings 4096->32768 and rope_theta 10k->300k; identical
    weight_map and total_size, so the config is the only difference)
  OpenR1-Distill-7B         -- what they got after training it

The gap between these two is what a working distillation looks like on this
benchmark. If OpenR1-Distill-7B also lands near 5%, the window is too small or
too hard to resolve anything and the local base-vs-SFT comparison is noise
rather than a result.

Same date window as configs/eval/lcb_7b.py on purpose: calibration only means
something on the identical problem set.

    pixi run mrun --context helios-eval run configs/eval/lcb_openr1.py

Caveat for the comparison: OpenR1-Distill-7B was trained on Mixture-of-Thoughts
(math + code + science), not on codeforces-cots alone, so it is a reference for
the harness rather than a target the local run should match.
"""

from configs.common import make_experiment

# results land beside the local 7B evals, under the same run directory
TRAIN_JOB_ID = "24321228"

COMMON = dict(
    scenario="codegeneration",
    release_version="release_v6",
    start_date="2024-08-01",
    end_date="2024-08-15",
    n=1,
    temperature=0.6,
    top_p=0.95,
    # both carry max_position_embeddings 32768; vllm rejects prompt + max_tokens
    # above it
    max_tokens=30720,
    dtype="bfloat16",
    tensor_parallel_size=1,
    report_to="wandb",
    train_job_id=TRAIN_JOB_ID,
)

distilled = make_experiment(
    "eval_lcb_openr1_distill",
    dict(
        model_path="open-r1/OpenR1-Distill-7B",
        model_name="openr1-distill-7b",
        **COMMON,
    ),
    script="python -m scripts.eval_lcb",
)

base = make_experiment(
    "eval_lcb_openr1_base",
    dict(
        model_path="open-r1/Qwen2.5-Math-7B-RoPE-300k",
        model_name="qwen2.5-math-7b-rope300k",
        **COMMON,
    ),
    script="python -m scripts.eval_lcb",
)

experiments_list = [distilled, base]
