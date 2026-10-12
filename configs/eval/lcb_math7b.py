"""Score the Math-7B-based SFT run on the same 27 problems as everything else.

No base-model job here: open-r1/Qwen2.5-Math-7B-RoPE-300k was already measured
at 0% by configs/eval/lcb_openr1.py, which is the baseline this run is against.
The reference points on this window are 0% for that base and 26% for
OpenR1-Distill-7B.

    pixi run mrun --context helios-eval run configs/eval/lcb_math7b.py
"""

from configs.common import make_experiment

TRAIN_JOB_ID = "24389672"  # the finished configs/train/sft_math7b.py run
CHECKPOINT = "checkpoint-290"  # `ls $RUNS_DIR/<id>` -- save_total_limit=2 keeps the last two

experiment = make_experiment(
    "eval_lcb_math7b",
    dict(
        train_job_id=TRAIN_JOB_ID,
        checkpoint=CHECKPOINT,
        model_name="sft-math7b",
        scenario="codegeneration",
        release_version="release_v6",
        start_date="2024-08-01",
        end_date="2024-08-15",
        n=1,
        # 0.6 / 0.95: DeepSeek's recommendation for R1-style distills.
        # LiveCodeBench's 0.2 default sent long CoT into repetition loops.
        temperature=0.6,
        top_p=0.95,
        max_tokens=30720,
        dtype="bfloat16",
        tensor_parallel_size=1,
        report_to="wandb",
    ),
    script="python -m scripts.eval_lcb",
)

experiments_list = [experiment]
