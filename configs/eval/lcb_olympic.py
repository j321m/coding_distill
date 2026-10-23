"""Score every checkpoint of the OlympicCoder reproduction on LiveCodeBench.

One job per checkpoint, so the epochs can be compared on the same problems --
which is also the overfitting check: 10 epochs over 8.1k problems is a lot of
passes, and the last checkpoint is not necessarily the best one.

    ls -d $RUNS_DIR/<job>/checkpoint-*        # fill CHECKPOINTS from this
    pixi run mrun --context helios-eval run configs/eval/lcb_olympic.py
"""

from configs.common import make_experiment
from configs.eval.lcb import COMMON, SCRIPT

TRAIN_JOB_ID = "24691030"
# save_total_limit=3 with save_strategy="epoch" keeps the last three
CHECKPOINTS = ["checkpoint-2416", "checkpoint-2718", "checkpoint-3020"]

experiments_list = [
    make_experiment(
        f"eval_lcb_olympic_{checkpoint}",
        dict(
            train_job_id=TRAIN_JOB_ID,
            checkpoint=checkpoint,
            # distinct, or the runs overwrite each other's output/ directory
            model_name=f"olympic-7b-{checkpoint}",
            **COMMON,
        ),
        script=SCRIPT,
    )
    for checkpoint in CHECKPOINTS
]
