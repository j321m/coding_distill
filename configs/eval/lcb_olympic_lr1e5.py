"""Score the 1e-5 run's checkpoints on LiveCodeBench.

Epochs 2, 4 and 6 of the rerun at the recipe's learning rate. The question is
lcb/pct_think_closed: the 4e-5 run closed <think> on 2-3 of 27 generations at
every epoch it was measured.

    pixi run mrun --context helios-eval run configs/eval/lcb_olympic_lr1e5.py
"""

from configs.common import make_experiment
from configs.eval.lcb import COMMON, SCRIPT

TRAIN_JOB_ID = "24931873"
CHECKPOINTS = ["checkpoint-604", "checkpoint-1208", "checkpoint-1812"]

experiments_list = [
    make_experiment(
        f"eval_lcb_lr1e5_{checkpoint}",
        dict(
            train_job_id=TRAIN_JOB_ID,
            checkpoint=checkpoint,
            model_name=f"olympic-lr1e5-{checkpoint}",
            **COMMON,
        ),
        script=SCRIPT,
    )
    for checkpoint in CHECKPOINTS
]
