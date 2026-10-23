"""Does the model terminate when asked the way it was trained?

The Python eval showed 8 of 10 generations never closing <think>, while drafting
code over and over. This asks the same problems with the verbatim C++ prompt
from solutions_decontaminated -- the one the model actually saw -- to separate
"the model cannot finish" from "the prompt is off distribution".

LiveCodeBench executes Python, so pass@1 is meaningless here. Read
lcb/pct_think_closed and lcb/pct_with_fence instead.

    pixi run mrun --context helios-eval run configs/eval/lcb_olympic_cpp.py
"""

from configs.common import make_experiment
from configs.eval.lcb import COMMON, SCRIPT
from configs.eval.lcb_olympic import TRAIN_JOB_ID

CHECKPOINT = "checkpoint-3020"

experiment = make_experiment(
    "eval_lcb_olympic_cpp",
    dict(
        train_job_id=TRAIN_JOB_ID,
        checkpoint=CHECKPOINT,
        model_name=f"olympic-7b-{CHECKPOINT}-cpp",
        prompt_language="cpp",
        **COMMON,
    ),
    script=SCRIPT,
)

experiments_list = [experiment]
