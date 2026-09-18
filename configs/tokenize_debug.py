"""Local subset of the same dataset, short enough to train on a laptop.

pixi run tokenize-debug
"""

from configs.common import make_experiment
from configs.tokenize import TOKENIZE

experiment = make_experiment(
    "tokenize_debug",
    dict(
        **TOKENIZE,
        data_dir="out/data",
        num_proc=8,
        # a threshold rather than "the N shortest", which would pick the most
        # atypical rows: trivial problems with stub reasoning. 1024 is what a
        # 0.5B in fp32 fits on MPS alongside Adam state.
        filter_max_tokens=1024,
        filter_n=64,
    ),
    script="python -m scripts.tokenize_dataset",
)

experiments_list = [experiment]
