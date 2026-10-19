"""Local subset of the same dataset, short enough to train on a laptop.

pixi run tokenize-debug
"""

from configs.common import data_params, make_experiment
from configs.data.tokenize import TOKENIZE

experiment = make_experiment(
    "tokenize_debug",
    {
        **data_params(
            TOKENIZE,
            data_dir="out/data",
            # a threshold rather than "the N shortest", which would pick the most
            # atypical rows: trivial problems with stub reasoning. 1024 is what a
            # 0.5B in fp32 fits on MPS alongside Adam state.
            filter_max_tokens=1024,
            filter_n=64,
        ),
        "num_proc": 8,
    },
    entry_point="scripts.tokenize_dataset",
)

experiments_list = [experiment]
