"""Local subset of the same dataset, short enough to train on a laptop.

pixi run tokenize-debug
"""

from configs.common import make_experiment

experiment = make_experiment(
    "tokenize_debug",
    {
        "data.dataset": "open-r1/codeforces-cots",
        "data.config": "solutions_py_decontaminated",
        "data.split": "train",
        "data.tokenizer": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "data.max_length": 32768,
        "data.drop_finish_reason": "length",
        "data.test_size": 0.05,
        "data.split_seed": 0,
        "data.data_dir": "out/data",
        # a threshold rather than "the N shortest", which would pick the most
        # atypical rows: trivial problems with stub reasoning. 1024 is what a
        # 0.5B in fp32 fits on MPS alongside Adam state.
        "data.filter_max_tokens": 1024,
        "data.filter_n": 64,
        "num_proc": 8,
    },
    entry_point="scripts.tokenize_dataset",
)

experiments_list = [experiment]
