"""Pre-tokenize the full Python split on the cluster.

pixi run mrun --context helios run configs/data/tokenize.py
"""

from configs.common import make_experiment

experiment = make_experiment(
    "tokenize",
    {
        "data.dataset": "open-r1/codeforces-cots",
        "data.config": "solutions_py_decontaminated",
        "data.split": "train",
        "data.tokenizer": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "data.max_length": 32768,
        "data.drop_finish_reason": "length",
        "data.test_size": 0.05,
        "data.split_seed": 0,
        "data.data_dir": "$DATA_DIR",
        "data.filter_max_tokens": None,
        "data.filter_n": None,
        "num_proc": 64,
    },
    entry_point="scripts.tokenize_dataset",
)

experiments_list = [experiment]
