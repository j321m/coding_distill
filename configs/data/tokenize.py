"""Pre-tokenize the full Python split on the cluster.

pixi run mrun --context helios run configs/data/tokenize.py
"""

from configs.common import make_experiment

TOKENIZE = dict(
    dataset="open-r1/codeforces-cots",
    config="solutions_py_decontaminated",
    split="train",
    tokenizer="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    max_length=32768,
    drop_finish_reason="length",
    test_size=0.05,
    split_seed=0,
)

experiment = make_experiment(
    "tokenize",
    dict(**TOKENIZE, data_dir="$DATA_DIR", num_proc=64),
    script="python -m scripts.tokenize_dataset",
)

experiments_list = [experiment]
