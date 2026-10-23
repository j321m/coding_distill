"""Pre-tokenize the C++ split, as OlympicCoder-7B was trained on.

pixi run mrun --context helios run configs/data/tokenize_cpp.py
"""

from configs.common import make_experiment

TOKENIZE_CPP = dict(
    dataset="open-r1/codeforces-cots",
    config="solutions_decontaminated",
    split="train",
    tokenizer="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_length=32768,
    drop_finish_reason="length",
    # by problem, not by row: five traces share an id. 1% of ~8.1k problems is
    # ~400 rows, which already costs ~13min per evaluation at 7B
    test_size=0.01,
    split_seed=0,
)

experiment = make_experiment(
    "tokenize_cpp",
    dict(**TOKENIZE_CPP, data_dir="$DATA_DIR", num_proc=64),
    script="python -m scripts.tokenize_dataset",
)

experiments_list = [experiment]
