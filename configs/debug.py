"""Tiny CPU run to check the pipeline end to end.

pixi run debug
"""

from munch import Munch
from mrunner.experiment import Experiment

from configs.toy import EXCLUDE

experiment = Experiment(
    name="debug",
    script="python main.py",
    exclude=EXCLUDE,
    # flat so mrunner can sweep them; main.py nests by prefix
    parameters=Munch(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        model_pad_token="<unk>",
        data_name="roneneldan/TinyStories",
        data_config=None,
        data_split="train[:64]",
        data_text_field="text",
        data_max_length=64,
        data_padding="max_length",
        data_truncation=True,
        train_output_dir="out/debug",
        train_learning_rate=1e-4,
        train_batch_size=4,
        train_num_epochs=1,
        train_max_steps=5,
        train_logging_steps=1,
        train_save_steps=1000,
        train_bf16=False,
    ),
)

experiments_list = [experiment]
