import json
import os
import shutil
import sys

LCB_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "LiveCodeBench",
)
if not os.path.isdir(os.path.join(LCB_ROOT, "lcb_runner")):
    raise SystemExit(f"{LCB_ROOT} is empty -- run `git submodule update --init`")
sys.path.insert(0, LCB_ROOT)

# `code_generation` reads its few-shot JSON through a CWD-relative path while
# being imported, so these have to happen from inside the checkout -- and before
# any caller chdirs elsewhere.
_cwd = os.getcwd()
os.chdir(LCB_ROOT)
try:
    from lcb_runner.lm_styles import (  # noqa: E402
        LanguageModel,
        LanguageModelStore,
        LMStyle,
    )
    from lcb_runner.prompts import code_generation  # noqa: E402
    from lcb_runner.runner.main import main  # noqa: E402
    from lcb_runner.runner.parser import get_args  # noqa: E402
    from lcb_runner.runner.scenario_router import build_prompt_benchmark  # noqa: E402
    from lcb_runner.utils.path_utils import get_output_path  # noqa: E402
finally:
    os.chdir(_cwd)

__all__ = [
    "LCB_ROOT",
    "build_prompt_benchmark",
    "get_args",
    "get_output_path",
    "install_prompt_style",
    "main",
    "readable_checkpoint",
]


def readable_checkpoint(params, output_dir):
    """A view of the checkpoint whose tokenizer config this env can read."""
    if params.get("model_path"):
        return params.model_path

    source = os.path.join(
        os.path.expandvars("$RUNS_DIR"), params.train_job_id, params.checkpoint
    )
    config_name = "tokenizer_config.json"
    with open(os.path.join(source, config_name)) as file:
        config = json.load(file)
    if not isinstance(config.get("extra_special_tokens"), list):
        return source  # already readable, e.g. a hub id downloaded by this env

    target = os.path.join(output_dir, "model")
    # rebuilt rather than updated: a stale symlink would quietly point at a
    # previously evaluated checkpoint while the logs name the new one
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target)
    for name in os.listdir(source):
        if name != config_name:
            os.symlink(os.path.join(source, name), os.path.join(target, name))

    config["additional_special_tokens"] = config.pop("extra_special_tokens")
    with open(os.path.join(target, config_name), "w") as file:
        json.dump(config, file, indent=2)
    print(f"rewrote {config_name} for transformers 4.x at {target}")
    return target


def install_prompt_style(model_repr, checkpoint, template):
    """Register the model under `model_repr` and have it use `template`."""
    from datetime import datetime

    code_generation.get_qwen_qwq_question_template_answer = template
    LanguageModelStore[model_repr] = LanguageModel(
        model_repr, model_repr, LMStyle.QwQ, datetime(2023, 1, 1), link=checkpoint
    )
