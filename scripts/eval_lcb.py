"""Score a checkpoint on LiveCodeBench.

    pixi run mrun --context helios-eval run configs/eval/lcb.py

LiveCodeBench ships no prompt style matching what this model was fine-tuned on,
so one is installed at runtime: `QwQ` omits the system turn and reframes the
task, `CodeQwenInstruct` is not think-aware, and `DeepSeekR1` emits DeepSeek
tokens absent from Qwen's vocabulary.
"""

import dataclasses
import json
import os
import sys

import tyro
from transformers import AutoTokenizer

from src.config import EvalArgs
from src.lcb import (
    build_prompt_benchmark,
    get_args,
    get_output_path,
    install_prompt_style,
    main,
    readable_checkpoint,
)
from src.lcb_eval_prompts import CODE_FENCE, build_sft_template
from src.provenance import load_git, write_run_metadata
from src.tracking import init_wandb


def report_generation_lengths(model_repr, tokenizer, max_tokens, run):
    """% of samples that hit the token cap, and % that contain a code fence."""
    path = get_output_path(model_repr, get_args())
    if not os.path.exists(path):
        print(f"no generations found to measure at {path}")
        return

    lengths, fenced = [], 0
    with open(path) as file:
        for entry in json.load(file):
            for text in entry.get("output_list") or []:
                lengths.append(len(tokenizer(text)["input_ids"]))
                fenced += bool(CODE_FENCE.search(text))
    if not lengths:
        print(f"no generations found to measure in {path}")
        return

    lengths.sort()
    stats = {
        "lcb/samples": len(lengths),
        "lcb/tokens_p50": lengths[len(lengths) // 2],
        "lcb/tokens_p90": lengths[int(0.9 * (len(lengths) - 1))],
        "lcb/tokens_max": lengths[-1],
        "lcb/pct_hit_cap": 100 * sum(n >= max_tokens for n in lengths) / len(lengths),
        "lcb/pct_with_fence": 100 * fenced / len(lengths),
    }
    print(stats)
    if run is not None:
        run.log(stats)
        run.summary.update(stats)


def lcb_argv(model_repr, checkpoint, params):
    """LiveCodeBench reads its configuration from argv, so build one."""
    return [
        "lcb_runner.runner.main",
        "--model",
        model_repr,
        "--local_model_path",
        checkpoint,
        "--scenario",
        params.scenario,
        "--release_version",
        params.release_version,
        "--start_date",
        params.start_date,
        "--end_date",
        params.end_date,
        # DeepSeek uses n=64 to estimate pass@1, see
        # (Usage Recommendations https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B#usage-recommendations)
        "--n",
        str(params.n),
        # LiveCodeBench defaults to 0.2, tuned for short code answers. DeepSeek's
        # guidance for R1-style distills is 0.5-0.7 (Usage Recommendations https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B#usage-recommendations)
        "--temperature",
        str(params.temperature),
        "--top_p",
        str(params.top_p),
        "--max_tokens",
        str(params.max_tokens),
        "--dtype",
        params.dtype,
        "--tensor_parallel_size",
        str(params.tensor_parallel_size),
        "--evaluate",
    ]


params = tyro.cli(EvalArgs)
model_repr = params.model_name or f"coding-distill-{params.train_job_id}"
# LiveCodeBench writes to relative `output/` and `cache/`, which would land in
# the mrunner code snapshot on $HOME. Put them next to the checkpoint instead.
output_dir = os.path.join(
    os.path.expandvars("$RUNS_DIR"), params.train_job_id, "lcb", model_repr
)
os.makedirs(output_dir, exist_ok=True)

checkpoint = readable_checkpoint(params, output_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

metadata = write_run_metadata(output_dir, {**dataclasses.asdict(params), **load_git()})
run = init_wandb(metadata) if params.report_to == "wandb" else None
os.chdir(output_dir)

install_prompt_style(model_repr, checkpoint, build_sft_template(tokenizer))

sys.argv = lcb_argv(model_repr, checkpoint, params)
print(
    f"checkpoint: {checkpoint}\noutput_dir: {output_dir}\nargv: {' '.join(sys.argv[1:])}"
)

# The date window's yield is not predictable from the flags: report it before
# generation so a run that selected 2 problems (or 200) can be killed early.
selected, _ = build_prompt_benchmark(get_args())
print(f"problems selected by {params.start_date}..{params.end_date}: {len(selected)}")

main()
report_generation_lengths(model_repr, tokenizer, params.max_tokens, run)
