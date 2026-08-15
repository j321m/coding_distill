# coding_distill

Distilling a small coding model from teacher reasoning traces.

PoC baseline: SFT `Qwen2.5-Coder-1.5B` on [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots),
then iterate on data preprocessing / mid-training before the distill stage and see whether it beats that baseline.

## Layout

```
main.py                        entrypoint -- mrunner params on cluster, JSON file locally
configs/tiny_local.json        smoke-test config (135M model, 32 samples, CPU-runnable)
mrunner_specs/sft_baseline.py  first cluster sweep: LR grid over the baseline
src/coding_distill/
  config.py                    attrs config sections; flat `section.field` params -> objects
  data.py                      codeforces-cots -> chat-templated, prompt-masked examples
  train.py                     TRL SFTTrainer
  evaluation.py                execution-based coding eval (STUB)
scripts/inspect_dataset.py     print schema + a sample of a codeforces-cots subset
```

No Hydra. Config is flat `section.field` keys — the same shape mrunner's params grid uses —
validated into attrs classes by `config_from_params`. Nothing has a default: a run is fully
described by its params, and a typo fails at startup instead of silently training something else.

## Running

Local smoke test:

```bash
pixi run python main.py --config configs/tiny_local.json
```

Cluster:

```bash
mrunner --config ~/.mrunner.yaml --context <your_context> run mrunner_specs/sft_baseline.py
```

## State

Sketch, not yet run. Before the first real launch:

1. `pixi run python scripts/inspect_dataset.py solutions_py` — confirm the column layout and
   fix `_to_messages` in `data.py`. The codeforces-cots subsets do not share a schema.
2. Verify the mrunner API against the fork you install (`create_experiments_helper` kwargs,
   and whether `get_configuration` returns what `main.py` expects). mrunner is not on PyPI.
3. Implement `evaluation.py`. Without execution-based pass@1 there is no baseline to beat —
   this is the piece that decides whether the whole project can measure anything.
4. Decide the eval split. `data.n_eval_samples` is wired into the config but not yet used;
   held-out codeforces problems and a public benchmark measure different things.

Generated code from `evaluation.py` must run sandboxed and without network — it is arbitrary
model output.
