# coding_distill

Distilling a small coding model from teacher reasoning traces.

PoC baseline: SFT `Qwen2.5-Coder-1.5B` on [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots),
then iterate on data preprocessing / mid-training before the distill stage and see whether it beats that baseline.

## Layout

```
main.py                        entrypoint -- mrunner params on cluster, Python config locally
configs/tiny_local.py          smoke-test config (135M model, 32 samples, CPU-runnable)
mrunner_specs/sft_baseline.py  first cluster sweep: LR grid over the baseline
src/coding_distill/
  config.py                    attrs config sections; flat `section.field` params -> objects
  presets.py                   BASE params + override/label/validate helpers
  data.py                      codeforces-cots -> chat-templated, prompt-masked examples
  train.py                     TRL SFTTrainer
  evaluation.py                execution-based coding eval (STUB)
scripts/inspect_dataset.py     print schema + a sample of a codeforces-cots subset
```

No Hydra. Config is flat `section.field` keys — the same shape mrunner's params grid uses —
validated into attrs classes by `config_from_params`. Nothing has a default: a run is fully
described by its params, and a typo fails at startup instead of silently training something else.

Configs are Python. `presets.BASE` is the one full set of params; local configs and mrunner
specs both `override` it, so the two paths can't drift. Values must stay JSON-ish scalars —
mrunner cloudpickles the params dict out to the job, so anything clever won't survive the trip.
`override` rejects a key that isn't already in `BASE`, and specs call `validate_experiments`
to parse every grid point through the real schema before submitting.

## Running

Local smoke test:

```bash
pixi run python main.py --local-config configs/tiny_local.py
```

First grid point of a spec, locally (starts a real wandb run if the spec has a logger):

```bash
pixi run python main.py --ex mrunner_specs/sft_baseline.py
```

Cluster:

```bash
mrunner --config ~/.mrunner.yaml --context <your_context> run mrunner_specs/sft_baseline.py
```

The local flag is `--local-config`, not `--config`: mrunner launches the remote job as
`main.py --config config_$SLURM_ARRAY_TASK_ID`, a cloudpickle file it reads itself.

## State

Sketch, not yet run. Before the first real launch:

1. `pixi run python scripts/inspect_dataset.py solutions_py` — confirm the column layout and
   fix `_to_messages` in `data.py`. The codeforces-cots subsets do not share a schema.
2. Check that `SFTTrainer` honours the `-100` masking in the pre-tokenized dataset rather
   than re-collating over it — otherwise `mask_prompt` is silently a no-op.
3. Implement `evaluation.py`. Without execution-based pass@1 there is no baseline to beat —
   this is the piece that decides whether the whole project can measure anything.
4. Decide the eval split. `data.n_eval_samples` is wired into the config but not yet used;
   held-out codeforces problems and a public benchmark measure different things. Nothing is
   passed as `eval_dataset` yet, so the LR sweep currently has no metric to rank runs by.
5. `data.py` truncates to `max_seq_length`, which drops the solution off the end of long
   traces; and `ds.select(range(n))` takes the first N rows rather than a shuffled sample.
6. No distributed launch — `script="python3 main.py"` is single-process, which won't fit a
   1.5B full finetune at 16k context.

Generated code from `evaluation.py` must run sandboxed and without network — it is arbitrary
model output.
