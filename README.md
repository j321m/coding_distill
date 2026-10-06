# coding_distill

Offline SFT of a code model on pre-generated chain-of-thought traces.

## Run

```bash
pixi install

# local CPU smoke test (tiny model, a few steps)
pixi run debug

# once per cluster, and after changing pixi.toml: build the remote env
pixi run update-pixi --context entropy

# submit to the cluster
pixi run mrun --context entropy run configs/toy.py
```

On submit mrunner prints a `ssh ... tail -f` line for the job log, and the spec
is copied to `configs/cemetery/<run name>.py`.

## How it fits together

```
configs/<spec>.py  ->  mrunner  ->  mrunner_run.py  ->  main.py (tyro)  ->  src/
```

- **Spec** (`configs/*.py`): one experiment = a flat `base_config` with dotted
  keys (`train.learning_rate`) plus an optional `params_grid` (cartesian
  product, one SLURM array task per combination).
- **mrunner** uploads a snapshot of the repo to the cluster (`clusters.yaml`
  holds the context: partition, GPUs, `prolog_cmd`) and submits it with sbatch.
- **`mrunner_run.py`** turns mrunner's config into CLI flags and calls
  `python -m main`.
- **`main.py`** parses the flags with tyro into `src/config.py:Args`, then
  builds model, data and trains with the HF `Trainer`. Config fields have no
  defaults: a key missing from the spec fails at startup.
- **Environment**: the pixi env is not shipped with the code (home has a file
  quota); it lives under `$PIXI_HOME` on the cluster storage, is built by
  `update_pixi.py` and activated by `prolog_cmd`.
- **Checkpoints** go to `train.output_dir/<SLURM_JOB_ID>` (`$USER` expanded at
  runtime).

## New experiment

Copy `configs/toy.py`, change `experiment_name` and the values, submit it.
Adding a new setting means a field in `src/config.py` and a key in every spec.
