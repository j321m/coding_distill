"""wandb run setup."""

import os


def init_wandb(metadata):
    """Start the run before training, so its config survives a crash in loading.

    HF's `WandbCallback` reuses a live `wandb.run` instead of creating its own at
    `on_train_begin`, so `report_to="wandb"` keeps working and picks this up.

    `wandb.init` takes no entity/project: both come from `WANDB_ENTITY` /
    `WANDB_PROJECT` in `[activation.env]` (pixi.toml), which reach the job
    through the `pixi shell-hook` in prolog_cmd. One definition for local,
    and remote machines, so they cannot disagree; to point a run elsewhere, change
    them there or export them in the shell before submitting.
    """
    import wandb  # ~1s to import; local runs never pay it

    run = wandb.init(
        config={key: metadata[key] for key in ("params", "job", "git", "versions")},
    )

    # Plot train/* against tokens consumed rather than wandb's _step, which only
    # counts log() calls. Tokens is the axis on which two runs with different
    # batch sizes or sequence lengths are comparable -- and once §9 makes
    # sequence lengths ragged, step number stops meaning much at all. Must happen
    # before the first log(); a metric already logged cannot be re-axed without
    # re-running. `eval/*` gets the same treatment when §11 adds it.
    wandb.define_metric("train/num_input_tokens_seen")
    wandb.define_metric("train/*", step_metric="train/num_input_tokens_seen")

    output_dir = metadata["output_dir"]
    for name in ("args.json", metadata["git"]["patch"]):
        if name and os.path.exists(os.path.join(output_dir, name)):
            # base_path: these live under $RUNS_DIR, outside the wandb run dir.
            # policy="now": written once at startup, never touched again.
            run.save(os.path.join(output_dir, name), base_path=output_dir, policy="now")
    return run
