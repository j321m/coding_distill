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
        name=os.environ.get("SLURM_JOB_ID"),  # None locally -> wandb names it
        config={key: metadata[key] for key in ("params", "job", "git", "versions")},
    )

    output_dir = metadata["output_dir"]
    for name in ("args.json", metadata["git"]["patch"]):
        if name and os.path.exists(os.path.join(output_dir, name)):
            # base_path: these live under $RUNS_DIR, outside the wandb run dir.
            # policy="now": written once at startup, never touched again.
            run.save(os.path.join(output_dir, name), base_path=output_dir, policy="now")
    return run
