"""wandb run setup."""

import os


def init_wandb(metadata):
    """Start the run before training, so its config survives a crash in loading.

    HF's `WandbCallback` reuses a live `wandb.run` instead of creating its own at
    `on_train_begin`, so `report_to="wandb"` keeps working and picks this up.

    `wandb.init` takes no entity/project: both come from `WANDB_ENTITY` /
    `WANDB_PROJECT` in `[activation.env]` (pixi.toml), which reach the job
    through the `pixi shell-hook` in prolog_cmd.
    """
    if os.environ.get("RANK", "0") != "0":
        return None

    import wandb

    run = wandb.init(
        config={key: metadata[key] for key in ("params", "job", "git", "versions")},
    )
    wandb.define_metric("train/num_input_tokens_seen")
    wandb.define_metric("train/*", step_metric="train/num_input_tokens_seen")

    output_dir = metadata["output_dir"]
    for name in ("args.json", metadata["git"]["patch"]):
        if name and os.path.exists(os.path.join(output_dir, name)):
            run.save(os.path.join(output_dir, name), base_path=output_dir, policy="now")
    return run
