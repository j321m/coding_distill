"""mrunner entry point. Turns mrunner's flat config into CLI flags for main.py,
so mrunner never sees tyro and tyro never sees mrunner's `--config`.

Keys are dotted (`train.learning_rate`) -- tyro's syntax for nested dataclasses.
"""

import subprocess
import sys

from mrunner.helpers.client_helper import get_configuration

if __name__ == "__main__":
    cfg = get_configuration()
    cfg.pop("experiment_id", None)  # added by mrunner, not ours

    flags = []
    for key, value in cfg.items():
        flags.append(f"--{key}")
        if isinstance(value, list):
            flags.extend(str(item) for item in value)
        else:
            # bools too: with no default, tyro parses them as `--x True`
            flags.append(str(value))

    cmd = [sys.executable, "-m", "main"] + flags
    print("Running command:", cmd, flush=True)
    subprocess.run(cmd, check=True)
