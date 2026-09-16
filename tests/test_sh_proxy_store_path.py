"""The token store's path is one string shared by two processes — when they disagreed, every episode 401'd."""

from __future__ import annotations

import re
from pathlib import Path


def test_the_restart_script_and_the_driver_name_the_same_store():
    """`proxy-restart.sh` starts the reader; `episode.py --tokens` is the writer. A stale `tokens.json` in the
    script against a `tokens/` directory in the driver cost a full 32-episode run."""
    script = (Path(__file__).parent.parent / "sh/validator/net/proxy-restart.sh").read_text()
    flag = re.search(r'--tokens\s+"([^"]+)"', script)
    assert flag, "proxy-restart.sh no longer passes --tokens"
    assert not flag.group(1).endswith(".json"), (
        f"--tokens {flag.group(1)} names a file; the store is a directory (one file per token)"
    )
