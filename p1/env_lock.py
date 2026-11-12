#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_lock.py

View or update the environment lock runs/env_lock.json (see "Environment lock" in analysis/Pilot_Plan.md).
The training and evaluation scripts check whether the current environment matches the lock before creating the output directory, and exit with an error if not; this script updates the lock after an intentional upgrade.

Usage:
    python env_lock.py --show                          # compare the current environment with the lock
    python env_lock.py --update --reason "<reason>"    # update the lock from the current environment; old and new versions and the reason are appended to env_lock_history.jsonl
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common import ENV_CHECKED_KEYS, ENV_LOCK_DEFAULT, collect_env, env_diff, save_json


def main():
    ap = argparse.ArgumentParser(description="Show or update the environment lock")
    ap.add_argument("--lock", default=str(ENV_LOCK_DEFAULT))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--show", action="store_true")
    g.add_argument("--update", action="store_true")
    ap.add_argument("--reason", default=None, help="Required with --update")
    args = ap.parse_args()

    lock_path = Path(args.lock).expanduser()
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    env = collect_env()

    if args.show:
        print(f"Lock: {lock_path}" + ("" if lock else " (not found)"))
        print(f"{'item':16s} {'locked':40s} {'current':40s}")
        for k in ENV_CHECKED_KEYS:
            old = None if lock is None else lock.get(k)
            mark = "" if lock is not None and old == env[k] else "  <- differs"
            print(f"{k:16s} {str(old):40s} {str(env[k]):40s}{mark}")
        print("Recorded, not checked: " + ", ".join(f"{k}={env[k]}" for k in ("kernel", "git_commit", "git_dirty")))
        return

    if not args.reason:
        raise SystemExit("[Error] --update requires --reason describing the upgrade")
    diff = env_diff(lock, env) if lock else [(k, None, env[k]) for k in ENV_CHECKED_KEYS]
    if lock is not None and not diff:
        print("The current environment matches the lock; nothing to update")
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history = lock_path.parent / "env_lock_history.jsonl"
    with open(history, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time_utc": now, "reason": args.reason,
                            "changes": {k: {"old": a, "new": b} for k, a, b in diff}}, ensure_ascii=False) + "\n")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    save_json({**{k: env[k] for k in ENV_CHECKED_KEYS}, "created_utc": now}, lock_path)
    print(f"Updated {lock_path}; changes appended to {history}:")
    for k, a, b in diff:
        print(f"  {k}: {a} -> {b}")


if __name__ == "__main__":
    main()
