#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""board_env.py: Print and save the board environment (JSON). Usage: python board_env.py [--out <path>] [--power] (--power reads Jetson power once)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devkit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--power", action="store_true")
    args = ap.parse_args()
    env = devkit.board_env()
    if args.power:
        mw, src = devkit.read_jetson_power_mw()
        env["power_probe"] = {"mw": mw, "source": src}
    print(json.dumps(env, indent=2, ensure_ascii=False))
    if args.out:
        devkit.write_json_atomic(env, args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
