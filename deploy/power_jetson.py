#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_jetson.py: Jetson CPU-only power. First idle (--idle-seconds), then continuous inference for a fixed duration (--duration-seconds).
Power comes from tegrastats VDD_IN (whole-board input of the on-board INA3221): each phase runs `tegrastats --interval <ms> --logfile <file>` in the background,
and after the phase ends, all VDD_IN current values (X in X/Y) of that phase are averaged. Power is not read in the inference loop, so sampling does not interrupt inference.
(The INA3221 hwmon driver in JetPack 5/6 only provides voltage and current, no power file, so sysfs is not read.) Results:
    P_idle, P_load (mean, standard deviation, sample count), time per sample (model / e2e), energy per sample E = P_load × t_e2e, incremental energy (P_load − P_idle) × t_e2e.

    python power_jetson.py --model models/moe_fixed_s3_top2.pt --root <image root dir> --list val.csv --board orin_nano --threads 6 \
        --idle-seconds 120 --duration-seconds 1200 --out results/orin_nano/power_moe_fixed_s3_top2.json

When tegrastats run as a normal user does not output VDD_IN (Xavier NX, JetPack 5), run the whole command with sudo; this is checked at startup and the script exits if it cannot be read.
Before running, record the power mode (sudo nvpmodel -q; jetson_clocks not enabled); the script saves the output of nvpmodel -q into the results.
"""
import argparse
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devkit  # noqa: E402

# tegrastats whole-board input: "VDD_IN 2945mW/2945mW" (JetPack 5/6) or "VDD_IN 2945/2945" (older versions); take X (current value)
TOTAL_RE = re.compile(r"\b(VDD_IN|POM_5V_IN)\s+([0-9]+)(?:mW)?/([0-9]+)")


def total_power_mw(line):
    m = TOTAL_RE.search(line or "")
    return (int(m.group(2)), m.group(1)) if m else (None, None)


class TegrastatsLogger:
    """Background tegrastats for one phase: start() starts it, stop() ends it and returns that phase's VDD_IN readings (mW) and the power rail name."""

    def __init__(self, path, interval_ms):
        self.path, self.interval_ms, self.proc = Path(path), interval_ms, None

    def start(self):
        self.path.unlink(missing_ok=True)
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms), "--logfile", str(self.path)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return self.read()

    def read(self):
        vals, rails = [], set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                mw, rail = total_power_mw(line)
                if mw is not None:
                    vals.append(mw)
                    rails.add(rail)
        return vals, rails

    def latest(self):
        vals, _ = self.read()
        return vals[-1] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sample", type=float, default=1.0, help="tegrastats interval in seconds")
    ap.add_argument("--idle-seconds", type=float, default=120.0)
    ap.add_argument("--duration-seconds", type=float, default=1200.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    out = Path(args.out)
    log = devkit.Logger(out.with_suffix(".log"))
    interval_ms = int(round(args.sample * 1000))

    # Startup check: tegrastats must output VDD_IN (or POM_5V_IN), otherwise do not measure
    mw, rail = None, None
    for line in (devkit.run_cmd(["timeout", "3", "tegrastats", "--interval", "500"], timeout=6) or "").splitlines():
        mw, rail = total_power_mw(line)
        if mw is not None:
            break
    if mw is None:
        log(f"[{devkit.now_str()}] [Error] tegrastats shows no VDD_IN for this user; rerun the whole command with sudo "
            f"(e.g. sudo $(which python) deploy/power_jetson.py ...). Nothing was measured.")
        sys.exit(4)

    env = devkit.board_env(torch.get_num_threads())
    log(f"[{devkit.now_str()}] board={args.board} model={args.model} threads={torch.get_num_threads()} "
        f"power=tegrastats {rail} every {interval_ms} ms (now {mw} mW) nvpmodel={(env.get('jetson') or {}).get('nvpmodel_q')!r}")

    rows = devkit.read_list(args.list)
    tf = devkit.eval_transform()
    model = torch.jit.load(args.model, map_location="cpu").eval()
    tg_idle = TegrastatsLogger(out.with_name(out.stem + "_tegrastats_idle.log"), interval_ms)
    tg_load = TegrastatsLogger(out.with_name(out.stem + "_tegrastats_load.log"), interval_ms)

    # Phase 1: idle (model loaded, no inference)
    tg_idle.start()
    time.sleep(args.idle_seconds)
    idle, rails = tg_idle.stop()
    log(f"[{devkit.now_str()}] idle done: {len(idle)} readings, mean {st.mean(idle) if idle else 'N/A'} mW")

    # Phase 2: inference (power recording starts only after warmup)
    with torch.no_grad():
        for _ in range(args.warmup):
            devkit.model_forward(model, torch.randn(1, 3, devkit.INPUT_SIZE, devkit.INPUT_SIZE))
    n_images, infer_s = 0, 0.0
    tg_load.start()
    t_start = time.time()
    last = t_start
    with torch.no_grad():
        while time.time() - t_start < args.duration_seconds:
            for rel, _l, _d in rows:
                x = devkit.load_image(args.root, rel, tf)
                t1 = time.perf_counter()
                devkit.model_forward(model, x)
                infer_s += time.perf_counter() - t1
                n_images += 1
                now = time.time()
                if now - last >= 60:  # progress, one line per minute (reads the log tail once, about 1 ms)
                    log(f"[{devkit.now_str()}] load {now - t_start:.0f}s images={n_images} VDD_IN {tg_load.latest()} mW")
                    last = now
                if now - t_start >= args.duration_seconds:
                    break
    loop_s = time.time() - t_start
    load, rails_load = tg_load.stop()
    rails |= rails_load
    e2e_ms = loop_s / n_images * 1000
    model_ms = infer_s / n_images * 1000

    def stats(v):
        return {"mean_mw": st.mean(v), "std_mw": st.pstdev(v) if len(v) > 1 else 0.0, "n": len(v), "min_mw": min(v), "max_mw": max(v)} if v else None

    # Fewer readings than 90% of the expected count means tegrastats dropped lines or exited midway
    short = [f"{k} {len(v)}/{exp:.0f}" for k, v, exp in (("idle", idle, args.idle_seconds / args.sample), ("load", load, loop_s / args.sample))
             if len(v) < 0.9 * exp]
    p_idle, p_load = stats(idle), stats(load)
    result = {
        "kind": "jetson_tegrastats_cpu_only", "board": args.board, "model": str(args.model), "list": str(args.list),
        "threads": torch.get_num_threads(), "sample_s": args.sample, "idle_seconds": args.idle_seconds, "duration_seconds": loop_s,
        "power_source": [f"tegrastats:{r}" for r in sorted(rails)], "power_readings_short": short,
        "p_idle": p_idle, "p_load": p_load,
        "n_images": n_images, "model_ms_mean": model_ms, "e2e_ms_mean": e2e_ms,
        "energy_per_image_mj": (p_load["mean_mw"] * e2e_ms / 1000) if p_load else None,
        "incremental_energy_per_image_mj": ((p_load["mean_mw"] - p_idle["mean_mw"]) * e2e_ms / 1000) if (p_load and p_idle) else None,
        "thermal_end_c": devkit.thermal_c(), "env": env, "finished": devkit.now_str(),
    }
    devkit.write_json_atomic(result, out)
    log(f"[{devkit.now_str()}] done: images={n_images} model {model_ms:.1f} ms e2e {e2e_ms:.1f} ms; "
        f"P_idle={p_idle['mean_mw'] if p_idle else 'N/A'} mW (n={len(idle)}) P_load={p_load['mean_mw'] if p_load else 'N/A'} mW (n={len(load)}) "
        f"E/img={result['energy_per_image_mj']} mJ; saved {out}")
    if short:
        log(f"[Warn] fewer tegrastats readings than expected ({', '.join(short)}); check the *_tegrastats_*.log files")
    if not p_load or not p_idle:
        log("[Warn] no power readings in a phase; this run is invalid")
        sys.exit(4)


if __name__ == "__main__":
    main()
