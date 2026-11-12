#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_rpi.py: Raspberry Pi power-bank discharge method. The board is powered by the power bank and runs batch-1 inference continuously until power is cut; the heartbeat log is written to disk every 10 seconds (fsync).
After power-off, read the log on the laptop with summarize_device.py: T = cumulative seconds of the last heartbeat, N = cumulative image count,
    energy per sample E_sample = E_out / N,  average power P_avg = E_out / T,
E_out is the usable energy on the power bank's 5 V output side: if --bank-rated-out-wh is given, the rated value is used; otherwise cell energy × efficiency range (--eff-low, --eff-high) gives lower and upper bounds.
Default range [0.70, 0.85]: the lower bound is the usable fraction given by the vendor manual (Intenso F10000 manual (f10000_bda_alle.pdf, intenso.de): "about 30 % of the total capacity is used up just through the charging process ... loss of power from circuit warmth and voltage conversion", i.e. about 70% usable);
the upper bound 0.85 is a common upper limit measured on similar 5 V boost power banks, not vendor data. The vendor gives no rated capacity on the 5 V output side, so --bank-rated-out-wh defaults to empty.
On the same board, the ratio of E_sample between two models = N_single / N_moe, independent of the E_out assumption.
Each heartbeat records vcgencmd get_throttled (Pi 5 also records the EXT5V_V input voltage). The voltage drops as the power bank nears empty, so undervoltage at the end is normal discharge;
summarize_device.py judges by the first time throttled is nonzero: within the last 5% of the total duration the run is valid, earlier and the run is discarded.

    python power_rpi.py --model models/moe_fixed_s3_top2.pt --root <image root dir> --list val.csv --board rpi4b --threads 4 \
        --bank-cell-wh 37.0 --eff-low 0.70 --eff-high 0.85 --run-id 1 --log results/rpi4b/power_moe_fixed_s3_top2_run1.log

Requirements: the board starts from a fully charged power bank, and the power bank is connected before the script starts; the heartbeat log is written to the SD card and is still readable after power-off. --drop-caches needs passwordless sudo
(the old script cleared the page cache once per round; off by default; if enabled, note it in the protocol). --max-seconds is only for smoke tests.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devkit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--heartbeat", type=float, default=10.0)
    ap.add_argument("--bank-name", default="Intenso F10000")
    ap.add_argument("--bank-cell-wh", type=float, default=37.0, help="Cell energy printed on the bank (3.7 V x 10000 mAh)")
    ap.add_argument("--bank-rated-out-wh", type=float, default=None, help="Rated 5 V output energy if the maker states it")
    ap.add_argument("--eff-low", type=float, default=0.70, help="Usable fraction of cell energy, lower bound; 0.70 = maker's manual (about 30 %% lost in conversion)")
    ap.add_argument("--eff-high", type=float, default=0.85, help="Upper bound; typical measured ceiling for 5 V boost power banks, not a maker figure")
    ap.add_argument("--run-id", type=int, default=1)
    ap.add_argument("--drop-caches", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0, help="Stop after this many seconds (smoke test only)")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    log = devkit.Logger(args.log)
    status_path = Path(args.log).with_suffix(".status.json")
    env = devkit.board_env(torch.get_num_threads())
    header = {"kind": "rpi_power_bank_discharge", "board": args.board, "model": str(args.model), "list": str(args.list),
              "run_id": args.run_id, "threads": torch.get_num_threads(), "heartbeat_s": args.heartbeat, "drop_caches": args.drop_caches,
              "bank": {"name": args.bank_name, "cell_wh": args.bank_cell_wh, "rated_out_wh": args.bank_rated_out_wh,
                       "eff_low": args.eff_low, "eff_high": args.eff_high},
              "start": devkit.now_str(), "env": env}
    log("HEADER " + json.dumps(header, ensure_ascii=False))
    if env["power"]["ok"] is False:
        log(f"[{devkit.now_str()}] [Warn] throttled={env['power']['throttled']} at start: under-voltage/throttling since boot; this run will be invalid. Reboot on the bank and restart.")

    rows = devkit.read_list(args.list)
    tf = devkit.eval_transform()
    model = torch.jit.load(args.model, map_location="cpu").eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            devkit.model_forward(model, torch.randn(1, 3, devkit.INPUT_SIZE, devkit.INPUT_SIZE))
    log(f"[{devkit.now_str()}] warmup done, {len(rows)} images per round, starting continuous inference")

    t_start = time.time()
    last_hb = t_start
    n_images, n_rounds, infer_s = 0, 0, 0.0
    with torch.no_grad():
        while True:
            n_rounds += 1
            if args.drop_caches:
                subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=False)
            for rel, _label, _d in rows:
                x = devkit.load_image(args.root, rel, tf)
                t0 = time.perf_counter()
                devkit.model_forward(model, x)
                infer_s += time.perf_counter() - t0
                n_images += 1
                now = time.time()
                if now - last_hb >= args.heartbeat:
                    elapsed = now - t_start
                    rec = {"elapsed_s": round(elapsed, 1), "images": n_images, "round": n_rounds,
                           "model_ms_mean": round(infer_s / n_images * 1000, 2), "e2e_ms_mean": round(elapsed / n_images * 1000, 2),
                           "thermal_c": devkit.thermal_c()}
                    pw = devkit.power_snapshot()
                    rec["throttled"], rec["ext5v_v"] = pw["throttled"], pw["ext5v_v"]
                    log(f"[{devkit.now_str()}] heartbeat {json.dumps(rec)}", echo=(n_images % 50 == 0 or elapsed < 60))
                    devkit.write_json_atomic({"header": header, "last": rec}, status_path)
                    last_hb = now
                if args.max_seconds and now - t_start >= args.max_seconds:
                    log(f"[{devkit.now_str()}] max-seconds reached (smoke test); images={n_images} elapsed={now - t_start:.1f}s")
                    return


if __name__ == "__main__":
    main()
