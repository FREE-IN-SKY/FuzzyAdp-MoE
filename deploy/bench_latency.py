#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_latency.py: CPU-only TorchScript latency measurement + consistency check of on-board predictions against experiment-machine predictions.

    python bench_latency.py --model models/moe_fixed_s3_top2.pt --root <image root dir> --list val.csv \
        --board rpi4b --threads 4 [--reference <experiment-machine predictions.csv>] [--limit N] --out results/rpi4b/latency_moe_fixed_s3_top2.json

Protocol (Pilot_Study.md section 14): batch 1, num_workers 0, 5 random-input warmups, per-image timing. Two times are recorded:
    model_ms   wall-clock time of model(x) only (perf_counter)
    e2e_ms     time including image reading, preprocessing and model(x)
Reports mean, median, p90, standard deviation; throughput = N / total loop time. Also outputs predictions.csv (path, label, prediction, experts, weights) and compares it with
--reference by path: pred agreement rate (MoE also has a top-k expert set agreement rate). If agreement < 0.999, reference_ok=false is set in the JSON and the script exits with code 3.
On Raspberry Pi, vcgencmd get_throttled is read once before and once after the measurement; if it is nonzero at the end (undervoltage or throttling has occurred since boot), power.ok=false in the JSON and that latency run is discarded.
"""
import argparse
import csv
import statistics as st
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
    ap.add_argument("--board", required=True, help="Board tag used in the output, e.g. rpi4b, rpi5, xavier_nx, orin_nano")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = leave default)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N images (smoke test)")
    ap.add_argument("--reference", default=None, help="predictions.csv from p1/evaluate.py (CPU fp32) for the same checkpoint and list")
    ap.add_argument("--no-ref-check", action="store_true", help="Report the agreement but never fail on it (top-1 deployment variant)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    env = devkit.board_env(torch.get_num_threads())
    if env["power"]["ok"] is False:
        print(f"[Warn] throttled={env['power']['throttled']} before measuring: under-voltage/throttling has occurred since boot. "
              "This run will be marked power_ok=false; fix the power supply and reboot before measuring.", flush=True)
    rows = devkit.read_list(args.list, args.limit)
    tf = devkit.eval_transform()
    model = torch.jit.load(args.model, map_location="cpu").eval()

    with torch.no_grad():
        for _ in range(args.warmup):
            devkit.model_forward(model, torch.randn(1, 3, devkit.INPUT_SIZE, devkit.INPUT_SIZE))

    model_ms, e2e_ms, preds = [], [], []
    temps_start = devkit.thermal_c()
    t_loop0 = devkit.perf_now()
    with torch.no_grad():
        for i, (rel, label, _date) in enumerate(rows):
            t0 = devkit.perf_now()
            x = devkit.load_image(args.root, rel, tf)
            t1 = devkit.perf_now()
            logits, topk, w = devkit.model_forward(model, x)
            t2 = devkit.perf_now()
            model_ms.append((t2 - t1) * 1000)
            e2e_ms.append((t2 - t0) * 1000)
            rec = {"path": rel, "label": label, "pred": int(logits.argmax(1).item())}
            if topk is not None:
                rec["experts"] = [int(v) for v in topk[0].tolist()]
                rec["w"] = [round(float(v), 5) for v in w[0].tolist()]
            preds.append(rec)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(rows)}  model {st.mean(model_ms[-500:]):.1f} ms  e2e {st.mean(e2e_ms[-500:]):.1f} ms", flush=True)
    loop_s = devkit.perf_now() - t_loop0
    temps_end = devkit.thermal_c()
    power_end = devkit.power_snapshot()

    def summary(v):
        v2 = sorted(v)
        return {"mean": st.mean(v), "median": st.median(v), "p90": v2[int(0.9 * (len(v2) - 1))],
                "p99": v2[int(0.99 * (len(v2) - 1))], "std": st.pstdev(v), "min": v2[0], "max": v2[-1]}

    acc = sum(p["pred"] == p["label"] for p in preds) / len(preds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_path = out.with_name(out.stem + "_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "label", "pred", "experts", "w"])
        for p in preds:
            wr.writerow([p["path"], p["label"], p["pred"], " ".join(map(str, p.get("experts", []))), " ".join(map(str, p.get("w", [])))])

    ref = None
    if args.reference:
        ref_rows = {}
        with open(args.reference, newline="") as f:
            for r in csv.DictReader(f):
                ref_rows[r["path"]] = r
        matched = [p for p in preds if p["path"] in ref_rows]
        agree = sum(int(ref_rows[p["path"]]["pred"]) == p["pred"] for p in matched)
        ref = {"file": args.reference, "n_matched": len(matched), "pred_agreement": agree / len(matched) if matched else None}
        if matched and "experts" in matched[0] and "expert_1" in ref_rows[matched[0]["path"]]:
            same_set = sum(set(p["experts"]) == {int(ref_rows[p["path"]]["expert_1"]), int(ref_rows[p["path"]]["expert_2"])} for p in matched)
            same_top1 = sum(p["experts"][0] == int(ref_rows[p["path"]]["expert_1"]) for p in matched)
            ref["expert_set_agreement"] = same_set / len(matched)
            ref["top1_expert_agreement"] = same_top1 / len(matched)
        is_top1 = args.no_ref_check or Path(args.model).stem.endswith("top1")
        ref["checked"] = not is_top1
        ref["ok"] = True if is_top1 else bool(ref["pred_agreement"] is not None and ref["pred_agreement"] >= 0.999)

    result = {
        "board": args.board, "model": str(args.model), "list": str(args.list), "n": len(rows), "batch_size": 1,
        "warmup": args.warmup, "threads": torch.get_num_threads(),
        "model_ms": summary(model_ms), "e2e_ms": summary(e2e_ms),
        "loop_seconds": loop_s, "throughput_img_per_s": len(rows) / loop_s, "e2e_ms_from_loop": loop_s / len(rows) * 1000,
        "accuracy": acc, "reference": ref, "thermal_start_c": temps_start, "thermal_end_c": temps_end,
        "power": {"start": env["power"], "end": power_end, "ok": power_end["ok"]},
        "predictions_csv": str(pred_path), "env": env, "finished": devkit.now_str(),
    }
    devkit.write_json_atomic(result, out)
    print(f"{args.board} {Path(args.model).name}: n={len(rows)} model {result['model_ms']['mean']:.2f} ms (median {result['model_ms']['median']:.2f}, "
          f"p90 {result['model_ms']['p90']:.2f}), e2e {result['e2e_ms']['mean']:.2f} ms, {result['throughput_img_per_s']:.2f} img/s, acc {acc:.4f}"
          + (f", reference agreement {ref['pred_agreement']:.4f} ({'not checked (top-1 variant)' if not ref['checked'] else 'OK' if ref['ok'] else 'FAILED'})" if ref else "")
          + ("" if power_end["ok"] is None else f", power {'OK' if power_end['ok'] else 'FAILED'} (throttled={power_end['throttled']})"))
    if power_end["ok"] is False:
        print("[Warn] under-voltage/throttling flagged (power_ok=false): this latency is not valid; fix the supply, reboot and re-run.", flush=True)
    print(f"saved {out}")
    if ref and not ref["ok"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
