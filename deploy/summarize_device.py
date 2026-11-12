#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_device.py: On the laptop, summarize the measurements under deploy/results/<board>/ and output deploy/results/summary.json and summary.md.

Reads three kinds of files:
    latency_*.json        bench_latency.py results (model time, end-to-end time, throughput, accuracy, agreement rate with experiment-machine predictions)
    power_*_run*.log      power_rpi.py heartbeat logs (discharge method): T = elapsed_s of the last heartbeat (+ a truncation correction of half a heartbeat interval), N = images
                          E_out: rated output energy, or cell energy × [eff_low, eff_high]; E_sample = E_out / N; P_avg = E_out / T
                          the efficiency range comes from the log header (power_rpi.py default [0.70, 0.85]; 0.70 is from the Intenso F10000 manual (f10000_bda_alle.pdf, intenso.de): "about 30 % of the total capacity is used up just through the charging process ... loss of power from circuit warmth and voltage conversion"; 0.85 is a common upper limit measured on similar products)
    power_*.json          power_jetson.py results (tegrastats / INA3221)
Power supply: power.ok of the latency results (whether vcgencmd get_throttled is 0 at the end of the measurement) is listed in the table; in a discharge log, if the first nonzero throttled is earlier than
the last --tail-tol (default 5%) of the total duration, or is already nonzero at the start, that run is discarded and not counted in the mean or the energy ratio.
Multiple discharges of the same model on the same board report the mean and range (R2-C3). The MoE-to-single ratio uses only the ratio of N (independent of the E_out assumption).

    python deploy/summarize_device.py [--results deploy/results]
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def model_tag(path):
    """models/moe_fixed_s3_top2.pt -> moe_fixed_s3_top2"""
    return Path(path).stem


def parse_rpi_log(path, tail_tol):
    header, last, n_hb, first_bad = None, None, 0, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("HEADER "):
                header = json.loads(line[7:])
            elif "heartbeat {" in line:
                n_hb += 1
                last = json.loads(line[line.index("{"):])
                if first_bad is None and last.get("throttled") not in (None, "0x0"):
                    first_bad = last
    if not header or not last:
        return None
    bank = header["bank"]
    hb = header.get("heartbeat_s", 10.0)
    T = last["elapsed_s"] + hb / 2  # power loss happens after the last heartbeat, at most one interval later; take half of it as the expected correction
    N = last["images"] * (T / last["elapsed_s"])  # extrapolate the image count by the same proportion
    if bank.get("rated_out_wh"):
        e_out = {"nominal": bank["rated_out_wh"]}
    else:
        e_out = {"low": bank["cell_wh"] * bank["eff_low"], "high": bank["cell_wh"] * bank["eff_high"]}
    # Power check: already abnormal at the start -> discard; first nonzero throttled within the last tail_tol -> normal voltage drop at the end, valid; earlier -> discard; field missing in the log -> None
    start = ((header.get("env") or {}).get("power") or {}).get("ok")
    if start is None and "throttled" not in last:
        power = {"ok": None, "first_bad_frac": None, "first_bad_throttled": None}
    else:
        frac = first_bad["elapsed_s"] / T if first_bad else None
        ok = start is not False and (frac is None or frac >= 1 - tail_tol)
        power = {"ok": ok, "start_ok": start, "first_bad_frac": frac, "first_bad_throttled": first_bad["throttled"] if first_bad else None}
    res = {"log": path, "board": header["board"], "model": model_tag(header["model"]), "run_id": header.get("run_id"),
           "threads": header.get("threads"), "heartbeats": n_hb, "elapsed_s": T, "images": N,
           "model_ms_mean": last.get("model_ms_mean"), "e2e_ms_mean": last.get("e2e_ms_mean"), "e_out_wh": e_out,
           "energy_per_image_j": {k: v * 3600 / N for k, v in e_out.items()}, "p_avg_w": {k: v * 3600 / T for k, v in e_out.items()},
           "bank": bank, "power": power}
    return res


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--tail-tol", type=float, default=0.05, help="Discharge run stays valid if the first throttled flag falls in the last this fraction of its duration")
    args = ap.parse_args()
    root = Path(args.results)
    lat, rpi, jet = [], [], []
    for p in sorted(glob.glob(str(root / "*" / "latency_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        d["model_tag"] = model_tag(d["model"])
        d["_file"] = os.path.relpath(p, root)
        lat.append(d)
    for p in sorted(glob.glob(str(root / "*" / "power_*run*.log"))):
        r = parse_rpi_log(p, args.tail_tol)
        if r:
            r["_file"] = os.path.relpath(p, root)
            rpi.append(r)
    rpi_bad = [r for r in rpi if r["power"]["ok"] is False]
    rpi = [r for r in rpi if r["power"]["ok"] is not False]
    for p in sorted(glob.glob(str(root / "*" / "power_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("kind") != "jetson_tegrastats_cpu_only":
            continue
        d["model_tag"] = model_tag(d["model"])
        d["_file"] = os.path.relpath(p, root)
        jet.append(d)

    md = ["# Device measurement summary (generated, do not edit by hand; source deploy/results/)", ""]
    md += ["## Latency (batch 1, CPU-only TorchScript; ms/image)", "",
           "| Board | Model | N | Threads | Model mean | Median | p90 | End-to-end mean | Throughput (img/s) | On-board accuracy | Agreement with experiment-machine predictions | Power supply |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in lat:
        ref = d.get("reference") or {}
        ag = ref.get("pred_agreement")
        pw = d.get("power") or {}
        pw_txt = "n/a" if pw.get("ok") is None else "clean" if pw["ok"] else f"abnormal {pw['end']['throttled']} (!)"
        md.append(f"| {d['board']} | {d['model_tag']} | {d['n']} | {d['threads']} | {fmt(d['model_ms']['mean'])} | {fmt(d['model_ms']['median'])} | "
                  f"{fmt(d['model_ms']['p90'])} | {fmt(d['e2e_ms']['mean'])} | {fmt(d['throughput_img_per_s'])} | {fmt(d['accuracy'], 4)} | "
                  f"{fmt(ag, 4) if ag is not None else '—'}{'' if ag is None or ref.get('ok') else ' (!)'} | {pw_txt} |")
    # MoE / single latency ratio
    by_board = defaultdict(dict)
    for d in lat:
        by_board[d["board"]][d["model_tag"]] = d
    md += ["", "Latency ratio (same board, model mean): "]
    for b, ms in by_board.items():
        single = next((v for k, v in ms.items() if k.startswith("single")), None)
        if single:
            parts = [f"{k} / single = {v['model_ms']['mean'] / single['model_ms']['mean']:.2f}" for k, v in ms.items() if not k.startswith("single")]
            md.append(f"- {b}: " + "; ".join(parts))

    md += ["", "## Raspberry Pi power-bank discharge method (energy per sample in J; when E_out is unknown, [low, high] efficiency assumptions are given)", "",
           "| Board | Model | Runs | Duration (h) | Images | E/sample (J) | P_avg (W) | Heartbeats | First undervoltage/throttling (fraction of duration) |", "|---|---|---|---|---|---|---|---|---|"]
    grp = defaultdict(list)
    for r in rpi:
        grp[(r["board"], r["model"])].append(r)
    for (b, m), rs in sorted(grp.items()):
        def rng(vals, nd=3):
            return f"{st.mean(vals):.{nd}f}" + (f" ({min(vals):.{nd}f} ~ {max(vals):.{nd}f})" if len(vals) > 1 else "")
        keys = list(rs[0]["energy_per_image_j"].keys())
        e_txt = " / ".join(f"{k}: {rng([r['energy_per_image_j'][k] for r in rs])}" for k in keys)
        p_txt = " / ".join(f"{k}: {rng([r['p_avg_w'][k] for r in rs], 2)}" for k in keys)
        md.append(f"| {b} | {m} | {len(rs)} | {rng([r['elapsed_s'] / 3600 for r in rs], 2)} | {rng([r['images'] for r in rs], 0)} | {e_txt} | {p_txt} | "
                  f"{', '.join(str(r['heartbeats']) for r in rs)} | "
                  f"{', '.join('n/a' if r['power']['ok'] is None else 'none' if r['power']['first_bad_frac'] is None else format(r['power']['first_bad_frac'], '.3f') for r in rs)} |")
    md.append("")
    md.append(f"Discharge validity: if the first nonzero throttled flag falls in the last {args.tail_tol:.0%} of the duration, it is the normal voltage drop of a nearly empty power bank and the run is valid; if it appears earlier or is already nonzero at the start, the run is invalid and excluded from the table above and from the energy ratios.")
    for r in rpi_bad:
        p = r["power"]
        md.append(f"- Invalid: {r['_file']} (ok at start={p['start_ok']}, first abnormal flag {p['first_bad_throttled']} at "
                  f"{'—' if p['first_bad_frac'] is None else format(p['first_bad_frac'], '.3f')})")
    md.append("")
    banks = {(r["bank"].get("name"), r["bank"].get("cell_wh"), r["bank"].get("eff_low"), r["bank"].get("eff_high"), r["bank"].get("rated_out_wh")) for r in rpi}
    for name, cell, lo, hi, rated in sorted(banks, key=str):
        if rated:
            md.append(f"E_out: {name} rated 5 V output {rated} Wh.")
        elif "intenso" in str(name).lower():
            md.append(f"E_out ({name}) = {cell} Wh × [{lo}, {hi}] (nominal 3.7 V cell energy × usable fraction). The lower bound {lo} comes from the manufacturer's manual (Intenso F10000 operating instructions, f10000_bda_alle.pdf: "
                      f"\"about 30 % of the total capacity is used up just through the charging process\"); the upper bound {hi} is a common measured upper limit for similar 5 V boost power banks, not manufacturer data. Absolute energy is reliable only within this interval.")
        else:
            md.append(f"E_out ({name}) = {cell} Wh × [{lo}, {hi}] (nominal 3.7 V cell energy × usable fraction). The manufacturer gives no 5 V output capacity or efficiency for this power bank, so both the lower bound {lo} and the upper bound {hi} are generic assumptions, not manufacturer data; "
                      f"its absolute energy is less well supported than for a power bank with manufacturer data, is for reference only, and is not compared across boards.")
    md += ["", "Energy ratio (same board, independent of the E_out assumption): E_MoE / E_single = N_single / N_MoE"]
    boards = sorted({b for b, _ in grp})
    for b in boards:
        single = [r for (bb, m), rs in grp.items() if bb == b and m.startswith("single") for r in rs]
        if not single:
            continue
        n_single = st.mean(r["images"] for r in single)
        parts = [f"{m}: {n_single / st.mean(r['images'] for r in rs):.2f}" for (bb, m), rs in grp.items() if bb == b and not m.startswith("single")]
        md.append(f"- {b}: " + "; ".join(parts))

    md += ["", "## Jetson (tegrastats / INA3221, CPU-only)", "",
           "| Board | Model | Duration (s) | Images | P_idle (W) | P_load (W) | Samples | e2e ms | E/sample (J) | Incremental E/sample (J) | Source |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in jet:
        pi, pl = d.get("p_idle"), d.get("p_load")
        md.append(f"| {d['board']} | {d['model_tag']} | {fmt(d['duration_seconds'], 0)} | {d['n_images']} | {fmt(pi['mean_mw'] / 1000) if pi else '—'} | "
                  f"{fmt(pl['mean_mw'] / 1000) if pl else '—'} | {pl['n'] if pl else 0} | {fmt(d['e2e_ms_mean'])} | "
                  f"{fmt((d['energy_per_image_mj'] or 0) / 1000, 3) if d.get('energy_per_image_mj') is not None else '—'} | "
                  f"{fmt((d['incremental_energy_per_image_mj'] or 0) / 1000, 3) if d.get('incremental_energy_per_image_mj') is not None else '—'} | "
                  f"{','.join(d.get('power_source') or []) or '—'} |")

    md += ["", "## Environment", ""]
    for p in sorted(glob.glob(str(root / "*" / "env.json"))):
        e = json.load(open(p, encoding="utf-8"))
        md.append(f"- {Path(p).parent.name}: {e.get('device_tree_model') or e.get('cpu', {}).get('cpu_model')}; {e.get('os')}; python {e.get('python')}, "
                  f"torch {e.get('torch')}, torchvision {e.get('torchvision')}, timm {e.get('timm')}; cpus {e.get('cpu', {}).get('cpu_count')}; "
                  f"governor {[c.get('governor') for c in e.get('cpu', {}).get('cpufreq', [])]}"
                  + (f"; nvpmodel: {str((e.get('jetson') or {}).get('nvpmodel_q')).strip()[:80]}" if e.get("jetson") else ""))

    out = {"latency": lat, "rpi_discharge": rpi, "rpi_discharge_invalid": rpi_bad, "jetson": jet}
    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    with open(root / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten {root / 'summary.md'} and summary.json")


if __name__ == "__main__":
    main()
