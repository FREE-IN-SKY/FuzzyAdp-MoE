#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
devkit.py: Shared utilities for the board scripts: environment collection, image list and preprocessing, Jetson power reading, logging with fsync.
Depends only on torch, torchvision, numpy, pillow. Any read failure returns None and continues, so a measurement is not interrupted by environment collection.
"""
import csv
import glob
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------
# Files and logging
# ----------------------------------------------------------------------
def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip().replace("\x00", "")
    except (OSError, TypeError):  # TypeError: read() gets None when sysfs returns EAGAIN (Orin Nano cv0 ~ cv2-thermal)
        return None


def run_cmd(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip() or (r.stderr or "").strip()
    except (OSError, subprocess.SubprocessError):
        return None


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    """Each log line is flushed + fsynced immediately; a power loss loses at most one line."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg, echo=True):
        if echo:
            print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
            os.fsync(f.fileno())


def write_json_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def _pkg_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def cpu_info():
    info = {"cpu_count": os.cpu_count(), "online": read_text("/sys/devices/system/cpu/online")}
    pol = []
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
        pol.append({"policy": os.path.basename(p), "cpus": read_text(p + "/affected_cpus"),
                    "governor": read_text(p + "/scaling_governor"),
                    "cur_khz": read_text(p + "/scaling_cur_freq"), "max_khz": read_text(p + "/scaling_max_freq")})
    info["cpufreq"] = pol
    model = None
    txt = read_text("/proc/cpuinfo") or ""
    m = re.search(r"^(?:model name|Model|Hardware)\s*:\s*(.+)$", txt, re.M)
    if m:
        model = m.group(1).strip()
    info["cpu_model"] = model
    return info


def thermal_c():
    out = {}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t, ty = read_text(z + "/temp"), read_text(z + "/type")
        if t and t.lstrip("-").isdigit():
            out[ty or os.path.basename(z)] = int(t) / 1000.0
    return out


def is_jetson():
    return os.path.exists("/etc/nv_tegra_release") or "tegra" in (read_text("/proc/device-tree/compatible") or "").lower()


def throttled():
    """Integer value of vcgencmd get_throttled; None on non-Raspberry Pi boards or on read failure.
    Bit definitions (Raspberry Pi documentation): 0 undervoltage, 1 frequency capped, 2 throttled, 3 soft temperature limit; +16 means "has occurred since boot", cleared only on reboot."""
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", run_cmd(["vcgencmd", "get_throttled"]) or "")
    return int(m.group(1), 16) if m else None


def ext5v_v():
    """5 V input voltage (V) read from the Pi 5 PMIC; other boards return None."""
    m = re.search(r"EXT5V_V volt\(\d+\)=([0-9.]+)V", run_cmd(["vcgencmd", "pmic_read_adc", "EXT5V_V"]) or "")
    return float(m.group(1)) if m else None


def power_snapshot():
    """Power supply status snapshot. ok: throttled is 0 (no undervoltage and no throttling since boot); None if unreadable (Jetson etc.), no judgment is made."""
    t = throttled()
    return {"throttled": None if t is None else hex(t), "ok": None if t is None else t == 0, "ext5v_v": ext5v_v()}


def board_env(torch_threads=None):
    """Collect board information; any item that fails is filled with None."""
    import torch
    env = {
        "timestamp": now_str(),
        "hostname": platform.node(),
        "device_tree_model": read_text("/proc/device-tree/model"),
        "os": (re.search(r'PRETTY_NAME="?([^"\n]+)', read_text("/etc/os-release") or "") or [None, None])[1],
        "kernel": platform.release(), "machine": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": _pkg_version("torchvision"), "timm": _pkg_version("timm"),
        "numpy": _pkg_version("numpy"), "pillow": _pkg_version("PIL"),
        "torch_threads": torch_threads if torch_threads is not None else torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu": cpu_info(),
        "mem_total_kb": (re.search(r"MemTotal:\s+(\d+)", read_text("/proc/meminfo") or "") or [None, None])[1],
        "thermal_c": thermal_c(),
        "power": power_snapshot(),
        "jetson": None,
    }
    if is_jetson():
        env["jetson"] = {"nv_tegra_release": read_text("/etc/nv_tegra_release"),
                         "nvpmodel_q": run_cmd(["nvpmodel", "-q"]) or run_cmd(["sudo", "-n", "nvpmodel", "-q"]),
                         "jetson_clocks_show": run_cmd(["sudo", "-n", "jetson_clocks", "--show"], timeout=10)}
    return env


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def read_list(list_csv, limit=0):
    """List from make_subset.py: path,class_name,class_idx,date. Returns [(path, label, date)]."""
    rows = []
    with open(list_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["path"], int(r["class_idx"]), r["date"]))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"[Error] {list_csv} is empty")
    return rows


def eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(root, rel, tf):
    from PIL import Image
    with Image.open(Path(root) / rel) as img:
        return tf(img.convert("RGB")).unsqueeze(0)


def model_forward(model, x):
    """The TorchScript MoE returns (logits, topk_idx, w), the single model returns logits. Returns uniformly (logits, topk_idx or None, w or None)."""
    y = model(x)
    if isinstance(y, (tuple, list)):
        return y[0], y[1], y[2]
    return y, None, None


# ----------------------------------------------------------------------
# Jetson power (ported from the old scripts power_consumption_test_Jetson_*_cpu.py, 2026-04)
# ----------------------------------------------------------------------
TOTAL_POWER_RAILS = ("VDD_IN", "POM_5V_IN")
SYSFS_POWER_PATTERNS = (
    "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input",
    "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input",
    "/sys/bus/i2c/devices/*/iio:device*/in_power*_input",
    "/sys/bus/iio/devices/iio:device*/in_power*_input",
    "/sys/devices/platform/*/iio:device*/in_power*_input",
    "/sys/class/hwmon/hwmon*/power*_input",
)


def _label_for(path):
    for pat in (r"(in_power\d+)_input$", r"(power\d+)_input$"):
        m = re.search(pat, path)
        if m:
            pre = m.group(1)
            for cand in (path.replace(f"{pre}_input", f"{pre}_label"), path.replace(f"{pre}_input", f"{pre}_name")):
                t = read_text(cand)
                if t:
                    return re.sub(r"[^A-Z0-9_]+", "_", t.upper()).strip("_")
    return ""


def read_sysfs_power_mw():
    """Returns (mW, source) or (None, ''). The unit of sysfs power*_input is usually uW."""
    rails, unlabeled = {}, []
    for pat in SYSFS_POWER_PATTERNS:
        for path in glob.glob(pat):
            t = read_text(path)
            if not t:
                continue
            try:
                v = int(float(t))
            except ValueError:
                continue
            if v <= 0:
                continue
            mw = int(round(v / 1000.0))
            lab = _label_for(path)
            if lab:
                rails[lab] = mw
            else:
                unlabeled.append(mw)
    for r in TOTAL_POWER_RAILS:
        if r in rails:
            return rails[r], f"sysfs:{r}"
    if rails:
        return sum(rails.values()), "sysfs:sum(" + ",".join(sorted(rails)) + ")"
    if unlabeled:
        return max(unlabeled), "sysfs:max-unlabeled"
    return None, ""


def parse_tegrastats_power_mw(line):
    rails = {}
    for name, cur, _avg in re.findall(r"\b([A-Z][A-Z0-9_]*)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:mW)?/([0-9]+(?:\.[0-9]+)?)\s*(?:mW)?\b", line or ""):
        if name.startswith(("VDD", "POM")):
            rails[name] = int(float(cur))
    for r in TOTAL_POWER_RAILS:
        if r in rails:
            return rails[r]
    return sum(rails.values()) if rails else None


def read_tegrastats_power_mw():
    for cmd in (["timeout", "3", "tegrastats", "--interval", "500"], ["sudo", "-n", "timeout", "3", "tegrastats", "--interval", "500"]):
        out = run_cmd(cmd, timeout=6)
        if out:
            for line in out.splitlines():
                v = parse_tegrastats_power_mw(line)
                if v is not None:
                    return v, ("sudo " if cmd[0] == "sudo" else "") + "tegrastats"
    return None, ""


def read_jetson_power_mw():
    v, src = read_sysfs_power_mw()
    if v is not None:
        return v, src
    return read_tegrastats_power_mw()


def perf_now():
    return time.perf_counter()
