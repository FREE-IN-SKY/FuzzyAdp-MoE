# Edge-device measurements (deploy/)

The protocol is in Section 14 of `pilot_study/Pilot_Study.md`. This directory holds all scripts for the board side and the host side. Four boards: RPi 4B, RPi 5, Jetson Xavier NX, and Jetson Orin Nano, all CPU-only, TorchScript, batch size 1.

## Host (once)

1. Place the `best.pth` of the five seed-3 runs in their run directories, `runs/final/<run>/best.pth` (the default of both scripts below). The checkpoints are not in the git tree; download `FuzzyAdp-MoE_best_seed3.tar.gz` from the GitHub Release of this repository and extract it at the repository root. If they are elsewhere, set `CKPT_ROOT` to an absolute path in steps 2 and 3.
2. Generate the CPU fp32 reference predictions (about 5 × 4 minutes): `PY=$(which python) bash deploy/make_references.sh` -> `deploy/references/<run>/predictions.csv`. `PY` selects the Python interpreter. The references used in the paper are already in `deploy/references/`.
3. Pack for the boards (about 250 MB, including images, checkpoints, reference predictions, and scripts): `bash deploy/pack_for_boards.sh` -> `deploy_bundle.tgz`. Copy it to the home directory of each board.

## On each board

```bash
mkdir -p ~/edgeai && cd ~/edgeai && tar xzf ~/deploy_bundle.tgz
BOARD=rpi4b bash deploy/setup_board.sh          # BOARD is rpi4b / rpi5 / xavier_nx / orin_nano; checks first, installs missing packages, then saves deploy/results/<BOARD>/env.json
source ~/edgeai-venv/bin/activate
```

Export TorchScript (on the board, to avoid loading problems across torch versions; 1 to 3 minutes each; a self-check confirms on 32 real images that the output matches eager mode):

```bash
R=checkpoints; D="data/images"; L=data/val.csv
python deploy/export_models.py --checkpoint $R/single_augB_lr1e-3_wd0.01_e50_s3/best.pth --out deploy/models/single_s3 --root "$D" --list $L
python deploy/export_models.py --checkpoint $R/moe-fixed_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3/best.pth --out deploy/models/moe_fixed_s3_top2 --root "$D" --list $L
python deploy/export_models.py --checkpoint $R/moe-fixed_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3/best.pth --out deploy/models/moe_fixed_s3_top1 --top-k 1 --root "$D" --list $L
python deploy/export_models.py --checkpoint $R/moe-schedule_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3/best.pth --out deploy/models/moe_schedule_s3_top2 --root "$D" --list $L
python deploy/export_models.py --checkpoint $R/moe-auxfree_R4_lam0_sig0_aw0_augB_lr1e-3_wd1e-4_e50_s3/best.pth --out deploy/models/moe_auxfree_s3_top2 --root "$D" --list $L
python deploy/export_models.py --checkpoint $R/moe-fuzzy_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3/best.pth --out deploy/models/moe_fuzzy_s3_top2 --root "$D" --list $L
```

### Latency (6 runs per board; about 30 minutes each on the RPi 4B, faster on the others)

Threads: 4 on the RPi 4B / RPi 5, 6 on the Xavier NX / Orin Nano (the number of physical cores; the script records the actual value in the result). Mains power, no load other than a display, nothing else running.

```bash
B=rpi4b; T=4; D="data/images"; L=data/val.csv
for m in single_s3 moe_fixed_s3_top2 moe_fixed_s3_top1 moe_schedule_s3_top2 moe_auxfree_s3_top2 moe_fuzzy_s3_top2; do
  case $m in single_s3) ref=references/single_augB_lr1e-3_wd0.01_e50_s3/predictions.csv;;
    moe_fixed_s3_*) ref=references/moe-fixed_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3/predictions.csv;;
    moe_schedule_s3_top2) ref=references/moe-schedule_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3/predictions.csv;;
    moe_auxfree_s3_top2) ref=references/moe-auxfree_R4_lam0_sig0_aw0_augB_lr1e-3_wd1e-4_e50_s3/predictions.csv;;
    moe_fuzzy_s3_top2) ref=references/moe-fuzzy_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3/predictions.csv;; esac
  python deploy/bench_latency.py --model deploy/models/$m.pt --root "$D" --list $L --board $B --threads $T --reference $ref --out deploy/results/$B/latency_$m.json || echo "FAILED $m"
done
```

The predictions of the `top1` variant differ from the reference (top-2 rule) by design, so its agreement column is for reference only and is not a pass criterion (the script skips the check for top1 file names). The agreement of the other 5 must be >= 0.999; otherwise the result is invalid, and the versions and preprocessing must be checked first.

On the Raspberry Pis, each measurement ends with `power OK (throttled=0x0)` or `power FAILED (throttled=...)`. FAILED means that under-voltage or throttling has occurred since boot, and that latency run is invalid: check the power supply (`dmesg -T | grep -i voltage`), reboot, and measure again.

### Energy: Raspberry Pi (power-bank discharge, 4 to 5 hours per run, 2 runs per model)

The board is powered only by the power bank (Intenso F10000, 37 Wh cells), starting fully charged. It runs until the bank shuts off; the log is on the SD card:

```bash
python deploy/power_rpi.py --model deploy/models/single_s3.pt --root "data/images" --list data/val.csv --board rpi4b --threads 4 --run-id 1 --log deploy/results/rpi4b/power_single_s3_run1.log
python deploy/power_rpi.py --model deploy/models/moe_fixed_s3_top2.pt --root "data/images" --list data/val.csv --board rpi4b --threads 4 --run-id 1 --log deploy/results/rpi4b/power_moe_fixed_s3_top2_run1.log
```

For the second run, use `--run-id 2` and a `_run2.log` log name. Suggested order: monolithic run1, MoE run1, monolithic run2, MoE run2 (interleaved, so that aging of the power bank affects both models equally). If Intenso's rated 5 V output capacity is known, add `--bank-rated-out-wh <value>`.

Before a discharge run, reboot on the power bank, run at full load for 10 minutes, and confirm that `dmesg -T | grep -i voltage` is empty and `vcgencmd get_throttled` is `0x0`. The heartbeat log records the throttled flag; in the summary, a first anomaly within the last 5% of the total duration counts as the normal voltage drop at the end, and an earlier one invalidates the run.

### Energy: Jetson (tegrastats reads VDD_IN of the INA3221; 2 + 20 minutes per model)

Record the power mode first: `sudo nvpmodel -q` (the script also records it). The Orin Nano uses the factory default 25W (ID 1); the Xavier NX uses `MODE_15W_6CORE` (ID 2), since the factory default ID 5 enables only 4 cores (Pilot_Study 14.1). Before measuring, confirm that `cat /sys/devices/system/cpu/online` is `0-5`. `jetson_clocks` is not enabled.

Power is the tegrastats VDD_IN (whole-board input), recorded in the background at 1 s intervals for each phase and not read inside the inference loop (Pilot_Study 14.4). Check once on the board before measuring (without sudo):

```bash
cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label; ls /sys/class/hwmon/hwmon*/power*_input 2>/dev/null; timeout 2 tegrastats | head -1
```

If the last line contains `VDD_IN ...mW/...mW`, sudo is not needed (Orin Nano); if it does not (Xavier NX, rail names give Permission denied), run the whole command as root. The script also checks this at start-up and exits if VDD_IN cannot be read. Logs are appended, so delete old `power_*` files before measuring again.

```bash
# Orin Nano (normal user)
for m in single_s3 moe_fixed_s3_top2 moe_fixed_s3_top1; do
  python deploy/power_jetson.py --model deploy/models/$m.pt --root "data/images" --list data/val.csv --board orin_nano --threads 6 --idle-seconds 120 --duration-seconds 1200 --out deploy/results/orin_nano/power_$m.json
done
# Xavier NX (root; one sudo wraps the whole loop so an expired sudo cache cannot block it midway; replace <user>)
sudo bash -c 'for m in single_s3 moe_fixed_s3_top2 moe_fixed_s3_top1; do /home/<user>/edgeai-venv/bin/python deploy/power_jetson.py --model deploy/models/$m.pt --root data/images --list data/val.csv --board xavier_nx --threads 6 --idle-seconds 120 --duration-seconds 1200 --out deploy/results/xavier_nx/power_$m.json; done'
```

The first line should be `power=tegrastats VDD_IN every 1000 ms`, followed after 2 minutes by `idle done: about 120 readings`. Afterwards, check: about 1200 readings in the inference phase (`power_readings_short` in the result is empty); the e2e time of the power run is within 2% of `e2e_ms_from_loop` in the `latency_*.json` of the same model.

### Bringing the results back to the host

Copy the whole `deploy/results/<BOARD>/` directory (JSON, logs, env.json; `*_predictions.csv` is optional) back to the same location in the repository, then run `python deploy/summarize_device.py` to generate `deploy/results/summary.md`.

## Known issues

- **torch version:** all four boards use torch 2.11.0 + torchvision 0.26.0 (pinned by setup_board.sh). Do not install the experiment machine's 2.10.0: on the RPi 4B (Cortex-A72, ARMv8.0), `import torch` fails with `Illegal instruction` because mimalloc in the wheel uses ARMv8.1 LSE instructions (pytorch/pytorch#174344). 2.11.0 fixes this and was tested on the 4B. Install from the CPU index (`--index-url https://download.pytorch.org/whl/cpu`), since from 2.11 on the default aarch64 wheels on PyPI include CUDA.
- **Boards with Python < 3.10 (JetPack 5 ships 3.8):** first `sudo apt install python3.10 python3.10-venv`, then `PYTHON=python3.10 bash deploy/setup_board.sh`. If an older venv exists, delete `~/edgeai-venv` and rebuild it.
- **What was done on the Jetsons (2026-11-11):** neither Jetson uses the system Python; both use a dedicated Miniforge environment (conda-forge Python 3.13.5, the same version as on the Raspberry Pis), and the system `python3` is left unchanged:
  ```bash
  conda create -n edgeai-py313 python=3.13.5 -y      # on the Orin Nano, install Miniforge first: bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
  BOARD=xavier_nx PYTHON=$(conda info --base)/envs/edgeai-py313/bin/python3 bash deploy/setup_board.sh   # BOARD=orin_nano on the Orin Nano
  ```
  Miniforge on the Orin Nano was installed silently with `-b`, without `conda init`, so replace both `conda` calls above with `~/miniforge3/bin/conda`. For measuring, only `source ~/edgeai-venv/bin/activate`; the conda environment does not need to be activated.
- **Keep the bundle in sync with the scripts:** after changing any script in `deploy/`, rerun `pack_for_boards.sh`.
- The Xavier NX CPU log of an earlier version of this work showed 5.1 s per image at 2.4 W, which is clearly abnormal (possibly wrong thread count or power mode). Before running, confirm with `nvpmodel -q` that the 6-core mode (`MODE_15W_6CORE`) is active, and look at the printout of the first 500 images of the benchmark; if it is slower than the RPi 4B, stop and investigate.
- The reference predictions must be generated on the CPU (`make_references.sh` does this); the `predictions.csv` of the experiment machine's GPU uses fp16 autocast and differs from the CPU in about 0.3% of the routing decisions.
- **RPi 5 power supply:** 5 V / 3 A supplies (including the power bank) can only negotiate 3000 mA (`od -An -tx1 /proc/device-tree/chosen/power/max_current` reads `00 00 0b b8`), so the Pi 5 limits USB peripheral current; pure inference is not affected. Do not connect USB peripherals. With poor supply voltage or thin cables, however, full load causes under-voltage (measured on 2026-11-09: a power-loss reboot while exporting the MoE, and repeated under-voltage after changing the cable). The official RPi 4B supply (5.1 V / 3 A) is now used, with no under-voltage after 10 minutes at full load. Verify again after changing the supply or cable.
- `--drop-caches` of `power_rpi.py` needs passwordless sudo and is off by default; if enabled, record it in the protocol.
