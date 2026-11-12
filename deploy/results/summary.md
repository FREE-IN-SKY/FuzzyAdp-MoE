# Device measurement summary (generated, do not edit by hand; source deploy/results/)

## Latency (batch 1, CPU-only TorchScript; ms/image)

| Board | Model | N | Threads | Model mean | Median | p90 | End-to-end mean | Throughput (img/s) | On-board accuracy | Agreement with experiment-machine predictions | Power supply |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orin_nano | moe_auxfree_s3_top2 | 5810 | 6 | 87.49 | 81.27 | 90.90 | 93.27 | 10.69 | 0.6511 | 1.0000 | n/a |
| orin_nano | moe_fixed_s3_top1 | 5810 | 6 | 82.32 | 76.23 | 86.08 | 88.10 | 11.31 | 0.6337 | 0.9628 | n/a |
| orin_nano | moe_fixed_s3_top2 | 5810 | 6 | 87.78 | 81.18 | 93.45 | 93.49 | 10.66 | 0.6404 | 1.0000 | n/a |
| orin_nano | moe_fuzzy_s3_top2 | 5810 | 6 | 88.77 | 82.98 | 90.62 | 94.64 | 10.53 | 0.6454 | 1.0000 | n/a |
| orin_nano | moe_schedule_s3_top2 | 5810 | 6 | 87.49 | 81.15 | 91.15 | 93.24 | 10.69 | 0.6492 | 1.0000 | n/a |
| orin_nano | single_s3 | 5810 | 6 | 80.57 | 75.12 | 83.07 | 86.46 | 11.54 | 0.6439 | 1.0000 | n/a |
| rpi4b | moe_auxfree_s3_top2 | 5810 | 4 | 242.07 | 241.79 | 243.93 | 249.12 | 4.01 | 0.6511 | 1.0000 | clean |
| rpi4b | moe_fixed_s3_top1 | 5810 | 4 | 229.54 | 229.23 | 231.35 | 236.60 | 4.22 | 0.6337 | 0.9628 | clean |
| rpi4b | moe_fixed_s3_top2 | 5810 | 4 | 241.31 | 241.04 | 243.11 | 248.40 | 4.02 | 0.6404 | 1.0000 | clean |
| rpi4b | moe_fuzzy_s3_top2 | 5810 | 4 | 241.31 | 240.86 | 242.95 | 248.32 | 4.02 | 0.6454 | 1.0000 | clean |
| rpi4b | moe_schedule_s3_top2 | 5810 | 4 | 241.80 | 241.57 | 243.79 | 248.81 | 4.02 | 0.6492 | 1.0000 | clean |
| rpi4b | single_s3 | 5810 | 4 | 227.45 | 226.84 | 231.09 | 236.55 | 4.23 | 0.6439 | 1.0000 | clean |
| rpi5 | moe_auxfree_s3_top2 | 5810 | 4 | 81.94 | 82.00 | 82.60 | 85.27 | 11.71 | 0.6511 | 1.0000 | clean |
| rpi5 | moe_fixed_s3_top1 | 5810 | 4 | 77.21 | 77.22 | 77.94 | 80.54 | 12.40 | 0.6337 | 0.9628 | clean |
| rpi5 | moe_fixed_s3_top2 | 5810 | 4 | 82.04 | 82.03 | 82.65 | 85.36 | 11.70 | 0.6404 | 1.0000 | clean |
| rpi5 | moe_fuzzy_s3_top2 | 5810 | 4 | 81.84 | 81.90 | 82.47 | 85.15 | 11.73 | 0.6454 | 1.0000 | clean |
| rpi5 | moe_schedule_s3_top2 | 5810 | 4 | 82.37 | 82.45 | 83.01 | 85.70 | 11.66 | 0.6492 | 1.0000 | clean |
| rpi5 | single_s3 | 5810 | 4 | 76.33 | 76.41 | 77.07 | 80.79 | 12.37 | 0.6439 | 1.0000 | clean |
| xavier_nx | moe_auxfree_s3_top2 | 5810 | 6 | 98.23 | 97.36 | 100.08 | 105.41 | 9.46 | 0.6511 | 1.0000 | n/a |
| xavier_nx | moe_fixed_s3_top1 | 5810 | 6 | 92.32 | 90.96 | 95.49 | 99.64 | 10.01 | 0.6337 | 0.9628 | n/a |
| xavier_nx | moe_fixed_s3_top2 | 5810 | 6 | 98.53 | 97.42 | 100.87 | 105.77 | 9.43 | 0.6404 | 1.0000 | n/a |
| xavier_nx | moe_fuzzy_s3_top2 | 5810 | 6 | 99.53 | 98.50 | 101.07 | 106.70 | 9.35 | 0.6454 | 1.0000 | n/a |
| xavier_nx | moe_schedule_s3_top2 | 5810 | 6 | 99.00 | 97.96 | 101.34 | 106.17 | 9.40 | 0.6492 | 1.0000 | n/a |
| xavier_nx | single_s3 | 5810 | 6 | 93.47 | 90.85 | 99.94 | 100.76 | 9.91 | 0.6439 | 1.0000 | n/a |

Latency ratio (same board, model mean): 
- orin_nano: moe_auxfree_s3_top2 / single = 1.09; moe_fixed_s3_top1 / single = 1.02; moe_fixed_s3_top2 / single = 1.09; moe_fuzzy_s3_top2 / single = 1.10; moe_schedule_s3_top2 / single = 1.09
- rpi4b: moe_auxfree_s3_top2 / single = 1.06; moe_fixed_s3_top1 / single = 1.01; moe_fixed_s3_top2 / single = 1.06; moe_fuzzy_s3_top2 / single = 1.06; moe_schedule_s3_top2 / single = 1.06
- rpi5: moe_auxfree_s3_top2 / single = 1.07; moe_fixed_s3_top1 / single = 1.01; moe_fixed_s3_top2 / single = 1.07; moe_fuzzy_s3_top2 / single = 1.07; moe_schedule_s3_top2 / single = 1.08
- xavier_nx: moe_auxfree_s3_top2 / single = 1.05; moe_fixed_s3_top1 / single = 0.99; moe_fixed_s3_top2 / single = 1.05; moe_fuzzy_s3_top2 / single = 1.06; moe_schedule_s3_top2 / single = 1.06

## Raspberry Pi power-bank discharge method (energy per sample in J; when E_out is unknown, [low, high] efficiency assumptions are given)

| Board | Model | Runs | Duration (h) | Images | E/sample (J) | P_avg (W) | Heartbeats | First undervoltage/throttling (fraction of duration) |
|---|---|---|---|---|---|---|---|---|
| rpi4b | moe_fixed_s3_top2 | 2 | 7.22 (7.16 ~ 7.27) | 104438 (104089 ~ 104787) | low: 0.893 (0.890 ~ 0.896) / high: 1.084 (1.080 ~ 1.088) | low: 3.59 (3.56 ~ 3.61) / high: 4.36 (4.32 ~ 4.39) | 2542, 2576 | none, none |
| rpi4b | single_s3 | 2 | 7.24 (7.21 ~ 7.27) | 110799 (110167 ~ 111431) | low: 0.842 (0.837 ~ 0.846) / high: 1.022 (1.016 ~ 1.028) | low: 3.58 (3.56 ~ 3.59) / high: 4.34 (4.33 ~ 4.36) | 2591, 2562 | none, none |
| rpi5 | moe_fixed_s3_top2 | 2 | 5.05 (5.04 ~ 5.06) | 210258 (209711 ~ 210806) | low: 0.443 (0.442 ~ 0.445) / high: 0.538 (0.537 ~ 0.540) | low: 5.13 (5.12 ~ 5.14) / high: 6.23 (6.22 ~ 6.24) | 1812, 1807 | none, none |
| rpi5 | single_s3 | 2 | 4.90 (4.72 ~ 5.08) | 219494 (211842 ~ 227146) | low: 0.425 (0.410 ~ 0.440) / high: 0.516 (0.498 ~ 0.534) | low: 5.30 (5.10 ~ 5.49) / high: 6.43 (6.19 ~ 6.67) | 1691, 1820 | none, none |

Discharge validity: if the first nonzero throttled flag falls in the last 5% of the duration, it is the normal voltage drop of a nearly empty power bank and the run is valid; if it appears earlier or is already nonzero at the start, the run is invalid and excluded from the table above and from the energy ratios.

E_out (Goji G6P10BK25) = 37.0 Wh × [0.7, 0.85] (nominal 3.7 V cell energy × usable fraction). The manufacturer gives no 5 V output capacity or efficiency for this power bank, so both the lower bound 0.7 and the upper bound 0.85 are generic assumptions, not manufacturer data; its absolute energy is less well supported than for a power bank with manufacturer data, is for reference only, and is not compared across boards.
E_out (Intenso F10000) = 37.0 Wh × [0.7, 0.85] (nominal 3.7 V cell energy × usable fraction). The lower bound 0.7 comes from the manufacturer's manual (Intenso F10000 operating instructions, f10000_bda_alle.pdf: "about 30 % of the total capacity is used up just through the charging process"); the upper bound 0.85 is a common measured upper limit for similar 5 V boost power banks, not manufacturer data. Absolute energy is reliable only within this interval.

Energy ratio (same board, independent of the E_out assumption): E_MoE / E_single = N_single / N_MoE
- rpi4b: moe_fixed_s3_top2: 1.06
- rpi5: moe_fixed_s3_top2: 1.04

## Jetson (tegrastats / INA3221, CPU-only)

| Board | Model | Duration (s) | Images | P_idle (W) | P_load (W) | Samples | e2e ms | E/sample (J) | Incremental E/sample (J) | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| orin_nano | moe_fixed_s3_top1 | 1200 | 13571 | 2.92 | 5.27 | 1191 | 88.43 | 0.466 | 0.208 | tegrastats:VDD_IN |
| orin_nano | moe_fixed_s3_top2 | 1200 | 12825 | 2.92 | 5.28 | 1191 | 93.57 | 0.494 | 0.221 | tegrastats:VDD_IN |
| orin_nano | single_s3 | 1200 | 13772 | 2.92 | 5.30 | 1191 | 87.14 | 0.461 | 0.207 | tegrastats:VDD_IN |
| xavier_nx | moe_fixed_s3_top1 | 1200 | 12253 | 1.74 | 5.19 | 1163 | 97.94 | 0.509 | 0.339 | tegrastats:VDD_IN |
| xavier_nx | moe_fixed_s3_top2 | 1200 | 11069 | 1.75 | 5.06 | 1166 | 108.41 | 0.548 | 0.358 | tegrastats:VDD_IN |
| xavier_nx | single_s3 | 1200 | 11745 | 1.71 | 5.05 | 1166 | 102.18 | 0.516 | 0.341 | tegrastats:VDD_IN |

## Environment

- orin_nano: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super; Ubuntu 22.04.5 LTS; python 3.13.5, torch 2.11.0+cpu, torchvision 0.26.0+cpu, timm 1.0.22; cpus 6; governor ['schedutil', 'schedutil']; nvpmodel: NV Power Mode: 25W
1
- rpi4b: Raspberry Pi 4 Model B Rev 1.4; Debian GNU/Linux 13 (trixie); python 3.13.5, torch 2.11.0+cpu, torchvision 0.26.0+cpu, timm 1.0.22; cpus 4; governor ['ondemand']
- rpi5: Raspberry Pi 5 Model B Rev 1.0; Debian GNU/Linux 13 (trixie); python 3.13.5, torch 2.11.0+cpu, torchvision 0.26.0+cpu, timm 1.0.22; cpus 4; governor ['ondemand']
- xavier_nx: NVIDIA Jetson Xavier NX Developer Kit; Ubuntu 20.04.6 LTS; python 3.13.5, torch 2.11.0+cpu, torchvision 0.26.0+cpu, timm 1.0.22; cpus 6; governor ['schedutil']; nvpmodel: NV Power Mode: MODE_15W_6CORE
2
NVPM ERROR: Error opening /sys/kernel/nvpmodel_e
