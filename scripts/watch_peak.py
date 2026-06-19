"""Watch a run's train.log and auto-capture latest.pt whenever vs_rand sets a new high.

Peaks are written into <RUN_DIR>/peaks/ so the run root stays clean
(latest.pt / best.pt / ship.pt / meta.json). Point RUN_DIR at the active run.
"""
import time
import shutil
import os
import re

RUN_DIR = "runs/n4_5x5_v3"
LOG = os.path.join(RUN_DIR, "train.log")
CKPT_DIR = RUN_DIR
PEAK_DIR = os.path.join(RUN_DIR, "peaks")
os.makedirs(PEAK_DIR, exist_ok=True)
PATTERN = re.compile(r">>> iter (\d+) .* vs_rand=([\d.]+)%")

best_pct = 0.0
last_line = 0

while True:
    if not os.path.exists(LOG):
        time.sleep(10)
        continue

    with open(LOG) as f:
        lines = f.readlines()

    for line in lines[last_line:]:
        m = PATTERN.search(line)
        if m:
            it = int(m.group(1))
            pct = float(m.group(2))
            if pct >= best_pct:
                best_pct = pct
                src = os.path.join(CKPT_DIR, "latest.pt")
                dst = os.path.join(PEAK_DIR, f"peak_iter{it}_{pct:.0f}.pt")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"[peak] iter {it}: {pct}% -> {dst}", flush=True)

    last_line = len(lines)
    time.sleep(30)
