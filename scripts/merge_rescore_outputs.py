"""Combine the rescoring runs into outputs/, dropping the contaminated v7 rows.

The main table loaded runs/n2_9x9_v7/latest.pt at 18:38 while the v7 training run
was still writing it (finished 19:00:38), so those four v7 rows are from a
pre-final iteration and are not comparable with anything else in the table. They
are replaced by two dedicated runs against the now-frozen iteration 60.

    python merge_outputs.py <repo_root> <scratchpad>
"""
import json
import os
import sys

CONTAMINATED = "runs/n2_9x9_v7/latest.pt"


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(repo, scratch):
    main_tbl = load(os.path.join(repo, "outputs", "held_out_eval.json"))
    if main_tbl is None:
        sys.exit("no outputs/held_out_eval.json - run the main table first")

    rows = [r for r in main_tbl["results"] if r["ckpt"] != CONTAMINATED]
    dropped = len(main_tbl["results"]) - len(rows)

    added = 0
    for sub in ("v7_200", "v7_600"):
        d = load(os.path.join(scratch, sub, "held_out_eval.json"))
        if d is None:
            print(f"  WARNING: {sub} missing - v7 will be incomplete")
            continue
        for r in d["results"]:
            r["checkpoint_iteration"] = 60
            r["provenance"] = (
                "dedicated run against the frozen iteration-60 checkpoint; the "
                "main table's v7 rows were read while the run was still writing")
            rows.append(r)
            added += 1

    controls = main_tbl.get("controls", [])
    meta = {k: v for k, v in main_tbl.items() if k not in ("results", "controls")}
    meta["note_v7"] = (
        "v7 rows come from dedicated runs at the frozen iteration 60, at 200 and "
        "600 sims. Rows from the main sweep were discarded: that sweep read "
        "latest.pt while the training run was still writing it.")

    out = {**meta, "results": rows, "controls": controls}
    with open(os.path.join(repo, "outputs", "held_out_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    ablation = [r for r in rows if r["board"] == 9]
    with open(os.path.join(repo, "outputs", "v7_vs_v4.json"), "w") as f:
        json.dump({**meta, "results": ablation}, f, indent=2)

    print(f"dropped {dropped} contaminated v7 rows, added {added} clean ones")
    print(f"outputs/held_out_eval.json: {len(rows)} rows, {len(controls)} controls")
    print(f"outputs/v7_vs_v4.json:      {len(ablation)} rows (9x9 only)")

    print("\n=== 9x9 summary, best cell per checkpoint ===")
    by = {}
    for r in rows:
        if r["board"] != 9:
            continue
        key = (r["ckpt"], r["opponent"])
        if key not in by or r["rate"] > by[key]["rate"]:
            by[key] = r
    for (ck, opp), r in sorted(by.items()):
        print(f"  {ck:32s} vs {opp:8s} {100*r['rate']:5.1f}%  "
              f"K={r['wall_candidates']:<3d} {r['sims']} sims  "
              f"({r['decided']}/{r['games']} decided)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
