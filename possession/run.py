"""Build a submission from the provided data.

    python possession/run.py                       # default parameters -> submissions/submission.csv
    python possession/run.py --set switch_cost=60  # override any Params field
    python possession/run.py --name myteam --check # write, validate, and sanity-check on public 5 min

Deterministic: the same data and parameters always give the same file.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from possession import atd as atd_mod  # noqa: E402
from possession.model import Params, load_tags, predict  # noqa: E402

OUT_DIR = os.path.join(ROOT, "submissions")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def parse_overrides(pairs):
    p = Params()
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        if not hasattr(p, key):
            raise SystemExit("unknown parameter {!r}; have {}".format(key, list(p.as_dict())))
        setattr(p, key, type(getattr(p, key))(value))
    return p


def write(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


def summarise(labels):
    """Possession count and mean duration on graded frames -- no ground truth involved."""
    alive = np.isin(labels, ["H", "A"])
    runs, prev = [], None
    for i, l in enumerate(labels):
        if l != prev:
            runs.append([l, 1]); prev = l
        else:
            runs[-1][1] += 1
    poss = [n for l, n in runs if l in ("H", "A")]
    return {"H": int((labels == "H").sum()), "A": int((labels == "A").sum()),
            "D": int((labels == "D").sum()), "X": int((labels == "X").sum()),
            "possessions": len(poss), "mean_poss_s": float(np.mean(poss) / 25) if poss else 0.0}


def score_public(labels):
    import possession_metrics as pm
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    r = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="submission")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    ap.add_argument("--check", action="store_true",
                    help="sanity-check against the public 5 minutes (pipeline check, not tuning)")
    args = ap.parse_args()

    p = parse_overrides(args.set)
    labels = predict(atd_mod.load(), load_tags(), p)
    path = write(os.path.join(OUT_DIR, args.name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
    print("summary", summarise(labels))
    if args.check:
        r = score_public(labels)
        print("public 5-min: spellF1 {:.4f}  macroF1 {:.4f}  poss {} (gt {})".format(
            r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))


if __name__ == "__main__":
    main()
