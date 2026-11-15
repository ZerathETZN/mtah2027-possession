"""One-at-a-time sweep of the model parameters.

For each value it reports
  * full-match statistics that need NO ground truth: possession count, mean possession length,
    dead-ball share of graded frames, home share of alive frames;
  * the public 5-minute spellF1 / macroF1.

The public columns are a pipeline sanity check and an illustration of sensitivity. RULES.md forbids
using them as a tuning set, and EVALUATION.md shows they rank badly (17 possessions; ~0.06 per
missed possession). Choose parameters on football grounds and the GT-free columns.

    python possession/sweep.py > TUNING_SWEEP.md
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import atd as atd_mod  # noqa: E402
from possession.model import Params, load_tags, predict  # noqa: E402
from possession.run import PUBLIC_GT, summarise  # noqa: E402

GRID = {
    "switch_cost": [10, 20, 30, 40, 60, 100, 150],
    "to_dead_cost": [20, 40, 80],
    "from_dead_cost": [10, 20, 40],
    "dead_bias": [0.5, 1.0, 1.5],
    "w_slow": [0.0, 1.0, 2.0],
    "slow_speed": [0.8, 1.2, 1.6],
    "w_tag": [0.0, 1.5, 3.0],
    "w_phase": [0.0, 0.15, 0.4],
    "control_radius": [2.0, 3.0, 5.0],
    "loose_weight": [0.0, 0.25, 1.0],
    "team_margin": [0.5, 1.5, 3.0],
    "interp_gap": [0, 12],
    "w_out_carry": [0.0, 1.0, 2.0],
}


def main():
    atd, tags = atd_mod.load(), load_tags()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    graded = np.zeros(len(gt), bool)
    graded[11185:90652] = True
    graded[110808:190198] = True
    default = Params().as_dict()

    print("| parameter | value | full: possessions | full: mean poss (s) | full: dead % | "
          "full: home % | public spellF1 | public macroF1 |")
    print("|---|---|---|---|---|---|---|---|")
    for key, values in GRID.items():
        for v in values:
            p = Params(**{**default, key: v})
            labels = predict(atd, tags, p)
            s = summarise(labels)
            g = labels[graded]
            alive = np.isin(g, ["H", "A"])
            r = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)
            mark = " **(default)**" if v == default[key] else ""
            print("| {} | {}{} | {} | {:.1f} | {:.1f} | {:.1f} | {:.4f} | {:.4f} |".format(
                key, v, mark, s["possessions"], s["mean_poss_s"],
                100 * np.mean(g == "D"), 100 * np.mean(g[alive] == "H"),
                r["spell_f1_mean"], r["macro_f1"]))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
