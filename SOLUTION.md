# Solution: evidence + 3-state Viterbi possession decoder

Our entry for More than a Hack 2027, Phase 1. Everything else in this repository is the
organisers' starter kit (MIT). Our code is in [`possession/`](possession/).

## Reproduce the submission

```bash
pip install -r requirements-solution.txt      # numpy + pandas, pinned
# unpack the match bundle into ./data (or set MTAH_DATA_DIR) -- see DATA.md
python possession/run.py                      # -> submissions/submission.csv
python evaluation/validate_submission.py submissions/submission.csv
```

Runs in a few seconds after the first ATD parse (cached in `cache/`, gitignored). Deterministic.

## Approach

1. **Ball-to-team distance.** For every frame with a ball position, the distance to the nearest
   home player `dH` and nearest away player `dA` (metres, 105.3 x 68 pitch). Team evidence is
   `tanh((dA - dH) / team_margin)`, scaled down by `loose_weight` when nobody is within
   `control_radius` (loose ball / pass in flight). No ball, no team evidence.
2. **Dead-ball evidence.** Ball seen outside the pitch (`w_out`); ball unseen since it was last seen
   outside (`w_out_carry`); mean player speed below `slow_speed` (stoppages are walked, `w_slow`);
   ball not tracked (`w_missing`); inside a Smart Tagging *set piece* window before the restart
   (`w_tag`). A constant `dead_bias` encodes the prior that the ball is usually in play.
3. **Smart Tagging team prior.** Phases tagged Barcelona / Athletic add a weak `w_phase` bonus.
4. **Viterbi decoding** over H / A / D with transition costs `switch_cost` (H<->A), `to_dead_cost`
   and `from_dead_cost`. A cost of 40 means roughly 1.6 s of clear evidence is needed to change
   possession -- our encoding of "controlled" possession, which ignores touches and deflections.
   `to_dead_cost + from_dead_cost > switch_cost` so changes are not routed through fake stoppages.
5. **Structure.** Ungraded frames are `X`; frames from the start of a half to its tagged kickoff are
   `D`.

## Assets used

| asset | used | how |
|---|---|---|
| tactical ATD | yes | ball + player positions, speeds, team of each slot |
| Smart Tagging | yes | set-piece windows (dead ball), team phases (weak prior), kickoffs |
| homography | no | ATD is already in pitch coordinates |
| videos, broadcast ATD | no | not synchronised; no time |

## Training data

None. No model is trained; no labels from any game are used. Parameters were set from football
reasoning and GT-free statistics (possession count, dead-ball share). The public 5 minutes were
used only to check the pipeline (alignment and file format), in line with RULES.md.

## Parameters

See `Params` in [`possession/model.py`](possession/model.py) and the sweep in
[`TUNING_SWEEP.md`](TUNING_SWEEP.md) (`python possession/sweep.py`).

## Limitations

- Nearest-player is a weak proxy for control; it fails in duels and during long passes.
- The ball is missing on 41% of graded frames, often for long stretches, so interpolation does not
  help and possession is carried by the decoder.
- Dead-ball detection is heuristic; throw-ins have no team tag.
- With more time: learn per-frame emissions from labelled *other* games, use ball velocity
  (pass/shot detection), use the video for ball detection where ATD loses it.
