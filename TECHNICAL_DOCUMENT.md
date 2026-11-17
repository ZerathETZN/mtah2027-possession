# Technical document: frame-level ball possession

More than a Hack 2027, Phase 1. Code: https://github.com/ZerathETZN/mtah2027-possession
(our code is in `possession/`; the rest is the organisers' starter kit).

## 1. Approach

We score every frame for three states (home in possession, away in possession, dead ball) from
hand-designed evidence, then decode the whole half with a 3-state Viterbi path that charges a
cost for every change of state.

We chose this over a per-frame classifier for two reasons. First, there is no labelled data for
this game that we are allowed to train on. Second, the ranking metric (spellF1) rewards whole,
correctly bounded possessions and punishes flicker heavily. A decoder with switching costs gives
whole possessions by construction, and its few parameters each have a football meaning.

Pipeline:

1. **Team evidence.** For each frame where the ball is tracked, we compute the distance from the
   ball to the nearest home player (dH) and the nearest away player (dA), in metres. The evidence
   for home is `tanh((dA - dH) / 1.5 m)` and the evidence for away is its negative. When nobody is
   within 3 m of the ball (a loose ball or a pass in flight), the evidence is scaled down to 25%.
   Frames without a ball carry no team evidence.
2. **Dead-ball evidence**, summed per frame:
   - the ball is tracked outside the pitch, or was last seen outside it and has not been seen since;
   - the mean player speed is below 1.2 m/s (players walk during stoppages);
   - the ball is not tracked;
   - the frame falls inside a Smart Tagging set-piece window, before the restart;
   - the ball is still (under 1 m/s for 1 s), more so at a touchline, goal line or goal area;
   - the frame lies between a goal and the kick-off after it (from the GOALS and
     "Kick Off (After Goal)" tags).

   A constant bias favours "ball in play".
3. **Restart team.** Play resumes when a still ball starts moving, or at a tagged set-piece
   restart. At that moment, the player nearest the tracked ball (within 3 m) is taken as the
   kicker. Their team gets a bonus for 2 s, and the dead-ball state is penalised over the same
   2 s.
4. **Team prior.** Smart Tagging phases tagged Barcelona or Athletic add a small bonus (0.15) to
   that team.
5. **Viterbi decoding**, run separately for each half. Changing H to A costs 40, which means
   about 1.6 s of clear evidence is needed. Alive to dead costs 40 and dead to alive costs 20.
   Their sum is kept above the switch cost, so possession changes are not routed through
   invented stoppages.
6. **Structure.** Ungraded frames (before the match, halftime, after the final whistle) are `X`.
   Frames from the start of a half to its tagged kick-off are `D`.

## 2. Assets used

| asset | used | how |
|---|---|---|
| Tactical ATD | yes | ball and player positions, player speeds, team of each track slot |
| Smart Tagging | yes | set-piece windows, kick-offs, goals, team-tagged phases (weak prior) |
| Tactical homography | no | the ATD is already in pitch coordinates |
| Tactical video | no | not needed for the features above; no time for ball detection |
| Broadcast video, ATD, homography | no | not synchronised to the tactical timeline |

Tried and dropped:
- **Linear interpolation of short ball gaps.** It filled only 0.03% of frames, because when the
  ball disappears it is almost always for long stretches.

## 3. Training data

None. No model is trained and no labelled data from any game is used. The parameters were set
from football reasoning and from statistics that need no ground truth: possession count,
possession length and dead-ball share over the full match.

The public 5-minute ground truth was used only to check the pipeline: frame alignment, file
format, and that the labels mean what we think. We did not use it to choose parameters.

## 4. Possession rules implemented

- **Change of possession:** the other team must be clearly nearer the ball for about 1.6 s in
  total (switch cost 40). Single touches, deflections and duels shorter than that do not change
  possession.
- **Loose balls:** possession stays with the last team. Evidence is down-weighted when nobody is
  within 3 m of the ball, and frames without a ball carry no evidence, so the decoder keeps the
  previous team.
- **Dead ball:** ball out of the pitch, the stoppage before a tagged set piece, a still ball
  (especially in restart zones), goal to kick-off, and before each half's kick-off.
- **Restarts:** the team of the player nearest the ball when play resumes starts the new
  possession.

## 5. Results

On the public 5-minute window, scored with `evaluation/run_evaluation.py` (pipeline check only;
17 possessions, so differences under about 0.06 are noise):

| submission | spellF1 | macroF1 | possessions (truth 17) |
|---|---|---|---|
| ours | 0.586 | 0.706 | 11 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |
| baseline_smooth50 (published floor) | 0.326 | 0.621 | 26 |

The 95% bootstrap interval for our spellF1 is 0.37 to 0.76.

Full match, with no ground truth available: 215 possessions (103 in the first half, 112 in the
second), mean length 23.6 s, median 15.2 s. 20.1% of graded frames are labelled dead.
Barcelona has 60.1% of the frames where the ball is in play.

## 6. Limitations

- **Nearest player is a weak stand-in for control.** It fails in duels and while long passes
  are in the air.
- **The ball is missing on 41% of graded frames**, so possession through those stretches is
  carried by the decoder rather than observed.
- **Too few dead balls.** We label about 20% of frames dead, below the 35-40% typical of a
  professional match.
- **A goalkeeper holding the ball can look like a dead ball**, because the ball is still.
- **Throw-ins carry no team tag,** and restarts are attributed only when the ball is tracked at
  the moment play resumes (73 restarts in this match).
- **Short possessions get merged.** The switch cost that removes flicker also absorbs
  possessions shorter than about 1.5 s.

With more time, we would:
- detect the ball in the tactical video where the ATD loses it, and project it with the
  homography;
- learn possession from player-only features (team shape, direction of movement), using our
  own algorithmic labels on frames with a ball, and apply it where the ball is missing;
- calibrate the parameters on labelled tracking from other games, such as Metrica's public
  sample data.

## 7. Reproduction

Requires Python 3.11 with numpy 1.26.4 and pandas 2.2.3 (`requirements-solution.txt`).

```bash
pip install -r requirements-solution.txt
# unpack the match bundle into ./data (or set MTAH_DATA_DIR), as described in DATA.md
python possession/run.py            # writes submissions/submission.csv
python evaluation/validate_submission.py submissions/submission.csv
```

The run is deterministic: two runs produce byte-identical files. It takes a few seconds once
the tracking file has been parsed and cached in `cache/` (gitignored).

Optional: `python possession/sweep.py` sweeps the parameters, and `python possession/plots.py`
draws the comparison figures.
