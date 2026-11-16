"""Possession model: per-frame evidence + a 3-state (H / A / D) Viterbi decoder.

Pipeline
--------
1. Ball track: linearly interpolate short gaps (<= interp_gap frames) inside each half.
2. Team evidence: for every frame with a ball position, the distance from the ball to the closest
   home player (dH) and closest away player (dA). The score tanh((dA - dH) / team_margin) says
   which team is nearer; it is down-weighted when nobody is within control_radius of the ball
   (a loose ball, a pass in flight) because "nearest" means little then.
3. Dead-ball evidence: ball seen outside the pitch, players moving slowly (stoppages are walked),
   ball not seen at all, and the Smart Tagging set-piece windows (each tags a restart).
4. Smart Tagging team prior (optional, weak): phases tagged with a team add a small bonus to it.
5. Viterbi over three states. Switching H<->A costs switch_cost, going alive->dead costs
   to_dead_cost, dead->alive costs from_dead_cost. Costs are in the same units as one frame of
   full-strength evidence, so switch_cost=40 means "the other team needs ~1.6 s of clear
   evidence before possession changes". This is what makes possessions whole instead of
   flickering, and is the main knob on the number of possessions.
6. Extra dead-ball evidence: a still ball (placed for a set piece), a still ball at a touchline /
   goal line / goal area (throw-in, corner, goal kick), and goal -> kick-off windows.
7. Restart team: when play resumes (a still ball starts moving, or a tagged set-piece restart),
   the team of the player nearest the ball gets a short bonus and the ball counts as alive.
8. Hard structure: frames outside the graded halves are X; frames between the start of a half and
   its tagged kickoff are D.

Everything here is computed from the provided ATD and Smart Tagging only. No ground truth is read.
"""

import os
import sys
from dataclasses import dataclass, asdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


@dataclass
class Params:
    # --- ball track
    interp_gap: int = 12            # max gap (frames) to interpolate the ball over
    # --- team evidence
    team_margin: float = 1.5        # metres; dA - dH of this size ~ 76% of full evidence
    control_radius: float = 3.0     # metres; nearest player farther than this = loose ball
    loose_weight: float = 0.25      # evidence multiplier for a loose ball
    team_weight: float = 1.0        # overall scale of team evidence
    # --- dead-ball evidence
    out_margin: float = 0.5         # metres beyond the line before the ball counts as out
    w_out: float = 3.0              # ball outside the pitch
    w_out_carry: float = 1.0        # ball unseen, but its last sighting was outside the pitch
    slow_speed: float = 1.2         # m/s; mean player speed below this looks like a stoppage
    w_slow: float = 1.0
    w_missing: float = 0.3          # ball not seen (after interpolation)
    w_tag: float = 1.5              # inside a Smart Tagging set-piece pre-restart window
    tag_restart_frac: float = 0.65  # restart sits this far into a set-piece tag window
    tag_pre_s: float = 4.0          # dead-ball window extends this many seconds before the tag
    dead_bias: float = 1.0          # constant cost of being dead (prior towards ball in play)
    # --- Smart Tagging team prior
    w_phase: float = 0.15
    # --- decoder
    switch_cost: float = 40.0
    to_dead_cost: float = 40.0     # keep to_dead + from_dead > switch_cost, or the decoder
                                   # "switches" by sneaking through a fake dead ball
    from_dead_cost: float = 20.0
    # --- step 1: extra dead-ball evidence
    still_speed: float = 1.0      # m/s; ball slower than this counts as still
    still_window: int = 25        # frames (1 s) the ball must stay still
    w_still: float = 1.5          # still ball -> looks like a set piece
    w_zone: float = 1.0           # still ball at touchline / goal line / goal area
    zone_m: float = 2.0           # metres from the line that counts as "at the line"
    w_goal: float = 5.0           # between a goal and its kick-off
    # --- step 2: who restarts
    min_still: int = 25           # a still run this long, then movement = a restart
    kicker_radius: float = 3.0    # kicker must be this close to the ball
    w_restart: float = 1.5        # team bonus for the kicker's side after a restart
    restart_hold_s: float = 2.0   # how long that bonus lasts
    w_restart_alive: float = 2.0  # just after a restart the ball is alive, not dead

    def as_dict(self):
        return asdict(self)


def interpolate_ball(bx, by, blocks, max_gap):
    bx, by = bx.copy(), by.copy()
    for _, lo, hi in blocks:
        x, y = bx[lo:hi + 1], by[lo:hi + 1]
        seen = ~np.isnan(x)
        idx = np.flatnonzero(seen)
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        for start, g in zip(idx[:-1], gaps):
            if 1 < g <= max_gap + 1:
                t = np.arange(1, g) / g
                x[start + 1:start + g] = x[start] + t * (x[start + g] - x[start])
                y[start + 1:start + g] = y[start] + t * (y[start + g] - y[start])
    return bx, by


def team_distances(atd, bx, by):
    """(dH, dA): distance from the ball to the nearest home / away player. NaN if unknown."""
    px, py, team = atd["px"], atd["py"], atd["team"]
    d = np.hypot(px - bx[:, None], py - by[:, None])
    out = []
    for k in (0, 1):
        dk = d[:, team == k]
        with np.errstate(all="ignore"):
            m = np.where(np.all(np.isnan(dk), axis=1), np.nan, np.nanmin(np.nan_to_num(dk, nan=np.inf), axis=1))
        out.append(m.astype(np.float32))
    return out


def load_tags():
    import load_smart_tagging
    return load_smart_tagging.load()


def team_of_tag(tag):
    t = (tag["tags"].get("Team") or "").strip().lower()
    if t.startswith("barcelona"):
        return H
    if t.startswith("atletic") or t.startswith("athletic"):
        return A
    return None

def ball_speed(bx, by, fps=25):
    v = np.full(len(bx), np.nan, dtype=np.float32)
    v[1:] = np.hypot(np.diff(bx), np.diff(by)) * fps   # NaN wherever the ball is unseen
    return v


def rolling_mean(x, w):
    """Centred mean over w frames, ignoring NaN."""
    ok = ~np.isnan(x)
    c = np.r_[0, np.cumsum(np.where(ok, x, 0))]
    k = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    lo = np.clip(i - w // 2, 0, len(x))
    hi = np.clip(i + w // 2 + 1, 0, len(x))
    n = k[hi] - k[lo]
    with np.errstate(all="ignore"):
        return np.where(n >= w // 2, (c[hi] - c[lo]) / n, np.nan)


def still_ball(bx, by, p):
    v = ball_speed(bx, by)
    slow = np.where(np.isnan(v), np.nan, (v < p.still_speed).astype(float))
    return np.nan_to_num(rolling_mean(slow, p.still_window), nan=0.0) > 0.8


def restart_zone(bx, by, p, game=GAME):
    L, W = GAMES[game]["pitch"]
    near_touch = (by < p.zone_m) | (by > W - p.zone_m)
    near_goal_line = (bx < p.zone_m) | (bx > L - p.zone_m)
    goal_area = ((bx < 6.0) | (bx > L - 6.0)) & (np.abs(by - W / 2) < 10.0)
    return near_touch | near_goal_line | goal_area


def goal_dead(atd, tags, p, game=GAME):
    """Dead from the ball entering the goal to the kick-off after it."""
    L, W = GAMES[game]["pitch"]
    n = len(atd["bx"])
    out = np.zeros(n, bool)
    kos = sorted(int(t["start_frame"] + p.tag_restart_frac * (t["end_frame"] - t["start_frame"]))
                 for t in tags if "after goal" in (t["tags"].get("Type") or "").lower())
    for t in tags:
        if (t["code"] or "").strip().lower() != "goals":
            continue
        ko = next((k for k in kos if k >= t["end_frame"] - 50), None)
        if ko is None:
            continue
        s = t["start_frame"]
        bx, by = atd["bx"][s:ko], atd["by"][s:ko]
        in_goal = ((bx < 0.5) | (bx > L - 0.5)) & (np.abs(by - W / 2) < 5.0)
        hits = np.flatnonzero(in_goal)
        start = s + hits[0] if len(hits) else max(s, ko - 60 * 25)   # fallback: 60 s before
        out[start:ko] = True
    return out


def restart_events(atd, still, tag_restarts, p):
    """[(frame, team)]: when play resumes, and the team of the player nearest the ball."""
    frames = []
    r = np.diff(np.r_[0, still.astype(int), 0])
    starts, ends = np.flatnonzero(r == 1), np.flatnonzero(r == -1)
    frames += [e for s, e in zip(starts, ends) if e - s >= p.min_still]   # (a) still ball kicked
    frames += list(tag_restarts)                                           # (b) tagged set pieces

    px, py, team, bx, by = atd["px"], atd["py"], atd["team"], atd["bx"], atd["by"]
    out = []
    for f in sorted(set(frames)):
        # the ball sighting closest to the restart, within 1 s before / 0.2 s after
        win = np.arange(max(0, f - 25), min(len(bx), f + 5))
        seen = win[~np.isnan(bx[win])]
        if len(seen) == 0:
            continue
        k = seen[np.argmin(np.abs(seen - f))]
        d = np.hypot(px[k] - bx[k], py[k] - by[k])
        if np.all(np.isnan(d)):
            continue
        j = np.nanargmin(d)
        if d[j] > p.kicker_radius:          # nobody close enough: don't guess
            continue
        out.append((int(f), int(team[j])))
    return out


def features(atd, tags, p, game=GAME):
    n = len(atd["bx"])
    length, width = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)

    raw_seen = ~np.isnan(atd["bx"])
    bx, by = interpolate_ball(atd["bx"], atd["by"], blocks, p.interp_gap)
    seen = ~np.isnan(bx)

    dH, dA = team_distances(atd, bx, by)
    nearest = np.fmin(dH, dA)
    diff = np.nan_to_num(dA - dH, nan=0.0)
    gate = np.where(nearest <= p.control_radius, 1.0, p.loose_weight)
    team_score = p.team_weight * np.tanh(diff / p.team_margin) * gate
    team_score[~seen] = 0.0

    out = raw_seen & ((atd["bx"] < -p.out_margin) | (atd["bx"] > length + p.out_margin) |
                      (atd["by"] < -p.out_margin) | (atd["by"] > width + p.out_margin))
    # last known ball position, carried across frames where it is not seen
    idx = np.where(raw_seen, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    last_out = out[idx] & ~raw_seen
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(atd["ps"], axis=1)
    mean_speed = np.nan_to_num(mean_speed, nan=p.slow_speed)
    slow = np.clip((p.slow_speed - mean_speed) / p.slow_speed, -1.0, 1.0)

    tag_dead = np.zeros(n, dtype=np.float32)
    phase = np.zeros(n, dtype=np.float32)
    kickoffs = []
    for t in tags:
        code = (t["code"] or "").strip().lower()
        s, e = t["start_frame"], t["end_frame"]
        if code == "set pieces":
            restart = int(s + p.tag_restart_frac * (e - s))
            lo = max(0, int(s - p.tag_pre_s * 25))
            tag_dead[lo:restart] = 1.0
            if (t["tags"].get("Type") or "").strip().lower() == "kick off (start)":
                kickoffs.append(restart)
        else:
            k = team_of_tag(t)
            if k is not None:
                phase[s:e] += 1.0 if k == H else -1.0
    phase = np.clip(phase, -1, 1)

    dead_score = (p.w_out * out + p.w_out_carry * last_out + p.w_slow * slow + p.w_missing * (~seen) +
                  p.w_tag * tag_dead - p.dead_bias)

    # log-emission scores, larger = more likely
    E = np.zeros((n, 3), dtype=np.float64)
    E[:, H] = team_score + p.w_phase * phase
    E[:, A] = -team_score - p.w_phase * phase
    E[:, D] = dead_score

    # --- step 1: still ball, restart zones, goals
    still = still_ball(atd["bx"], atd["by"], p)
    zone = restart_zone(atd["bx"], atd["by"], p) & still
    E[:, D] += p.w_still * still + p.w_zone * zone + p.w_goal * goal_dead(atd, tags, p)

    # --- step 2: give the restart to the kicker's team
    tag_restarts = [int(t["start_frame"] + p.tag_restart_frac * (t["end_frame"] - t["start_frame"]))
                    for t in tags if (t["code"] or "").strip().lower() == "set pieces"]
    hold = int(p.restart_hold_s * 25)
    for f, k in restart_events(atd, still, tag_restarts, p):
        window = slice(f, min(n, f + hold))
        E[window, k] += p.w_restart          # kicker's team has the ball
        E[window, D] -= p.w_restart_alive    # and it is in play
    return E, kickoffs



def viterbi(E, p):
    """Max-score path through H/A/D with transition penalties. E is (T, 3)."""
    T = len(E)
    cost = np.array([[0, p.switch_cost, p.to_dead_cost],
                     [p.switch_cost, 0, p.to_dead_cost],
                     [p.from_dead_cost, p.from_dead_cost, 0]], dtype=np.float64)
    score = E[0].copy()
    back = np.zeros((T, 3), dtype=np.int8)
    for t in range(1, T):
        cand = score[:, None] - cost          # from i to j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + E[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(atd, tags, p=None, game=GAME):
    p = p or Params()
    E, kickoffs = features(atd, tags, p, game)
    n = len(E)
    labels = np.full(n, "X", dtype="<U1")
    for _, lo, hi in half_video_blocks(game):
        ko = [k for k in kickoffs if lo - 250 <= k <= hi]
        start = lo
        if ko:
            k = max(lo, min(ko))
            labels[lo:k] = "D"
            start = k
        if start <= hi:
            labels[start:hi + 1] = LETTERS[viterbi(E[start:hi + 1], p)]
    return labels
