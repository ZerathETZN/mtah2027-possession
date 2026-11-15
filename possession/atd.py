"""Fast loader for the tactical ATD: parse the raw text once into numpy arrays and cache them.

Arrays (all indexed by *video frame* 0..n-1, i.e. ATD frame_id - 1):
    px, py   (n, S) float32  player positions in metres, NaN where the slot is not tracked
    ps       (n, S) float32  player speed m/s
    team     (S,)   int8     0 = home, 1 = away
    bx, by   (n,)   float32  ball position in metres, NaN where the ball is not seen
    bz       (n,)   float32  third ball channel as delivered (unused by default)

The cache lives in ./cache (gitignored): it is derived from the gated data and must not be
published.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    length, width = GAMES[game]["pitch"]

    with open(raw_path) as fh:
        text = fh.read()
    text = text.replace(":", ",").replace(";", ",")
    n_cols = 1 + 3 * n_slots + 3
    df = pd.read_csv(io.StringIO(text), header=None, names=range(n_cols), dtype=np.float64,
                     na_values=["NaN"])
    del text
    arr = df.to_numpy()
    frame_id = arr[:, 0].astype(np.int64)
    n = source_frames("tactical", game)
    idx = frame_id - 1

    def full(shape):
        return np.full(shape, np.nan, dtype=np.float32)

    px, py, ps = full((n, n_slots)), full((n, n_slots)), full((n, n_slots))
    players = arr[:, 1:1 + 3 * n_slots].reshape(len(arr), n_slots, 3)
    px[idx] = players[:, :, 0] * length
    py[idx] = players[:, :, 1] * width
    ps[idx] = players[:, :, 2]
    bx, by, bz = full(n), full(n), full(n)
    ball = arr[:, 1 + 3 * n_slots:]
    bx[idx] = ball[:, 0] * length
    by[idx] = ball[:, 1] * width
    bz[idx] = ball[:, 2]
    return dict(px=px, py=py, ps=ps, team=team, bx=bx, by=by, bz=bz)


def load(game=GAME, data_dir=None, refresh=False):
    path = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(game))
    if os.path.isfile(path) and not refresh:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    data = parse(game, data_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, **data)
    return data


if __name__ == "__main__":
    d = load(refresh=True)
    print({k: v.shape for k, v in d.items()})
