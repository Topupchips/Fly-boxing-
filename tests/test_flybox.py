import csv
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flybox.arena import MAX_HEALTH, fight
from flybox.brain import BrainSpec
from flybox.connectome import build_default_connectome, load_flywire


def test_connectome_respects_dales_law():
    c = build_default_connectome()
    spec = BrainSpec(c)
    W, _ = spec.weights(spec.random_genomes(4, np.random.default_rng(0)))
    for i in range(4):
        dense = spec.dense(W[i])
        for pre in range(c.n):
            col = dense[:, pre]
            assert (col[col != 0] * c.sign[pre] > 0).all()
    assert not c.mask[:, c.layer_slice("descending")].any(), "descending neurons have no brain outputs"


def test_sparse_step_matches_dense():
    spec = BrainSpec(build_default_connectome())
    rng = np.random.default_rng(1)
    W, b = spec.weights(spec.random_genomes(3, rng))
    r = rng.random((3, spec.n)).astype(np.float32)
    s = r[:, : spec.n_sens]
    out = spec.step(W, b, r, s)
    x = np.stack([spec.dense(W[i]) @ r[i] for i in range(3)]) + b
    ref = (1 - spec.leak) * r + spec.leak * np.tanh(np.maximum(x, 0))
    ref[:, : spec.n_sens] = s
    assert np.abs(out - ref).max() < 1e-5


def test_fight_runs_and_damage_is_consistent():
    spec = BrainSpec(build_default_connectome())
    rng = np.random.default_rng(2)
    W, b = spec.weights(spec.random_genomes(8, rng))
    res = fight(spec, np.stack([W[:4], W[4:]]), np.stack([b[:4], b[4:]]), rng, rounds=(200,),
                bot=np.array([[False] * 4, [True] * 4]))
    assert np.allclose(res.dealt[0], res.taken[1]) and np.allclose(res.dealt[1], res.taken[0])
    assert (res.taken <= MAX_HEALTH + 10).all()
    assert set(np.unique(res.winner())) <= {-1, 0, 1}


def test_flywire_loader(tmp_path):
    types = tmp_path / "types.csv.gz"
    conns = tmp_path / "conn.csv"
    with gzip.open(types, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_id", "primary_type"])
        for i, t in enumerate(["LC4", "LC4", "DNp01", "DNp01", "PFL3", "DNa02", "P1_a", "DNp04"]):
            w.writerow([100 + i, t])
    with open(conns, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"])
        w.writerow([100, 102, "LO_L", 40, "ACH"])
        w.writerow([104, 105, "LAL_L", 12, "ACH"])
        w.writerow([106, 107, "SMP_L", 9, "GABA"])
    c = load_flywire(str(conns), str(types))
    assert c.n == build_default_connectome().n
    assert "FlyWire" in c.source
    assert (c.sign[c.idx("P1")] < 0).all()  # P1 took its sign from the (fake) GABA vote
