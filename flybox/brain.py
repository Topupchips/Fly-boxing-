"""
A rate-based fly brain that runs many flies at once.

Each fly's brain is the same Connectome (same wires, same transmitter signs);
what differs between flies is its genome: the strength of every existing
synapse plus a resting bias for each non-sensory neuron. Evolution tunes the
genome, the wiring stays fly-shaped.

Dynamics (per 1/30 s step):
    r_sensory  = sensory input (clamped, 0..1)
    x          = W @ r + bias
    r          = (1 - leak) * r + leak * tanh(relu(x))      # firing rates >= 0
"""
from __future__ import annotations

import numpy as np

from .connectome import Connectome

LEAK = {"sensory": 1.0, "central": 0.45, "descending": 0.7}


class BrainSpec:
    """Precomputed indexing for turning genomes into weight matrices."""

    def __init__(self, conn: Connectome):
        self.conn = conn
        self.post, self.pre = np.nonzero(conn.mask)
        self.n_syn = len(self.post)
        sens = conn.layer_slice("sensory")
        self.n_sens = sens.stop - sens.start
        self.n = conn.n
        self.n_bias = self.n - self.n_sens
        self.genome_size = self.n_syn + self.n_bias
        self.wsign = conn.sign[self.pre].astype(np.float32)
        # synapses come sorted by postsynaptic neuron, so summing each neuron's
        # inputs is one np.add.reduceat over contiguous runs
        self.targets, self.starts = np.unique(self.post, return_index=True)
        self.leak = np.ones(self.n, dtype=np.float32)
        for g in conn.groups:
            self.leak[conn.idx(g.name)] = LEAK[g.layer]
        s = conn.slices
        self.out = {k: s[k] for k in ["DNa02", "DNp09", "MDN", "DN_strafe", "GF", "DN_lunge", "DN_guard"]}

    def random_genomes(self, count: int, rng: np.random.Generator) -> np.ndarray:
        init = self.conn.init_strength[self.post, self.pre]
        syn = init[None, :] * rng.lognormal(0.0, 0.5, (count, self.n_syn))
        bias = rng.normal(0.0, 0.3, (count, self.n_bias))
        # descending neurons start slightly below threshold so newborn flies aren't hyperactive
        d0 = self.conn.layer_slice("descending").start - self.n_sens
        bias[:, d0:] -= 0.3
        return np.concatenate([syn, bias], axis=1).astype(np.float32)

    def dense(self, w_row: np.ndarray) -> np.ndarray:
        """One fly's signed weights as an (N, N) matrix [post, pre]."""
        M = np.zeros((self.n, self.n), dtype=np.float32)
        M[self.post, self.pre] = w_row
        return M

    def weights(self, genomes: np.ndarray):
        """genomes (P, G) -> signed synapse weights W (P, S), bias (P, N)."""
        P = genomes.shape[0]
        W = (np.abs(genomes[:, : self.n_syn]) * self.wsign).astype(np.float32)
        b = np.zeros((P, self.n), dtype=np.float32)
        b[:, self.n_sens:] = genomes[:, self.n_syn:]
        return W, b

    def step(self, W, b, r, sensory):
        """One tick for a batch of brains. W (B,S), b (B,N), r (B,N), sensory (B,n_sens)."""
        r = r.copy()
        r[:, : self.n_sens] = sensory
        x = b.copy()
        x[:, self.targets] += np.add.reduceat(W * r[:, self.pre], self.starts, axis=1)
        new = np.tanh(np.maximum(x, 0.0))
        r = (1.0 - self.leak) * r + self.leak * new
        r[:, : self.n_sens] = sensory
        return r

    def motor(self, r):
        """Read descending-neuron rates as fighting commands."""
        o = self.out
        dna, strafe = r[:, o["DNa02"]], r[:, o["DN_strafe"]]
        gf = r[:, o["GF"]]
        return {
            "turn": dna[:, 0] - dna[:, 1],                      # left DNa02 turns left (+, CCW)
            "forward": r[:, o["DNp09"]][:, 0] - 0.6 * r[:, o["MDN"]][:, 0],
            "strafe": strafe[:, 0] - strafe[:, 1],              # + = step left
            "lunge": r[:, o["DN_lunge"]].mean(1) > 0.5,
            "guard": r[:, o["DN_guard"]].mean(1) > 0.55,
            "dodge": gf.max(1) > 0.6,
            "dodge_dir": np.where(gf[:, 0] >= gf[:, 1], -1.0, 1.0),  # left GF looms -> hop right
        }
