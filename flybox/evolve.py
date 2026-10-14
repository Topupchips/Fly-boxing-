"""
Training: self-play neuroevolution.

Every generation each fly fights rival flies from its own population, past
champions from the Hall of Fame, and a hand-written sparring bot. Fitness
rewards damage landed, punishes damage taken, and pays a bonus for a knockout.
The best flies breed (uniform crossover) and mutate their synapse strengths;
the wiring itself never changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .arena import fight
from .brain import BrainSpec


@dataclass
class Config:
    pop: int = 64
    generations: int = 300
    peers: int = 3            # fights vs own population per fly
    hall: int = 2             # fights vs Hall of Fame per fly
    bot: int = 2              # fights vs sparring bot per fly
    steps: int = 600          # 20 s bouts
    elite: int = 6
    tourney: int = 4
    mut_rate: float = 0.08
    mut_sigma: float = 0.25
    seed: int = 0


@dataclass
class History:
    gen: list = field(default_factory=list)
    best: list = field(default_factory=list)
    mean: list = field(default_factory=list)
    bot_winrate: list = field(default_factory=list)
    hits: list = field(default_factory=list)


def fitness_of(res, side):
    o = 1 - side
    f = res.dealt[side] - 0.8 * res.taken[side] + 40.0 * res.ko[side]
    f = f + 0.01 * res.close[side] + 0.02 * res.aim[side]   # nudge: get close, face the rival
    return f, res.winner() == side


def evaluate(spec, genomes, hof, rng, cfg):
    P = len(genomes)
    bank = np.concatenate([genomes, hof]) if len(hof) else genomes
    W, b = spec.weights(bank)
    A, Bs, bot_b = [], [], []
    for i in range(P):
        for j in rng.choice([k for k in range(P) if k != i], cfg.peers, replace=False):
            A.append(i); Bs.append(j); bot_b.append(False)
        for _ in range(cfg.hall if len(hof) else 0):
            A.append(i); Bs.append(P + rng.integers(len(hof))); bot_b.append(False)
        for _ in range(cfg.bot):
            A.append(i); Bs.append(i); bot_b.append(True)
    A, Bs, bot_b = map(np.asarray, (A, Bs, bot_b))
    bot = np.stack([np.zeros_like(bot_b), bot_b])
    res = fight(spec, np.stack([W[A], W[Bs]]), np.stack([b[A], b[Bs]]), rng, rounds=(cfg.steps,), bot=bot)
    fit = np.zeros(P); cnt = np.zeros(P)
    fa, wa = fitness_of(res, 0)
    fb, _ = fitness_of(res, 1)
    np.add.at(fit, A, fa); np.add.at(cnt, A, 1)
    peer = (~bot_b) & (Bs < P)
    np.add.at(fit, Bs[peer], fb[peer]); np.add.at(cnt, Bs[peer], 1)
    bot_wr = wa[bot_b].mean()
    return fit / cnt, bot_wr, res.hits[0].sum() / max(len(A), 1)


def breed(fit, genomes, rng, cfg, sigma_scale):
    P, G = genomes.shape
    order = np.argsort(-fit)
    kids = [genomes[order[:cfg.elite]]]
    n = P - cfg.elite

    def pick():
        c = rng.choice(P, cfg.tourney, replace=False)
        return genomes[c[np.argmax(fit[c])]]

    pa = np.stack([pick() for _ in range(n)])
    pb = np.stack([pick() for _ in range(n)])
    child = np.where(rng.random((n, G)) < 0.5, pa, pb)
    m = rng.random((n, G)) < cfg.mut_rate
    scale = np.maximum(np.abs(child), 0.1)
    child = child + m * rng.normal(0, cfg.mut_sigma * sigma_scale, (n, G)) * scale
    kids.append(child.astype(np.float32))
    return np.concatenate(kids), genomes[order[0]].copy()


def train(spec: BrainSpec, cfg: Config, on_gen=None, snapshot_every=25):
    rng = np.random.default_rng(cfg.seed)
    genomes = spec.random_genomes(cfg.pop, rng)
    hof = np.zeros((0, spec.genome_size), np.float32)
    hist = History()
    snapshots = {}
    for g in range(cfg.generations):
        t0 = time.time()
        fit, bot_wr, hits = evaluate(spec, genomes, hof, rng, cfg)
        sigma_scale = 1.0 if g < cfg.generations * 0.6 else 0.5
        if g == 0:
            snapshots[0] = genomes[np.argsort(-fit)[:2]].copy()
        genomes, champ = breed(fit, genomes, rng, cfg, sigma_scale)
        if g % 5 == 0:
            hof = np.concatenate([hof, champ[None]])[-20:]
        if (g > 0 and g % snapshot_every == 0) or g == cfg.generations - 1:
            snapshots[g] = champ[None]
        hist.gen.append(g); hist.best.append(float(fit.max())); hist.mean.append(float(fit.mean()))
        hist.bot_winrate.append(float(bot_wr)); hist.hits.append(float(hits))
        if on_gen:
            on_gen(g, fit, bot_wr, hits, time.time() - t0)
    # final ranking of the last population by a fresh round-robin-ish evaluation
    fit, _, _ = evaluate(spec, genomes, hof, rng, cfg)
    order = np.argsort(-fit)
    return genomes[order], hof, hist, snapshots
