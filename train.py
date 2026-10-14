"""
Train fly boxers and export fights for the browser ring.

    python train.py                          # default connectome, ~15 min on a laptop
    python train.py --generations 50         # quick run
    python train.py --flywire-connections connections.csv.gz \
                    --flywire-types consolidated_cell_types.csv.gz

Outputs
    models/champion.npz     best genome + the wiring it runs on
    models/history.json     fitness curve
    docs/fights.js          replays + brain map for docs/index.html (GitHub Pages)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from flybox.arena import fight
from flybox.brain import BrainSpec
from flybox.connectome import build_default_connectome, load_flywire
from flybox.evolve import Config, train

ROOT = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Iron Mel", "Kid Lunge", "The Giant Fiber", "Buzzsaw", "Proboscis Pete", "Six-Leg Sally",
         "Compound Eye Carl", "Wingnut", "Haltere Harry", "Larva Loca", "Thorax Thompson", "Fruit Punch"]


def replay(spec, genomes, rng, rounds=(450, 450, 450), bot_b=False):
    """Record one 3-round fight between genomes[0] and genomes[1]."""
    W, b = spec.weights(genomes)
    Wp = np.stack([W[0:1], W[1:2]])
    bp = np.stack([b[0:1], b[1:2]])
    bot = np.array([[False], [bot_b]])
    res = fight(spec, Wp, bp, rng, rounds=rounds, bot=bot, randomize=False, record=True)
    frames = [f[0] for f in res.frames]
    w = int(res.winner()[0])
    return {
        "frames": frames,
        "winner": w,
        "ko": bool(res.ko[0, 0] or res.ko[1, 0]),
        "stats": [{k: float(getattr(res, k)[s, 0]) for k in ["dealt", "taken", "hits", "blocks", "dodges"]} for s in (0, 1)],
        "round_len": list(rounds),
    }


def brain_map(spec):
    c = spec.conn
    return {
        "groups": [{"name": g.name, "layer": g.layer, "size": g.size, "nt": g.nt, "about": g.about,
                    "start": c.idx(g.name).start} for g in c.groups],
        "edges": [[int(p), int(q), int(c.sign[p] > 0)] for q, p in zip(spec.post, spec.pre)],  # same order as genome synapses
        "leak": [float(v) for v in spec.leak],
        "n_sens": spec.n_sens,
        "out": {k: [v.start, v.stop] for k, v in spec.out.items()},
        "source": c.source,
        "n": c.n,
        "synapses": c.n_synapses,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flywire-connections")
    ap.add_argument("--flywire-types")
    args = ap.parse_args()

    if args.flywire_connections and args.flywire_types:
        conn = load_flywire(args.flywire_connections, args.flywire_types, seed=args.seed)
    else:
        conn = build_default_connectome(seed=7)
    print(conn.summary())
    spec = BrainSpec(conn)
    cfg = Config(pop=args.pop, generations=args.generations, steps=args.steps, seed=args.seed,
                 peers=2, hall=2, bot=3)

    def log(g, fit, bot_wr, hits, dt):
        if g % 5 == 0 or g == cfg.generations - 1:
            print(f"gen {g:4d}  best {fit.max():7.1f}  mean {fit.mean():7.1f}  "
                  f"beats bot {bot_wr * 100:5.1f}%  hits/fight {hits:4.2f}  ({dt:.1f}s)", flush=True)

    ranked, hof, hist, snaps = train(spec, cfg, on_gen=log)

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "models", "champion.npz"), genome=ranked[0], runner_up=ranked[1],
                        mask=conn.mask, sign=conn.sign, source=conn.source)
    with open(os.path.join(ROOT, "models", "history.json"), "w") as f:
        json.dump(hist.__dict__, f)

    rng = np.random.default_rng(args.seed + 1)
    names = list(rng.permutation(NAMES))
    gen0 = snaps[0]
    mid_gen = sorted(k for k in snaps if 0 < k < cfg.generations - 1)
    mid_gen = mid_gen[len(mid_gen) // 3] if mid_gen else 0
    champ, runner = ranked[0], ranked[1]
    card = [
        ("Title Fight", f"Champion vs #2 contender, generation {cfg.generations}", champ, runner, names[0], names[1], False),
        ("Sparring Session", "Champion vs the hand-coded sparring bot", champ, champ, names[0], "Sparring Bot", True),
        ("Glow-Up", f"Champion vs the best fly from generation 0", champ, gen0[0], names[0], names[2], False),
        ("Comeback Trail", f"Champion vs its ancestor from generation {mid_gen}", champ, snaps[mid_gen][0], names[0], names[3], False),
        ("Newborns", "Two untrained flies from generation 0", gen0[0], gen0[1], names[2], names[4], False),
    ]
    fights = []
    for title, sub, a, b, na, nb, is_bot in card:
        rep = replay(spec, np.stack([a, b]), rng, bot_b=is_bot)
        rep.update(title=title, subtitle=sub, names=[na, nb])
        fights.append(rep)
        w = rep["names"][rep["winner"]] if rep["winner"] >= 0 else "draw"
        print(f"{title:18s} {na} vs {nb}: {w}{' by KO' if rep['ko'] else ''}")

    # roster for live mode: the browser runs these brains itself (docs/engine.js)
    picks = [(names[0], f"Champion, generation {cfg.generations}", champ),
             (names[1], f"#2 contender, generation {cfg.generations}", runner)]
    for k, nm in zip(range(2, 5), (names[5], names[6], names[7])):
        picks.append((nm, f"#{k + 1} contender, generation {cfg.generations}", ranked[k]))
    for g, nm in ((mid_gen, names[3]), (sorted(snaps)[-2] if len(snaps) > 2 else mid_gen, names[8])):
        picks.append((nm, f"Champion of generation {g}", snaps[g][0]))
    picks += [(names[2], "Best newborn, generation 0", gen0[0]), (names[4], "Newborn, generation 0", gen0[1])]
    roster = []
    seen = set()
    for nm, about, g in picks:
        if nm in seen:
            continue
        seen.add(nm)
        W, b = spec.weights(g[None])
        roster.append({"name": nm, "about": about,
                       "w": [round(float(v), 4) for v in W[0]],
                       "b": [round(float(v), 4) for v in b[0]]})
    roster.append({"name": "Sparring Bot", "about": "Hand-coded opponent, no brain", "bot": True})

    data = {"fights": fights, "brain": brain_map(spec), "history": hist.__dict__, "roster": roster,
            "config": {"generations": cfg.generations, "pop": cfg.pop}}
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "fights.js"), "w") as f:
        f.write("window.FLYBOX = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("wrote docs/fights.js - run `python -m http.server -d docs` and open http://localhost:8000")


if __name__ == "__main__":
    main()
