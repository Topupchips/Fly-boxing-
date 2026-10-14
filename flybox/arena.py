"""
The boxing ring: a top-down 2D physics sim that runs many fights in parallel.

Distances in mm, time step 1/30 s. A fruit fly is ~2.5 mm long, so the
28 mm ring is roughly a petri-dish-sized fighting arena.

Moves available to a fly (all driven by descending neurons):
  walk / back up / side-step / turn
  LUNGE  - rear up and slam forward with the forelegs (real Drosophila do this)
  GUARD  - raise the forelegs; blocks most of a lunge that comes from the front
  DODGE  - giant-fiber escape hop away from a looming attack
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brain import BrainSpec

DT = 1 / 30
HALF = 14.0          # ring half-width (mm)
BODY_R = 1.2         # body radius (mm)
REACH = 3.8          # centre-to-centre distance a lunge can connect at
STRIKE_CONE = 0.55   # rad either side of the head
BLOCK_CONE = 0.9
LUNGE_T, LUNGE_HIT, LUNGE_CD, LUNGE_COST = 5, 3, 14, 0.14
DODGE_T, DODGE_CD, DODGE_COST = 4, 24, 0.22
DAMAGE, BLOCKED_DAMAGE = 10.0, 2.0
MAX_HEALTH = 100.0

LC10_CENTRES = np.deg2rad(np.arange(-157.5, 158, 45.0))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FightResult:
    dealt: np.ndarray        # (2, B) damage dealt by each side
    taken: np.ndarray
    hits: np.ndarray
    blocks: np.ndarray
    dodges: np.ndarray
    close: np.ndarray        # steps spent within striking distance
    aim: np.ndarray          # ... of which spent facing the rival
    ko: np.ndarray           # (2, B) bool: this side won by KO
    steps: np.ndarray        # (B,) steps fought
    frames: list | None = None

    def winner(self):
        """0 / 1 / -1 (draw) per fight: KO first, then remaining health."""
        hp = MAX_HEALTH - self.taken
        w = np.where(hp[0] > hp[1] + 1e-6, 0, np.where(hp[1] > hp[0] + 1e-6, 1, -1))
        w = np.where(self.ko[0], 0, np.where(self.ko[1], 1, w))
        return w


def sparring_bot(st, o, s, rng):
    """A hand-written boxer used as a steady training partner."""
    dx, dy = st["x"][o] - st["x"][s], st["y"][o] - st["y"][s]
    d = np.hypot(dx, dy)
    bear = wrap(np.arctan2(dy, dx) - st["h"][s])
    opp_lunging = st["lunge_t"][o] > 0
    B = d.shape[0]
    return {
        "turn": np.clip(bear * 2.0, -1, 1),
        "forward": np.where(d > 3.2, 0.8, np.where(d < 2.6, -0.4, 0.0)),
        "strafe": np.sin(st["t"] * 0.07 + np.arange(B)) * 0.5,
        "lunge": (d < REACH - 0.2) & (np.abs(bear) < 0.4) & (rng.random(B) < 0.35),
        "guard": opp_lunging & (d < 5) & (rng.random(B) < 0.5),
        "dodge": np.zeros(B, bool),
        "dodge_dir": np.ones(B),
    }


def _sensory(st, s, prev_ang):
    o = 1 - s
    dx, dy = st["x"][o] - st["x"][s], st["y"][o] - st["y"][s]
    d = np.maximum(np.hypot(dx, dy), 1e-3)
    bear = wrap(np.arctan2(dy, dx) - st["h"][s])
    ang = 2 * np.arctan(BODY_R / d)
    loom = np.clip((ang - prev_ang[s]) * 25.0, 0, 1)
    visible = np.abs(bear) < np.deg2rad(165)
    size = np.clip(ang * 1.5, 0, 1) * visible
    lc10 = size[:, None] * np.exp(-wrap(bear[:, None] - LC10_CENTRES[None]) ** 2 / (2 * 0.35 ** 2))
    left = bear > 0
    lc4 = np.stack([loom * left, loom * ~left], 1) * visible[:, None]
    burst = ((st["lunge_t"][o] > 0) | (st["dodge_t"][o] > 0)).astype(np.float32) * visible
    lc16 = np.stack([burst * left, burst * ~left], 1)
    hit = st["was_hit"][s]
    hb = st["hit_bear"][s]
    ms = np.stack([hit * (hb > 0), hit * (hb <= 0)], 1)
    # bristles: how close the ropes are ahead / left / right / behind
    br = []
    for off in (0.0, np.pi / 2, -np.pi / 2, np.pi):
        a = st["h"][s] + off
        c, sn = np.cos(a), np.sin(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(c > 1e-6, (HALF - st["x"][s]) / c, np.where(c < -1e-6, (-HALF - st["x"][s]) / c, np.inf))
            ty = np.where(sn > 1e-6, (HALF - st["y"][s]) / sn, np.where(sn < -1e-6, (-HALF - st["y"][s]) / sn, np.inf))
        br.append(np.clip(1 - (np.minimum(tx, ty) - BODY_R) / 4.0, 0, 1))
    cva = np.exp(-d / 6.0)
    intero = np.stack([st["stamina"][s], st["health"][s] / MAX_HEALTH, (st["lunge_cd"][s] == 0).astype(np.float32)], 1)
    sens = np.concatenate([lc10, lc4, lc16, ms, np.stack(br, 1), cva[:, None], intero], 1)
    return sens.astype(np.float32), ang, d, bear


def _starts(B, rng, randomize):
    if not randomize:
        j = rng.normal(0, 0.4, (2, B))
        x = np.stack([np.full(B, -9.0), np.full(B, 9.0)]) + j
        y = np.stack([np.full(B, -9.0), np.full(B, 9.0)]) + rng.normal(0, 0.4, (2, B))
        h = np.stack([np.full(B, np.pi / 4), np.full(B, -3 * np.pi / 4)]) + rng.normal(0, 0.2, (2, B))
        return x, y, h
    x = rng.uniform(-HALF + 3, HALF - 3, (2, B))
    y = rng.uniform(-HALF + 3, HALF - 3, (2, B))
    for _ in range(20):  # keep them apart at the bell
        close = np.hypot(x[0] - x[1], y[0] - y[1]) < 9
        if not close.any():
            break
        x[1, close] = rng.uniform(-HALF + 3, HALF - 3, close.sum())
        y[1, close] = rng.uniform(-HALF + 3, HALF - 3, close.sum())
    h = rng.uniform(-np.pi, np.pi, (2, B))
    return x, y, h


def fight(spec: BrainSpec, W, b, rng, rounds=(600,), bot=None, randomize=True, record=False):
    """Run B fights at once.

    W (2, B, S), b (2, B, N): brains (synapse weights, biases) for side 0 and side 1.
    bot (2, B) bool: replace that side with the sparring bot.
    rounds: steps per round; positions reset between rounds, damage carries over.
    """
    B = W.shape[1]
    N = spec.n
    bot = np.zeros((2, B), bool) if bot is None else bot
    Wf, bf = W.reshape(2 * B, -1), b.reshape(2 * B, N)
    r = np.zeros((2 * B, N), np.float32)

    z = lambda v=0.0: np.full((2, B), v, np.float32)
    st = dict(health=z(MAX_HEALTH), stamina=z(1.0), lunge_t=z(), lunge_cd=z(), dodge_t=z(),
              dodge_cd=z(), dodge_dir=z(1.0), guard=np.zeros((2, B), bool), was_hit=z(), hit_bear=z(), t=0)
    stats = {k: z() for k in ["dealt", "taken", "hits", "blocks", "dodges", "close", "aim"]}
    ko = np.zeros((2, B), bool)
    done = np.zeros(B, bool)
    steps = np.zeros(B, np.int32)
    frames = [] if record else None

    for ri, n_steps in enumerate(rounds):
        x, y, h = _starts(B, rng, randomize)
        st.update(x=x.astype(np.float32), y=y.astype(np.float32), h=h.astype(np.float32))
        st["stamina"][:] = 1.0
        for k in ("lunge_t", "dodge_t", "lunge_cd", "dodge_cd"):
            st[k][:] = 0
        r[:] = 0
        prev_ang = z(2 * np.arctan(BODY_R / 18))
        for t in range(n_steps):
            if done.all():
                break
            st["t"] += 1
            sens, ang, dist, bear = zip(*[_sensory(st, s, prev_ang) for s in (0, 1)])
            prev_ang = np.stack(ang)
            r = spec.step(Wf, bf, r, np.concatenate(sens))
            m = spec.motor(r)
            act = {k: np.stack([v[:B], v[B:]]) for k, v in m.items()}
            for s in (0, 1):
                if bot[s].any():
                    ba = sparring_bot(st, 1 - s, s, rng)
                    for k in act:
                        act[k][s] = np.where(bot[s], ba[k], act[k][s])

            live = ~done[None, :]
            # --- start moves ---
            busy = (st["lunge_t"] > 0) | (st["dodge_t"] > 0)
            start_l = act["lunge"] & (st["lunge_cd"] == 0) & (st["stamina"] >= LUNGE_COST) & ~busy & live
            start_d = act["dodge"] & (st["dodge_cd"] == 0) & (st["stamina"] >= DODGE_COST) & ~busy & ~start_l & live
            st["lunge_t"] = np.where(start_l, LUNGE_T, st["lunge_t"])
            st["lunge_cd"] = np.where(start_l, LUNGE_CD, st["lunge_cd"])
            st["stamina"] -= start_l * LUNGE_COST + start_d * DODGE_COST
            st["dodge_t"] = np.where(start_d, DODGE_T, st["dodge_t"])
            st["dodge_cd"] = np.where(start_d, DODGE_CD, st["dodge_cd"])
            st["dodge_dir"] = np.where(start_d, act["dodge_dir"], st["dodge_dir"])
            stats["dodges"] += start_d
            lunging, dodging = st["lunge_t"] > 0, st["dodge_t"] > 0
            st["guard"] = act["guard"] & ~lunging & ~dodging & live

            # --- movement ---
            turn = np.clip(act["turn"], -1, 1) * np.where(lunging, 1.5, 6.0)
            st["h"] = wrap(st["h"] + turn * DT * live)
            fwd = np.clip(act["forward"], -1, 1)
            fwd = np.where(fwd > 0, fwd * 20.0, fwd * 9.0)
            side = np.clip(act["strafe"], -1, 1) * 9.0
            slow = np.where(st["guard"], 0.45, 1.0)
            fwd, side = fwd * slow, side * slow
            fwd = np.where(lunging, 42.0, fwd)
            side = np.where(lunging, 0.0, side)
            fwd = np.where(dodging, -30.0, fwd)
            side = np.where(dodging, 45.0 * st["dodge_dir"], side)
            c, s_ = np.cos(st["h"]), np.sin(st["h"])
            st["x"] += (c * fwd - s_ * side) * DT * live
            st["y"] += (s_ * fwd + c * side) * DT * live

            # --- strikes ---
            st["was_hit"][:] = 0
            st["blk"] = np.zeros((2, B), bool)
            for a in (0, 1):
                d_ = 1 - a
                strike = (st["lunge_t"][a] == LUNGE_HIT) & ~done
                dx, dy = st["x"][d_] - st["x"][a], st["y"][d_] - st["y"][a]
                dd = np.hypot(dx, dy)
                ab = wrap(np.arctan2(dy, dx) - st["h"][a])
                connects = strike & (dd < REACH) & (np.abs(ab) < STRIKE_CONE) & (st["dodge_t"][d_] == 0)
                db = wrap(np.arctan2(-dy, -dx) - st["h"][d_])
                blocked = connects & st["guard"][d_] & (np.abs(db) < BLOCK_CONE)
                dmg = np.where(blocked, BLOCKED_DAMAGE, DAMAGE) * connects
                st["health"][d_] -= dmg
                st["stamina"][a] -= blocked * 0.1
                stats["dealt"][a] += dmg
                stats["taken"][d_] += dmg
                stats["hits"][a] += connects & ~blocked
                stats["blocks"][d_] += blocked
                st["blk"][d_] |= blocked
                st["was_hit"][d_] = np.maximum(st["was_hit"][d_], connects.astype(np.float32))
                st["hit_bear"][d_] = np.where(connects, db, st["hit_bear"][d_])
                kb = np.where(blocked, 0.6, 1.6) * connects / np.maximum(dd, 1e-3)
                st["x"][d_] += dx * kb
                st["y"][d_] += dy * kb

            # --- bodies and ropes ---
            dx, dy = st["x"][1] - st["x"][0], st["y"][1] - st["y"][0]
            dd = np.maximum(np.hypot(dx, dy), 1e-3)
            push = np.maximum(2 * BODY_R - dd, 0) / 2 / dd
            st["x"][0] -= dx * push; st["y"][0] -= dy * push
            st["x"][1] += dx * push; st["y"][1] += dy * push
            lim = HALF - BODY_R
            st["x"] = np.clip(st["x"], -lim, lim)
            st["y"] = np.clip(st["y"], -lim, lim)

            # --- timers / stamina ---
            st["lunge_t"] = np.maximum(st["lunge_t"] - 1, 0)
            st["dodge_t"] = np.maximum(st["dodge_t"] - 1, 0)
            st["lunge_cd"] = np.maximum(st["lunge_cd"] - 1, 0)
            st["dodge_cd"] = np.maximum(st["dodge_cd"] - 1, 0)
            st["stamina"] = np.clip(st["stamina"] + 0.009 - st["guard"] * 0.006, 0, 1)
            near = (dd < REACH + 1.5)[None, :] & live
            stats["close"] += near
            stats["aim"] += near & (np.abs(np.stack(bear)) < 0.5)
            steps += ~done

            if record:
                frames.append(_frame(st, r, B, start_l, start_d))

            kod = (st["health"] <= 0) & live
            ko[0] |= kod[1]
            ko[1] |= kod[0]
            done |= kod.any(0)

    return FightResult(stats["dealt"], stats["taken"], stats["hits"], stats["blocks"],
                       stats["dodges"], stats["close"], stats["aim"], ko, steps, frames)


def _frame(st, r, B, start_l, start_d):
    """Snapshot of every fight for replays (brain rates as 0-9 digit strings)."""
    digits = np.clip((r * 10).astype(int), 0, 9)
    out = []
    for f in range(B):
        flies = []
        for s in (0, 1):
            flies.append({
                "x": round(float(st["x"][s, f]), 2), "y": round(float(st["y"][s, f]), 2),
                "h": round(float(st["h"][s, f]), 3),
                "hp": round(float(max(st["health"][s, f], 0)), 1), "st": round(float(st["stamina"][s, f]), 2),
                "l": int(st["lunge_t"][s, f] > 0), "g": int(st["guard"][s, f]), "d": int(st["dodge_t"][s, f] > 0),
                "hit": int(st["was_hit"][s, f] > 0), "blk": int(st["blk"][s, f]),
                "b": "".join(map(str, digits[s * B + f])),
            })
        out.append(flies)
    return out
