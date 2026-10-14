"""
Fly brain wiring for Fly Boxing.

The network is organised into neuron groups named after real Drosophila
cell types that matter for fighting and steering:

  sensory   LC10 (small-object columns), LC4 (looming -> giant fiber),
            LC16 (sudden motion), mechanosensory hit, wall bristles,
            Or67d cVA pheromone ORN, interoception (stamina / health)
  central   PFL3 (steering), fan-shaped body, P1 (aggression / arousal),
            lateral horn, Kenyon cells, MBONs, inhibitory local neurons
  descending DNa02 (turning), DNp09 (walk forward), MDN (moonwalk back),
            DN_strafe, Giant Fiber (escape dodge), DN_lunge, DN_guard

Two ways to build it:

  build_default_connectome()  - a scaled-down wiring whose block structure
                                follows the known pathways above (no download).
  load_flywire(...)           - derive the block densities and transmitter
                                signs from FlyWire Codex CSV exports.

Either way the result is a Connectome: a fixed binary wiring mask plus a sign
per presynaptic neuron (Dale's law). Training only tunes synaptic *strengths*
on existing wires; it never adds a wire the connectome doesn't have.
"""
from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass, field

import numpy as np

ACH, GABA, GLUT, MIXED = "ACh", "GABA", "Glu", "mixed"


@dataclass
class Group:
    name: str
    size: int
    layer: str          # sensory | central | descending
    nt: str             # dominant transmitter
    about: str
    flywire: str = ""   # regex matched against FlyWire cell types


GROUPS: list[Group] = [
    # ---- sensory ----
    Group("LC10", 8, "sensory", ACH, "visual small-object columns tuned to opponent bearing", r"^LC10"),
    Group("LC4", 2, "sensory", ACH, "looming detectors L/R, feed the giant fiber", r"^LC4$"),
    Group("LC16", 2, "sensory", ACH, "sudden-motion detectors: opponent lunging, L/R", r"^LC16$"),
    Group("MS_hit", 2, "sensory", ACH, "mechanosensory: got struck on left / right", r"^JO"),
    Group("BR_wall", 4, "sensory", ACH, "bristles touching ropes: front / left / right / back", r"^BM_"),
    Group("ORN_cVA", 1, "sensory", ACH, "Or67d neurons smelling rival cVA pheromone", r"^ORN_DA1"),
    Group("INT", 3, "sensory", ACH, "interoception: stamina, health, strike ready", r""),
    # ---- central brain ----
    Group("PFL3", 4, "central", ACH, "central complex steering cells", r"^PFL3"),
    Group("FB", 12, "central", MIXED, "fan-shaped body integrators", r"^FB\d"),
    Group("P1", 6, "central", ACH, "P1 aggression / arousal neurons", r"^(P1_|pC1)"),  # pC1: female counterparts (FlyWire is female)
    Group("LH", 8, "central", MIXED, "lateral horn innate behaviour", r"^LH"),
    Group("KC", 16, "central", ACH, "mushroom body Kenyon cells", r"^KC"),
    Group("MBON", 6, "central", MIXED, "mushroom body output neurons", r"^MBON"),
    Group("LN", 8, "central", GABA, "inhibitory local neurons", r"^(LN|APL)"),
    # ---- descending (motor commands) ----
    Group("DNa02", 2, "descending", ACH, "turn left / right", r"^DNa02$"),
    Group("DNp09", 1, "descending", ACH, "walk forward", r"^DNp09$"),
    Group("MDN", 1, "descending", ACH, "moonwalker: back up", r"^MDN$"),
    Group("DN_strafe", 2, "descending", ACH, "side-step left / right", r"^DNa01$"),
    Group("GF", 2, "descending", ACH, "giant fiber escape hop (dodge) L/R", r"^(DNp01|GF)$"),
    Group("DN_lunge", 2, "descending", ACH, "aggressive lunge / punch", r"^DNp(04|11)$"),
    Group("DN_guard", 2, "descending", ACH, "raise forelegs to block", r"^DNg(11|12)$"),
]

# Which group-to-group pathways exist, and how dense they are.
# Densities are fractions of possible pre->post neuron pairs that get a synapse.
PATHWAYS: dict[tuple[str, str], float] = {
    # vision
    ("LC10", "PFL3"): 0.7, ("LC10", "DNa02"): 0.6, ("LC10", "P1"): 0.4,
    ("LC10", "FB"): 0.4, ("LC10", "KC"): 0.3, ("LC10", "DN_strafe"): 0.4, ("LC10", "DN_lunge"): 0.3,
    ("LC4", "GF"): 0.9, ("LC4", "LH"): 0.5, ("LC4", "DN_guard"): 0.5,
    ("LC16", "GF"): 0.6, ("LC16", "DN_guard"): 0.6, ("LC16", "LH"): 0.5, ("LC16", "MDN"): 0.5,
    # touch / smell / interoception
    ("MS_hit", "LN"): 0.5, ("MS_hit", "P1"): 0.6, ("MS_hit", "LH"): 0.5, ("MS_hit", "DN_guard"): 0.5,
    ("BR_wall", "DNa02"): 0.5, ("BR_wall", "MDN"): 0.5, ("BR_wall", "DNp09"): 0.5,
    ("BR_wall", "FB"): 0.4, ("BR_wall", "DN_strafe"): 0.5,
    ("ORN_cVA", "P1"): 1.0, ("ORN_cVA", "LH"): 1.0, ("ORN_cVA", "KC"): 0.5,
    ("INT", "FB"): 0.5, ("INT", "MBON"): 0.5, ("INT", "P1"): 0.6, ("INT", "DN_lunge"): 0.6,
    ("INT", "GF"): 0.5,
    # central processing
    ("KC", "MBON"): 0.4, ("MBON", "FB"): 0.4, ("MBON", "P1"): 0.4, ("MBON", "LH"): 0.3,
    ("FB", "FB"): 0.25, ("FB", "PFL3"): 0.5, ("FB", "P1"): 0.3, ("FB", "LN"): 0.3,
    ("LH", "P1"): 0.4, ("LH", "LN"): 0.3, ("LH", "FB"): 0.3,
    ("P1", "P1"): 0.3, ("P1", "FB"): 0.4, ("P1", "LN"): 0.3,
    ("LN", "FB"): 0.3, ("LN", "P1"): 0.3, ("LN", "LH"): 0.3, ("LN", "KC"): 0.3, ("LN", "PFL3"): 0.3,
    # descending outputs
    ("PFL3", "DNa02"): 0.9, ("PFL3", "DN_strafe"): 0.5,
    ("FB", "DNp09"): 0.4, ("FB", "MDN"): 0.3, ("FB", "DNa02"): 0.3, ("FB", "DN_strafe"): 0.3,
    ("P1", "DN_lunge"): 0.8, ("P1", "DNp09"): 0.6, ("P1", "DNa02"): 0.3,
    ("LH", "GF"): 0.4, ("LH", "DN_guard"): 0.4, ("LH", "MDN"): 0.4,
    ("MBON", "DN_lunge"): 0.3, ("MBON", "DN_guard"): 0.3, ("MBON", "DNp09"): 0.3,
    ("LN", "DN_lunge"): 0.4, ("LN", "DN_guard"): 0.4, ("LN", "GF"): 0.4,
    ("LN", "DNp09"): 0.3, ("LN", "MDN"): 0.3,
}

SIGN_OF = {ACH: 1.0, GABA: -1.0, GLUT: -1.0}


@dataclass
class Connectome:
    groups: list[Group]
    mask: np.ndarray            # (N, N) bool, mask[post, pre]
    sign: np.ndarray            # (N,)  +1 / -1 per presynaptic neuron
    init_strength: np.ndarray   # (N, N) starting |weight| for existing wires
    source: str = "default"
    slices: dict[str, slice] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.mask.shape[0]

    @property
    def n_synapses(self) -> int:
        return int(self.mask.sum())

    def idx(self, name: str) -> slice:
        return self.slices[name]

    def layer_slice(self, layer: str) -> slice:
        idx = [i for g in self.groups if g.layer == layer for i in range(self.slices[g.name].start, self.slices[g.name].stop)]
        return slice(min(idx), max(idx) + 1)

    def summary(self) -> str:
        lines = [f"Connectome ({self.source}): {self.n} neurons, {self.n_synapses} synapses"]
        for g in self.groups:
            lines.append(f"  {g.layer:10s} {g.name:9s} x{g.size:<3d} {g.nt:5s} {g.about}")
        return "\n".join(lines)


def _slices(groups: list[Group]) -> dict[str, slice]:
    out, i = {}, 0
    for g in groups:
        out[g.name] = slice(i, i + g.size)
        i += g.size
    return out


def _signs(groups: list[Group], rng: np.random.Generator) -> np.ndarray:
    signs = []
    for g in groups:
        if g.nt == MIXED:
            signs.extend(np.where(rng.random(g.size) < 0.7, 1.0, -1.0))
        else:
            signs.extend([SIGN_OF[g.nt]] * g.size)
    return np.asarray(signs, dtype=np.float32)


def _wire(groups, densities, signs, rng, source, strengths=None) -> Connectome:
    sl = _slices(groups)
    n = sum(g.size for g in groups)
    mask = np.zeros((n, n), dtype=bool)
    strength = np.zeros((n, n), dtype=np.float32)
    for (pre, post), p in densities.items():
        if pre not in sl or post not in sl or p <= 0:
            continue
        a, b = sl[post], sl[pre]
        block = rng.random((a.stop - a.start, b.stop - b.start)) < p
        if pre == post:
            np.fill_diagonal(block, False)
        mask[a, b] |= block
        s = 1.0 if strengths is None else strengths.get((pre, post), 1.0)
        strength[a, b] = np.where(block, s, strength[a, b])
    fan_in = np.maximum(mask.sum(1, keepdims=True), 1)
    init = strength * (1.5 / np.sqrt(fan_in))
    return Connectome(groups, mask, signs, init.astype(np.float32), source, sl)


def add_innate_topography(conn: Connectome) -> Connectome:
    """Lay down the left/right map every fly is born with.

    Flies hatch knowing how to orient: visual columns on the left side of the
    eye drive left-side steering neurons, a looming rival on the left fires the
    left giant fiber, and touching a rope ahead drives backing up. Here those
    ipsilateral wires are guaranteed to exist and start stronger; evolution can
    still weaken them or learn to use them differently.
    """
    sl, M, S = conn.slices, conn.mask, conn.init_strength
    cols = np.deg2rad(np.arange(-157.5, 158, 45.0))   # LC10 column centres (+ = left)

    def link(pre, i, post, j, s):
        a, b = sl[post].start + j, sl[pre].start + i
        M[a, b] = True
        S[a, b] = max(S[a, b], s)

    for i, th in enumerate(cols):
        left = th > 0
        link("LC10", i, "DNa02", 0 if left else 1, 0.9 * abs(np.sin(th)) + 0.3)
        link("LC10", i, "PFL3", (0 if left else 2) + int(abs(th) > 1.0), 0.8)
        if abs(th) < 0.9:                       # rival straight ahead -> lunge, arousal
            link("LC10", i, "DN_lunge", 0, 0.5)
            link("LC10", i, "DN_lunge", 1, 0.5)
            link("LC10", i, "P1", i % 6, 0.6)
        else:
            link("LC10", i, "DNp09", 0, 0.3)
    for p in range(4):
        link("PFL3", p, "DNa02", 0 if p < 2 else 1, 0.8)
    for side in (0, 1):
        link("LC4", side, "GF", side, 1.2)
        link("LC16", side, "DN_guard", side, 0.8)
        link("MS_hit", side, "DN_guard", side, 0.6)
    link("BR_wall", 0, "MDN", 0, 1.0)            # rope ahead -> back up
    link("BR_wall", 1, "DNa02", 1, 0.8)          # rope on the left -> turn right
    link("BR_wall", 2, "DNa02", 0, 0.8)
    link("ORN_cVA", 0, "DNp09", 0, 0.5)          # smell the rival -> advance
    return conn


def build_default_connectome(seed: int = 7) -> Connectome:
    """Scaled-down wiring following known fly fighting/steering pathways."""
    rng = np.random.default_rng(seed)
    conn = _wire(GROUPS, PATHWAYS, _signs(GROUPS, rng), rng, "default (FlyWire-inspired pathways)")
    return add_innate_topography(conn)


# ---------------------------------------------------------------------------
# FlyWire import
# ---------------------------------------------------------------------------

def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def _col(header: list[str], *names: str) -> int:
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    raise KeyError(f"none of {names} in columns {header}")


def load_flywire(connections_csv: str, cell_types_csv: str, seed: int = 7,
                 min_syn: int = 5, keep_default_paths: bool = True) -> Connectome:
    """Build the fly brain from FlyWire Codex exports.

    connections_csv  e.g. connections.csv(.gz) from codex.flywire.ai/api/download
                     (columns: pre_root_id, post_root_id, syn_count, nt_type)
    cell_types_csv   e.g. consolidated_cell_types.csv(.gz) or classification.csv(.gz)
                     (columns: root_id + primary_type / cell_type)

    For every pair of our groups we measure, in the real connectome, what
    fraction of neuron pairs share >= min_syn synapses and how strong those
    connections are. Those numbers set the wiring density and starting weight
    of the model, and each group's transmitter sign comes from the majority
    nt_type of its real neurons. Groups with no matching real cells fall back
    to the default pathway table.
    """
    rng = np.random.default_rng(seed)

    # 1. map real neurons -> our groups
    pats = {g.name: re.compile(g.flywire) for g in GROUPS if g.flywire}
    member: dict[int, str] = {}
    with _open(cell_types_csv) as f:
        r = csv.reader(f)
        h = next(r)
        ci = _col(h, "root_id")
        ct = _col(h, "primary_type", "cell_type", "hemibrain_type")
        for row in r:
            t = row[ct].strip()
            if not t:
                continue
            for name, pat in pats.items():
                if pat.search(t):
                    member[int(row[ci])] = name
                    break
    counts = {name: 0 for name in pats}
    for v in member.values():
        counts[v] += 1

    # 2. aggregate group-to-group synapses and transmitter votes
    pair_n: dict[tuple[str, str], int] = {}
    pair_syn: dict[tuple[str, str], int] = {}
    nt_votes: dict[str, dict[str, int]] = {}
    with _open(connections_csv) as f:
        r = csv.reader(f)
        h = next(r)
        pi, qi = _col(h, "pre_root_id", "pre_pt_root_id"), _col(h, "post_root_id", "post_pt_root_id")
        si = _col(h, "syn_count", "count")
        try:
            ni = _col(h, "nt_type")
        except KeyError:
            ni = None
        for row in r:
            pre, post = member.get(int(row[pi])), member.get(int(row[qi]))
            if pre is None:
                continue
            if ni is not None and row[ni]:
                v = nt_votes.setdefault(pre, {})
                v[row[ni]] = v.get(row[ni], 0) + 1
            if post is None:
                continue
            syn = int(row[si])
            if syn < min_syn:
                continue
            pair_n[(pre, post)] = pair_n.get((pre, post), 0) + 1
            pair_syn[(pre, post)] = pair_syn.get((pre, post), 0) + syn

    # 3. densities + strengths
    dens = dict(PATHWAYS) if keep_default_paths else {}
    strengths = {}
    for (pre, post), k in pair_n.items():
        possible = counts[pre] * counts[post]
        if possible == 0:
            continue
        d = float(np.clip(k / possible * 4.0, 0.05, 1.0))  # small model: boost sparse real densities
        dens[(pre, post)] = max(d, dens.get((pre, post), 0.0)) if keep_default_paths else d
        strengths[(pre, post)] = float(np.clip(np.log1p(pair_syn[(pre, post)] / k) / np.log1p(20), 0.3, 2.0))

    groups = []
    for g in GROUPS:
        votes = nt_votes.get(g.name)
        nt = g.nt
        if votes:
            top = max(votes, key=votes.get).upper()
            nt = GABA if "GABA" in top else GLUT if "GLUT" in top else ACH
        groups.append(Group(g.name, g.size, g.layer, nt, g.about, g.flywire))

    matched = sum(1 for v in counts.values() if v)
    src = f"FlyWire ({len(member)} real neurons matched across {matched} groups, {len(pair_n)} real pathways)"
    return add_innate_topography(_wire(groups, dens, _signs(groups, rng), rng, src, strengths))
