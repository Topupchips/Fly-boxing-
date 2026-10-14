"""The browser engine (docs/engine.js) must fight the same way as the Python sim.

Runs one brain-vs-brain fight with fixed starting positions in both, then
compares the recorded frames. Python keeps fly state in float32 and the JS
port in float64, so tiny rounding differences are expected and can snowball
in a chaotic fight; we check the opening exchanges tightly.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flybox import arena
from flybox.arena import fight
from flybox.brain import BrainSpec
from flybox.connectome import build_default_connectome

STARTS = [[[-6.0, -5.0, 0.6], [5.0, 6.5, -2.4]]]
STEPS = 240
COMPARE = 150

NODE_SCRIPT = """
import { BrainSpec, simulateFight } from '%s';
import fs from 'fs';
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const spec = new BrainSpec(d.brain);
const out = simulateFight(spec, d.fighters, { rounds: [%d], starts: d.starts });
fs.writeFileSync(process.argv[3], JSON.stringify(out.frames));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_engine_matches_python(tmp_path, monkeypatch):
    sys.path.insert(0, ROOT)
    from train import brain_map

    spec = BrainSpec(build_default_connectome())
    genomes = spec.random_genomes(2, np.random.default_rng(3))
    W, b = spec.weights(genomes)

    def fixed_starts(B, rng, randomize):
        s = np.array(STARTS[0], dtype=np.float64)
        return s[:, 0:1], s[:, 1:2], s[:, 2:3]

    monkeypatch.setattr(arena, "_starts", fixed_starts)
    res = fight(spec, np.stack([W[0:1], W[1:2]]), np.stack([b[0:1], b[1:2]]), np.random.default_rng(0),
                rounds=(STEPS,), record=True)
    py = [f[0] for f in res.frames]

    data = {"brain": brain_map(spec), "starts": STARTS,
            "fighters": [{"w": W[i].astype(float).tolist(), "b": b[i].astype(float).tolist()} for i in range(2)]}
    inp, outp, script = tmp_path / "in.json", tmp_path / "out.json", tmp_path / "run.mjs"
    inp.write_text(json.dumps(data))
    engine = "file://" + os.path.join(ROOT, "docs", "engine.js")
    script.write_text(NODE_SCRIPT % (engine, STEPS))
    subprocess.run(["node", str(script), str(inp), str(outp)], check=True)
    js = json.loads(outp.read_text())

    assert len(js) == len(py)
    moves = 0
    for i in range(COMPARE):
        for s in (0, 1):
            p, j = py[i][s], js[i][s]
            assert abs(p["x"] - j["x"]) < 0.05 and abs(p["y"] - j["y"]) < 0.05, (i, s, p, j)
            assert abs(np.angle(np.exp(1j * (p["h"] - j["h"])))) < 0.01, (i, s)
            assert abs(p["hp"] - j["hp"]) < 1e-6, (i, s)
            for k in ("l", "g", "d", "hit", "blk"):
                assert p[k] == j[k], (i, s, k)
            moves += p["l"] + p["g"] + p["d"]
            diff = sum(abs(int(a) - int(c)) for a, c in zip(p["b"], j["b"]))
            assert diff <= 4, (i, s, diff)
    assert moves > 0, "pick a seed where the flies actually do something"
