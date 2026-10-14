"""
Export the real fly brain for the 3D viewer.

    python tools/export_brain3d.py Supplemental_file1_neuron_annotations.tsv

Input: the FlyWire neuron annotation table (139,248 neurons) from
https://github.com/flyconnectome/flywire_annotations (supplemental_files/).
Output: docs/brain3d.bin + docs/brain3d.json, about 1.1 MB.

Each neuron becomes one point at its FlyWire position. Neurons whose cell type
matches one of the model's 21 groups (the same patterns the FlyWire loader
uses) are tagged with that group, so the viewer can light every real LC4,
DNa02, Kenyon cell... as its model counterpart fires.

Data: FlyWire Consortium. Cite Dorkenwald et al. 2024 and Schlegel et al. 2024
(Nature), plus the papers listed in the flywire_annotations README.
"""
import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flybox.connectome import GROUPS  # noqa: E402

VOXEL_NM = np.array([4.0, 4.0, 40.0])   # FlyWire positions are in 4 x 4 x 40 nm voxels


def main(tsv):
    groups = [g for g in GROUPS]
    pats = [(i, re.compile(g.flywire)) for i, g in enumerate(groups) if g.flywire]
    classes, pos, cls, grp = [], [], [], []
    with open(tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pos.append((float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])))
            c = row["super_class"] or "unknown"
            if c not in classes:
                classes.append(c)
            cls.append(classes.index(c))
            g = 255
            for t in (row["cell_type"], row["hemibrain_type"]):
                if not t:
                    continue
                for i, pat in pats:
                    if pat.search(t):
                        g = i
                        break
                if g != 255:
                    break
            grp.append(g)
    um = np.asarray(pos) * VOXEL_NM / 1000.0                 # micrometres
    lo, hi = um.min(0), um.max(0)
    q = np.round((um - lo) / (hi - lo) * 65535).astype("<u2")
    out_bin = os.path.join(ROOT, "docs", "brain3d.bin")
    with open(out_bin, "wb") as f:
        f.write(q.tobytes())
        f.write(np.asarray(cls, "u1").tobytes())
        f.write(np.asarray(grp, "u1").tobytes())
    counts = np.bincount([g for g in grp if g != 255], minlength=len(groups))
    meta = {
        "n": len(pos), "bbox_um": [lo.round(1).tolist(), hi.round(1).tolist()],
        "classes": classes, "groups": [g.name for g in groups],
        "group_counts": counts.tolist(),
        "source": "FlyWire Consortium (Dorkenwald et al. 2024; Schlegel et al. 2024), flyconnectome/flywire_annotations",
    }
    with open(os.path.join(ROOT, "docs", "brain3d.json"), "w") as f:
        json.dump(meta, f)
    print(f"{len(pos)} neurons, {int(counts.sum())} tagged to model cell types -> docs/brain3d.bin "
          f"({os.path.getsize(out_bin) / 1e6:.2f} MB)")
    for g, c in zip(groups, counts):
        print(f"  {g.name:10s} {c}")


if __name__ == "__main__":
    main(sys.argv[1])
