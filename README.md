# 🥊 Fly Boxing

Fruit flies with brains wired after the *Drosophila* connectome, trained by evolution to box each other in a 28 mm ring.

**[▶ Watch the fights](https://topupchips.github.io/Fly-boxing-/)** (after enabling GitHub Pages, see below)

Every move a fighter makes, from turning to face its rival to lunging, guarding and escape-hopping, comes out of simulated neurons. Nobody programmed a punch: the flies found them.

![ring](docs/preview.png)

## How it works

```
 eyes / nose / bristles          central brain                  descending neurons        body
 ─────────────────────           ─────────────                  ──────────────────        ────
 LC10  rival bearing   ──┐     PFL3  steering        ──┐       DNa02   turn L/R      ──▶  walk, turn
 LC4   looming rival   ──┼──▶  FB    fan-shaped body ──┼──▶    DNp09   walk forward  ──▶  back up, side-step
 LC16  sudden lunge    ──┤     P1    aggression      ──┤       MDN     back up       ──▶  LUNGE (punch)
 MS    got hit L/R     ──┤     LH    lateral horn    ──┤       GF      giant fiber   ──▶  DODGE (escape hop)
 BR    touching ropes  ──┤     KC/MBON mushroom body ──┤       DN_lunge / DN_guard   ──▶  GUARD (forelegs up)
 Or67d rival's cVA     ──┘     LN    inhibition      ──┘
```

1. **Connectome** (`flybox/connectome.py`): 94 neurons in 21 real fly cell types and ~850 synapses. Which cell types talk to which follows known fly circuits (LC4 → giant fiber, P1 → aggression, PFL3 → DNa02 steering...), every neuron has one transmitter sign (Dale's law), and flies are born with an innate left/right map (a rival on the left excites left-side steering neurons).
2. **Brain** (`flybox/brain.py`): a leaky firing-rate network, stepped 30 times a second. Sensory neurons are driven by what the fly sees, smells and feels; descending-neuron rates become movement commands.
3. **Ring** (`flybox/arena.py`): top-down physics. Lunges connect only in range and within the strike cone, a guard facing the attacker blocks most of the damage, a giant-fiber hop dodges entirely. Lunging and dodging cost stamina.
4. **Training** (`flybox/evolve.py`): self-play neuroevolution. Each fly's genome is the strength of every synapse in its wiring diagram. Flies fight their peers, past champions (Hall of Fame) and a scripted sparring bot. Fitness = damage dealt − 0.8 × damage taken + KO bonus. The best breed and mutate. The wiring never changes, only how strongly each existing synapse fires.
5. **Ring viewer** (`docs/index.html`): replays the fight card with both fighters' brains lighting up live next to the ring.

Real male fruit flies do fight over food and mates: they lunge, wing-threat, and rear up and "box" with their forelegs. This project just gives them gloves.

## Run it

```bash
pip install -r requirements.txt
python train.py                  # 150 generations, about 15 min on a laptop
python train.py --generations 50 # quick run
open docs/index.html             # watch the fights
```

`train.py` writes `models/champion.npz` (best genome), `models/history.json` (fitness curve) and `docs/fights.js` (replays for the viewer).

## Use real FlyWire data

The default wiring is a scaled-down model whose pathways come from the literature. To derive the wiring densities, synapse strengths and transmitter signs from the actual FlyWire connectome:

1. Go to [codex.flywire.ai](https://codex.flywire.ai/api/download) and download `connections.csv.gz` and `consolidated_cell_types.csv.gz` (free account, cite FlyWire per their terms).
2. Train on it:

```bash
python train.py --flywire-connections connections.csv.gz --flywire-types consolidated_cell_types.csv.gz
```

The loader matches real neurons to the 21 model cell types by name (e.g. `^LC4$`, `^DNa02$`, `^PFL3`), measures how densely and strongly each pair of types connects in the real brain, and takes each type's transmitter from FlyWire's predictions. Cell types that have no match fall back to the default pathway table. The model is still 94 neurons, a cartoon of the 139,255-neuron fly brain, so think "wired by FlyWire statistics", not "a full brain emulation".

## Put the ring online (GitHub Pages)

Repo **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `docs`**. The ring will be at `https://topupchips.github.io/Fly-boxing-/`.

## Project layout

```
flybox/connectome.py   cell types, pathways, innate wiring, FlyWire loader
flybox/brain.py        batched rate-model brain, motor readout
flybox/arena.py        the ring: physics, strikes, guards, dodges, replays
flybox/evolve.py       self-play neuroevolution
train.py               train + export fights
tests/test_flybox.py   sanity checks (pytest)
docs/index.html        the boxing ring viewer
docs/fights.js         generated fight data
```

## Ideas for round two

- Live mode: run the brains in the browser and let visitors pick two champions.
- Swap in more FlyWire cell types (Tk aggression neurons, octopaminergic modulation).
- Spiking neurons instead of rates.
- Tag-team fights.

## License

MIT
