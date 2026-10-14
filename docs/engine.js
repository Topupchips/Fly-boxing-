// Fly Boxing live engine: a JavaScript port of flybox/brain.py and flybox/arena.py.
// Runs evolved fly brains in the browser so every fight is new.
// Kept step-for-step faithful to the Python sim; tests/engine_parity.mjs checks it.

export const HALF = 14, BODY_R = 1.2, REACH = 3.8, STRIKE_CONE = 0.55, BLOCK_CONE = 0.9;
export const LUNGE_T = 5, LUNGE_HIT = 3, LUNGE_CD = 14, LUNGE_COST = 0.14;
export const DODGE_T = 4, DODGE_CD = 24, DODGE_COST = 0.22;
export const DAMAGE = 10, BLOCKED_DAMAGE = 2, MAX_HEALTH = 100, DT = 1 / 30;
const LC10_CENTRES = [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5].map(d => d * Math.PI / 180);
const VIS = 165 * Math.PI / 180;

const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export function wrap(a) {
  const t = 2 * Math.PI;
  return ((((a + Math.PI) % t) + t) % t) - Math.PI;
}

// ---------------------------------------------------------------- brain
export class BrainSpec {
  constructor(brain) {
    this.n = brain.n;
    this.nSens = brain.n_sens;
    this.pre = Int32Array.from(brain.edges, e => e[0]);
    this.post = Int32Array.from(brain.edges, e => e[1]);
    this.leak = Float64Array.from(brain.leak);
    this.out = brain.out;   // {name: [start, stop]}
  }
}

export class Brain {
  constructor(spec, fighter) {
    this.spec = spec;
    this.w = Float64Array.from(fighter.w);
    this.b = Float64Array.from(fighter.b);
    this.r = new Float64Array(spec.n);
    this.x = new Float64Array(spec.n);
  }
  reset() { this.r.fill(0); }
  // noise: optional neural noise (real neurons are noisy); 0 reproduces the Python sim exactly
  step(sensory, noise = 0, rand = null) {
    const { n, nSens, pre, post, leak } = this.spec, r = this.r, x = this.x;
    for (let i = 0; i < nSens; i++) r[i] = sensory[i];
    x.set(this.b);
    const w = this.w;
    for (let k = 0; k < w.length; k++) x[post[k]] += w[k] * r[pre[k]];
    if (noise > 0) for (let i = nSens; i < n; i++) x[i] += noise * rand.normal();
    for (let i = 0; i < n; i++) r[i] = (1 - leak[i]) * r[i] + leak[i] * Math.tanh(Math.max(x[i], 0));
    for (let i = 0; i < nSens; i++) r[i] = sensory[i];
    return r;
  }
  motor() {
    const r = this.r, o = this.spec.out;
    const at = (name, k) => r[o[name][0] + k];
    const mean = name => { let s = 0; for (let i = o[name][0]; i < o[name][1]; i++) s += r[i]; return s / (o[name][1] - o[name][0]); };
    const gfL = at('GF', 0), gfR = at('GF', 1);
    return {
      turn: at('DNa02', 0) - at('DNa02', 1),
      forward: at('DNp09', 0) - 0.6 * at('MDN', 0),
      strafe: at('DN_strafe', 0) - at('DN_strafe', 1),
      lunge: mean('DN_lunge') > 0.5,
      guard: mean('DN_guard') > 0.55,
      dodge: Math.max(gfL, gfR) > 0.6,
      dodgeDir: gfL >= gfR ? -1 : 1,
    };
  }
}

function sparringBot(me, opp, t, rand) {
  const dx = opp.x - me.x, dy = opp.y - me.y, d = Math.hypot(dx, dy);
  const bear = wrap(Math.atan2(dy, dx) - me.h);
  return {
    turn: clip(bear * 2, -1, 1),
    forward: d > 3.2 ? 0.8 : d < 2.6 ? -0.4 : 0,
    strafe: Math.sin(t * 0.07) * 0.5,
    lunge: d < REACH - 0.2 && Math.abs(bear) < 0.4 && rand() < 0.35,
    guard: opp.lungeT > 0 && d < 5 && rand() < 0.5,
    dodge: false,
    dodgeDir: 1,
  };
}

// ---------------------------------------------------------------- senses
function sensory(me, opp, prevAng, out) {
  const dx = opp.x - me.x, dy = opp.y - me.y;
  const d = Math.max(Math.hypot(dx, dy), 1e-3);
  const bear = wrap(Math.atan2(dy, dx) - me.h);
  const ang = 2 * Math.atan(BODY_R / d);
  const loom = clip((ang - prevAng) * 25, 0, 1);
  const visible = Math.abs(bear) < VIS ? 1 : 0;
  const size = clip(ang * 1.5, 0, 1) * visible;
  let k = 0;
  for (const c of LC10_CENTRES) { const e = wrap(bear - c); out[k++] = size * Math.exp(-e * e / (2 * 0.35 * 0.35)); }
  const left = bear > 0 ? 1 : 0;
  out[k++] = loom * left * visible; out[k++] = loom * (1 - left) * visible;           // LC4
  const burst = (opp.lungeT > 0 || opp.dodgeT > 0 ? 1 : 0) * visible;
  out[k++] = burst * left; out[k++] = burst * (1 - left);                             // LC16
  out[k++] = me.wasHit * (me.hitBear > 0 ? 1 : 0); out[k++] = me.wasHit * (me.hitBear <= 0 ? 1 : 0); // MS_hit
  for (const off of [0, Math.PI / 2, -Math.PI / 2, Math.PI]) {                        // BR_wall
    const a = me.h + off, c = Math.cos(a), s = Math.sin(a);
    const tx = c > 1e-6 ? (HALF - me.x) / c : c < -1e-6 ? (-HALF - me.x) / c : Infinity;
    const ty = s > 1e-6 ? (HALF - me.y) / s : s < -1e-6 ? (-HALF - me.y) / s : Infinity;
    out[k++] = clip(1 - (Math.min(tx, ty) - BODY_R) / 4, 0, 1);
  }
  out[k++] = Math.exp(-d / 6);                                                         // ORN_cVA
  out[k++] = me.stamina; out[k++] = me.health / MAX_HEALTH; out[k++] = me.lungeCd === 0 ? 1 : 0; // INT
  return ang;
}

// ---------------------------------------------------------------- random
export function makeRng(seed = (Math.random() * 2 ** 32) >>> 0) {
  let s = seed >>> 0;
  const rand = () => {                // mulberry32
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rand.normal = (mu = 0, sd = 1) => {
    const u = 1 - rand(), v = rand();
    return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return rand;
}

// ---------------------------------------------------------------- the fight
/**
 * Simulate a whole fight and return it in the same shape as the recorded replays.
 * fighters: two roster entries ({w, b} brains, or {bot: true}).
 * opts.rounds: steps per round. opts.seed: makes the fight repeatable.
 * opts.starts: optional [[x, y, h], [x, y, h]] per round (used by the parity test).
 * opts.noise: neural noise per step (default 0.04); opts.spread: how far from the corners
 * fighters may start each round, in mm (default 3). Both 0 = the training conditions.
 */
export function simulateFight(spec, fighters, opts = {}) {
  const rounds = opts.rounds || [450, 450, 450];
  const rand = makeRng(opts.seed);
  const noise = opts.noise ?? 0.04;
  const brains = fighters.map(f => (f.bot ? null : new Brain(spec, f)));
  const fly = () => ({ x: 0, y: 0, h: 0, health: MAX_HEALTH, stamina: 1, lungeT: 0, lungeCd: 0, dodgeT: 0, dodgeCd: 0,
    dodgeDir: 1, guard: false, wasHit: 0, hitBear: 0, blk: false });
  const F = [fly(), fly()];
  const stats = [0, 1].map(() => ({ dealt: 0, taken: 0, hits: 0, blocks: 0, dodges: 0 }));
  const sens = [new Float64Array(spec.nSens), new Float64Array(spec.nSens)];
  const frames = [];
  let done = false, ko = [false, false], t = 0;

  for (let ri = 0; ri < rounds.length; ri++) {
    const sp = opts.spread ?? 3;
    const jit = sd => clip(rand.normal(0, Math.max(sd, 0.4)), -2.5 * Math.max(sd, 0.4), 2.5 * Math.max(sd, 0.4));
    const st = opts.starts ? opts.starts[ri] : [
      [clip(-9 + jit(sp), -HALF + 2, HALF - 2), clip(-9 + jit(sp), -HALF + 2, HALF - 2), Math.PI / 4 + jit(sp * 0.25)],
      [clip(9 + jit(sp), -HALF + 2, HALF - 2), clip(9 + jit(sp), -HALF + 2, HALF - 2), -3 * Math.PI / 4 + jit(sp * 0.25)],
    ];
    for (let s = 0; s < 2; s++) {
      Object.assign(F[s], { x: st[s][0], y: st[s][1], h: st[s][2], stamina: 1, lungeT: 0, dodgeT: 0, lungeCd: 0, dodgeCd: 0 });
      if (brains[s]) brains[s].reset();
    }
    const prevAng = [2 * Math.atan(BODY_R / 18), 2 * Math.atan(BODY_R / 18)];
    for (let step = 0; step < rounds[ri]; step++) {
      if (done) break;
      t++;
      // senses -> brain -> motor commands
      const angs = [sensory(F[0], F[1], prevAng[0], sens[0]), sensory(F[1], F[0], prevAng[1], sens[1])];
      prevAng[0] = angs[0]; prevAng[1] = angs[1];
      const act = [0, 1].map(s => {
        if (brains[s]) { brains[s].step(sens[s], noise, rand); return brains[s].motor(); }
        return sparringBot(F[s], F[1 - s], t, rand);
      });
      // start moves
      const lunging = [], dodging = [];
      for (let s = 0; s < 2; s++) {
        const f = F[s], a = act[s];
        const busy = f.lungeT > 0 || f.dodgeT > 0;
        const startL = a.lunge && f.lungeCd === 0 && f.stamina >= LUNGE_COST && !busy;
        const startD = a.dodge && f.dodgeCd === 0 && f.stamina >= DODGE_COST && !busy && !startL;
        if (startL) { f.lungeT = LUNGE_T; f.lungeCd = LUNGE_CD; f.stamina -= LUNGE_COST; }
        if (startD) { f.dodgeT = DODGE_T; f.dodgeCd = DODGE_CD; f.stamina -= DODGE_COST; f.dodgeDir = a.dodgeDir; stats[s].dodges++; }
        lunging[s] = f.lungeT > 0; dodging[s] = f.dodgeT > 0;
        f.guard = a.guard && !lunging[s] && !dodging[s];
      }
      // movement
      for (let s = 0; s < 2; s++) {
        const f = F[s], a = act[s];
        f.h = wrap(f.h + clip(a.turn, -1, 1) * (lunging[s] ? 1.5 : 6) * DT);
        let fwd = clip(a.forward, -1, 1);
        fwd = fwd > 0 ? fwd * 20 : fwd * 9;
        let side = clip(a.strafe, -1, 1) * 9;
        const slow = f.guard ? 0.45 : 1;
        fwd *= slow; side *= slow;
        if (lunging[s]) { fwd = 42; side = 0; }
        if (dodging[s]) { fwd = -30; side = 45 * f.dodgeDir; }
        const c = Math.cos(f.h), sn = Math.sin(f.h);
        f.x += (c * fwd - sn * side) * DT;
        f.y += (sn * fwd + c * side) * DT;
      }
      // strikes
      F[0].wasHit = F[1].wasHit = 0; F[0].blk = F[1].blk = false;
      for (let a = 0; a < 2; a++) {
        const A = F[a], Dd = F[1 - a];
        if (A.lungeT !== LUNGE_HIT) continue;
        const dx = Dd.x - A.x, dy = Dd.y - A.y, dd = Math.hypot(dx, dy);
        const ab = wrap(Math.atan2(dy, dx) - A.h);
        if (!(dd < REACH && Math.abs(ab) < STRIKE_CONE && Dd.dodgeT === 0)) continue;
        const db = wrap(Math.atan2(-dy, -dx) - Dd.h);
        const blocked = Dd.guard && Math.abs(db) < BLOCK_CONE;
        const dmg = blocked ? BLOCKED_DAMAGE : DAMAGE;
        Dd.health -= dmg;
        if (blocked) { A.stamina -= 0.1; stats[1 - a].blocks++; Dd.blk = true; } else stats[a].hits++;
        stats[a].dealt += dmg; stats[1 - a].taken += dmg;
        Dd.wasHit = 1; Dd.hitBear = db;
        const kb = (blocked ? 0.6 : 1.6) / Math.max(dd, 1e-3);
        Dd.x += dx * kb; Dd.y += dy * kb;
      }
      // bodies and ropes
      {
        const dx = F[1].x - F[0].x, dy = F[1].y - F[0].y, dd = Math.max(Math.hypot(dx, dy), 1e-3);
        const push = Math.max(2 * BODY_R - dd, 0) / 2 / dd;
        F[0].x -= dx * push; F[0].y -= dy * push; F[1].x += dx * push; F[1].y += dy * push;
        const lim = HALF - BODY_R;
        for (const f of F) { f.x = clip(f.x, -lim, lim); f.y = clip(f.y, -lim, lim); }
      }
      // timers and stamina
      for (const f of F) {
        f.lungeT = Math.max(f.lungeT - 1, 0); f.dodgeT = Math.max(f.dodgeT - 1, 0);
        f.lungeCd = Math.max(f.lungeCd - 1, 0); f.dodgeCd = Math.max(f.dodgeCd - 1, 0);
        f.stamina = clip(f.stamina + 0.009 - (f.guard ? 0.006 : 0), 0, 1);
      }
      frames.push(F.map((f, s) => ({
        x: +f.x.toFixed(2), y: +f.y.toFixed(2), h: +f.h.toFixed(3),
        hp: +Math.max(f.health, 0).toFixed(1), st: +f.stamina.toFixed(2),
        l: f.lungeT > 0 ? 1 : 0, g: f.guard ? 1 : 0, d: f.dodgeT > 0 ? 1 : 0, hit: f.wasHit ? 1 : 0, blk: f.blk ? 1 : 0,
        b: brains[s] ? digits(brains[s].r) : '0'.repeat(spec.n),
      })));
      if (F[0].health <= 0 || F[1].health <= 0) {
        ko = [F[1].health <= 0, F[0].health <= 0];
        done = true;
      }
    }
  }
  const hp = [MAX_HEALTH - stats[0].taken, MAX_HEALTH - stats[1].taken];
  let winner = hp[0] > hp[1] + 1e-6 ? 0 : hp[1] > hp[0] + 1e-6 ? 1 : -1;
  if (ko[0]) winner = 0; else if (ko[1]) winner = 1;
  return { frames, winner, ko: ko[0] || ko[1], stats, round_len: rounds };
}

function digits(r) {
  let s = '';
  for (let i = 0; i < r.length; i++) s += Math.min(9, Math.max(0, Math.floor(r[i] * 10)));
  return s;
}
