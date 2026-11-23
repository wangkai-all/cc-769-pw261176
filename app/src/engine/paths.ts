/**
 * Path ensemble.
 *
 * The per-segment posterior tells us how much of each report interval was spent
 * running / slow / stopped, but it cannot say *where* inside the interval an
 * event sits, and several different splits can explain the same count. We turn
 * that into an explicit set of hidden paths:
 *
 *   - one time split is drawn per segment from its posterior grid;
 *   - the down and slow minutes of that split are placed as contiguous blocks
 *     at a position uniform over the segment (a stop is a block, not a drizzle);
 *   - operator-confirmed minutes are placed exactly where they were confirmed.
 *
 * The spread of block boundaries across the ensemble is the only honest source
 * of a boundary range. No timestamp below is finer than a report interval
 * unless an operator confirmed it.
 */
import type { Constraint, Segment, SegmentKind, StateName } from './types';
import { STATES } from './types';

const MIN = 60_000;
export const PATH_SAMPLES = 320;
/** Blocks shorter than this are sampling dust, not identifiable events. */
const MIN_BLOCK_MIN = 3;
export const BIN_MIN = 2;

export interface SegGrid {
  free: number;
  pinned: number;
  /** constraint-derived down intervals, minutes relative to seg.t0 */
  fixedDown: { a: number; b: number }[];
  points: { fr: number; fs: number; fd: number; weight: number }[];
  /** cumulative weights for sampling */
  cumulative: number[];
}

export interface Span {
  t0: number;
  t1: number;
  state: StateName;
}

export interface PathSample {
  index: number;
  logW: number;
  minutes: Record<StateName, number>;
  spans: Span[];
  /** per-segment state assignment, compact: [segIndex, runMin, slowMin, downMin] */
  producedParts: number;
  lossParts: number;
}

export interface PathSet {
  samples: PathSample[];
  /** per-state share of paths that have any time in that state at a moment */
  coverage: { t0: number; t1: number; c: Record<StateName, number> }[];
  bins: { t0: number; t1: number; p: Record<StateName, number> }[];
  /** distinct down structures, best first */
  clusters: PathCluster[];
  loss: { mean: number; lo: number; hi: number; samples: number[] };
  produced: { mean: number; lo: number; hi: number };
}

export interface PathCluster {
  id: string;
  weight: number;
  logGap: number;
  representative: PathSample;
  /** the down spans of the representative, merged to minute precision */
  downSpans: { t0: number; t1: number }[];
  minutes: Record<StateName, number>;
  /** mean down / slow minutes this group claims */
  downMin: number;
  slowMin: number;
  /** number of substantive stops in the representative */
  stops: number;
  /** no single reading dominates: show the spread, not a ranked list */
  diffuse: boolean;
}

export function buildSegGrid(
  seg: Segment,
  grid: { fr: number; fs: number; fd: number; weight: number }[],
  constraints: Constraint[],
): SegGrid {
  const T = seg.dt / MIN;
  const fixedDown: { a: number; b: number }[] = [];
  let pinned = 0;
  for (const c of constraints) {
    const a = Math.max(seg.t0, c.start);
    const b = Math.min(seg.t1, c.end);
    if (b > a) {
      fixedDown.push({ a: (a - seg.t0) / MIN, b: (b - seg.t0) / MIN });
      pinned += (b - a) / MIN;
    }
  }
  const free = Math.max(0, T - pinned);
  const cumulative: number[] = [];
  let acc = 0;
  for (const p of grid) {
    acc += p.weight;
    cumulative.push(acc);
  }
  return { free, pinned, fixedDown, points: grid, cumulative };
}

function sampleIndex(grid: SegGrid, u: number): number {
  const c = grid.cumulative;
  if (!c.length) return 0;
  const target = Math.min(0.999999, Math.max(0, u)) * c[c.length - 1];
  let lo = 0;
  let hi = c.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (c[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

type Rng = () => number;

/**
 * Draw paths. Uses a seeded PRNG so the same dataset always yields the same
 * ensemble, which makes the UI reproducible and the verification honest.
 */
export function samplePaths(segments: Segment[], grids: SegGrid[], seed = 0x5eed, counterBias = 1): PathSet {
  const rng = mulberry32(seed);
  const samples: PathSample[] = [];

  // ideal parts per segment, from that era's own standard
  const ideal = segments.map((s) => (s.dt / MIN) * (60_000 / s.idealCycleMs));

  for (let n = 0; n < PATH_SAMPLES; n++) {
    let logW = 0;
    const minutes: Record<StateName, number> = { run: 0, slow: 0, down: 0 };
    const spans: Span[] = [];
    let produced = 0;

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      const g = grids[i];
      if (!g.points.length) continue;
      const idx = sampleIndex(g, rng());
      const pt = g.points[idx];
      logW += pt.weight > 0 ? Math.log(pt.weight) : -50;

      const T = seg.dt / MIN;
      // A block shorter than a few minutes is not an event anyone can see in
      // the counts; snapping it away keeps the ensemble from covering the whole
      // day in sub-minute speckle.
      let downMin = pt.fd * g.free + g.pinned;
      let slowMin = pt.fs * g.free;
      if (downMin < MIN_BLOCK_MIN) downMin = 0;
      if (slowMin < MIN_BLOCK_MIN) slowMin = 0;
      if (downMin + slowMin > T) {
        const k = T / (downMin + slowMin);
        downMin *= k;
        slowMin *= k;
      }
      const runMin = Math.max(0, T - downMin - slowMin);

      const placed = placeBlocks(seg, T, downMin, slowMin, g.fixedDown, rng);
      for (const sp of placed) {
        spans.push(sp);
        minutes[sp.state] += (sp.t1 - sp.t0) / MIN;
      }
      minutes.run += runMin;

      const rIdeal = (60_000 / seg.idealCycleMs) * counterBias;
      const rSlow = rIdeal / 2.6;
      if (seg.delta !== null && seg.kind === 'normal') {
        produced += seg.delta;
      } else {
        produced += runMin * rIdeal + slowMin * rSlow;
      }
    }

    const producedParts = produced;
    const lossParts = Math.max(0, ideal.reduce((a, b) => a + b, 0) - producedParts);
    samples.push({ index: n, logW, minutes, spans, producedParts, lossParts });
  }

  const bins = buildBins(segments, samples);
  const coverage = buildCoverage(segments, samples);
  const clusters = clusterPaths(samples);
  return {
    samples,
    coverage,
    bins,
    clusters,
    loss: dist(samples.map((s) => s.lossParts)),
    produced: dist(samples.map((s) => s.producedParts)),
  };
}

function placeBlocks(
  seg: Segment,
  T: number,
  downMin: number,
  slowMin: number,
  fixedDown: { a: number; b: number }[],
  rng: Rng,
): Span[] {
  interface Occ {
    a: number;
    b: number;
    state: StateName;
  }
  const occ: Occ[] = fixedDown.map((f) => ({ a: Math.max(0, f.a), b: Math.min(T, f.b), state: 'down' as StateName }));
  const fixedDownTotal = occ.reduce((s, o) => s + (o.b - o.a), 0);
  const extraDown = Math.max(0, downMin - fixedDownTotal);
  const free = Math.max(0, T - fixedDownTotal);

  // extra down: one block, uniform over the segment, retried off the fixed ones
  if (extraDown > 0.01) {
    const len = Math.min(extraDown, free);
    const start = placeAvoiding(len, T, occ, rng);
    if (start !== null) occ.push({ a: start, b: start + len, state: 'down' });
  }
  // slow: one block placed in what is still free
  const slowFree = Math.max(0, T - occ.reduce((s, o) => s + (o.b - o.a), 0));
  if (slowMin > 0.01 && slowFree > 0.01) {
    const len = Math.min(slowMin, slowFree);
    const start = placeAvoiding(len, T, occ, rng);
    if (start !== null) occ.push({ a: start, b: start + len, state: 'slow' });
  }

  occ.sort((x, y) => x.a - y.a);
  // fill the gaps with run
  const out: Span[] = [];
  let cursor = 0;
  for (const o of occ) {
    if (o.a > cursor + 1e-6) out.push({ t0: seg.t0 + cursor * MIN, t1: seg.t0 + o.a * MIN, state: 'run' });
    out.push({ t0: seg.t0 + o.a * MIN, t1: seg.t0 + o.b * MIN, state: o.state });
    cursor = Math.max(cursor, o.b);
  }
  if (cursor < T - 1e-6) out.push({ t0: seg.t0 + cursor * MIN, t1: seg.t0 + T * MIN, state: 'run' });

  // drop zero-length spans and merge neighbours of the same state
  const clean = out.filter((s) => s.t1 - s.t0 > 1);
  const merged: Span[] = [];
  for (const s of clean) {
    const last = merged[merged.length - 1];
    if (last && last.state === s.state && Math.abs(last.t1 - s.t0) < 1) last.t1 = s.t1;
    else merged.push({ ...s });
  }
  return merged;
}

function placeAvoiding(len: number, T: number, occ: { a: number; b: number }[], rng: Rng): number | null {
  if (len <= 0 || T <= len) return 0;
  for (let attempt = 0; attempt < 10; attempt++) {
    const start = rng() * (T - len);
    const end = start + len;
    if (!occ.some((o) => end > o.a + 0.5 && start < o.b - 0.5)) return start;
  }
  // fall back to the largest gap
  const sorted = [...occ].sort((a, b) => a.a - b.a);
  let best = 0;
  let bestGap = -1;
  let cursor = 0;
  for (const o of sorted) {
    if (o.a - cursor > bestGap) {
      bestGap = o.a - cursor;
      best = cursor;
    }
    cursor = Math.max(cursor, o.b);
  }
  if (T - cursor > bestGap) best = cursor;
  return best;
}

function buildBins(segments: Segment[], samples: PathSample[]) {
  if (!segments.length) return [];
  const t0 = segments[0].t0;
  const t1 = segments[segments.length - 1].t1;
  const step = BIN_MIN * MIN;
  const n = Math.max(1, Math.ceil((t1 - t0) / step));
  const bins: { t0: number; t1: number; p: Record<StateName, number> }[] = [];
  // time-weighted occupancy per state, so one path that has both a slow and a
  // down block in the same bin cannot contribute more than 1
  const occ: Record<StateName, number>[] = Array.from({ length: n }, () => ({ run: 0, slow: 0, down: 0 }));
  for (const s of samples) {
    for (const sp of s.spans) {
      const a = Math.max(t0, sp.t0);
      const b = Math.min(t1, sp.t1);
      if (b <= a) continue;
      const i0 = Math.max(0, Math.floor((a - t0) / step));
      const i1 = Math.min(n - 1, Math.floor((b - 1 - t0) / step));
      for (let i = i0; i <= i1; i++) {
        const ba = t0 + i * step;
        const bb = Math.min(t1, ba + step);
        const w = Math.max(0, Math.min(b, bb) - Math.max(a, ba)) / (bb - ba);
        occ[i][sp.state] += w;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const tot = occ[i].run + occ[i].slow + occ[i].down;
    const k = tot > 1e-9 ? 1 / tot : 0;
    bins.push({
      t0: t0 + i * step,
      t1: Math.min(t1, t0 + (i + 1) * step),
      p: { run: occ[i].run * k, slow: occ[i].slow * k, down: occ[i].down * k },
    });
  }
  return bins;
}

/** Per bin, the share of sampled paths in which a state has any presence.
 *  Boundary ranges are built from this: a bin where 40% of paths keep a stop
 *  running is plausibly part of that stop. */
function buildCoverage(segments: Segment[], samples: PathSample[]) {
  if (!segments.length) return [];
  const t0 = segments[0].t0;
  const t1 = segments[segments.length - 1].t1;
  const step = BIN_MIN * MIN;
  const n = Math.max(1, Math.ceil((t1 - t0) / step));
  const out: { t0: number; t1: number; c: Record<StateName, number> }[] = [];
  const tally: Record<StateName, number>[] = Array.from({ length: n }, () => ({ run: 0, slow: 0, down: 0 }));
  const seen: Record<StateName, Uint8Array>[] = Array.from({ length: n }, () => ({
    run: new Uint8Array(samples.length),
    slow: new Uint8Array(samples.length),
    down: new Uint8Array(samples.length),
  }));
  samples.forEach((s, si) => {
    for (const sp of s.spans) {
      const a = Math.max(t0, sp.t0);
      const b = Math.min(t1, sp.t1);
      if (b <= a) continue;
      const i0 = Math.max(0, Math.floor((a - t0) / step));
      const i1 = Math.min(n - 1, Math.floor((b - 1 - t0) / step));
      for (let i = i0; i <= i1; i++) {
        if (seen[i][sp.state][si]) continue; // one path counts once per bin
        seen[i][sp.state][si] = 1;
        tally[i][sp.state] += 1;
      }
    }
  });
  for (let i = 0; i < n; i++) {
    out.push({
      t0: t0 + i * step,
      t1: Math.min(t1, t0 + (i + 1) * step),
      c: { run: tally[i].run / samples.length, slow: tally[i].slow / samples.length, down: tally[i].down / samples.length },
    });
  }
  return out;
}

function clusterPaths(samples: PathSample[]): PathCluster[] {
  // Group paths by the claim they make, not by the minute they place a block on.
  // The claim is: how much of the horizon was stopped, how much was slow, and
  // whether the stoppage came as one block or several. Where exactly inside a
  // report interval a block sits is precisely the uncertainty this app shows as
  // a boundary range; treating it as a different hypothesis would bury the one
  // distinction the counts cannot make (a slow hour vs. half an hour stopped)
  // under a dozen placement variants.
  interface Group {
    rep: PathSample;
    members: PathSample[];
    spans: { t0: number; t1: number }[];
    down: number;
    slow: number;
    stops: number;
  }
  const SUBSTANTIVE = 5 * 60_000;
  const downOf = (s: PathSample) =>
    mergeSpans(
      s.spans
        .filter((x) => x.state === 'down' && x.t1 - x.t0 >= SUBSTANTIVE)
        .map((x) => ({ t0: x.t0, t1: x.t1 })),
    );
  const bucket = (min: number) => Math.round(min / 10);
  const keyOf = (s: PathSample) => {
    const stops = downOf(s).length;
    return [bucket(s.minutes.down), bucket(s.minutes.slow), Math.min(3, stops)].join('|');
  };

  const raw = new Map<string, Group>();
  for (const s of samples) {
    const key = keyOf(s);
    const g = raw.get(key);
    if (g) g.members.push(s);
    else
      raw.set(key, {
        rep: s,
        members: [s],
        spans: downOf(s),
        down: s.minutes.down,
        slow: s.minutes.slow,
        stops: downOf(s).length,
      });
  }

  // Fold slivers into the nearest surviving group by (down, slow) distance; a
  // 2%-weight group is a rounding artifact, not a reading.
  const groups = [...raw.values()];
  groups.sort((a, b) => b.members.length - a.members.length);
  const kept: Group[] = [];
  const minWeight = Math.max(4, samples.length * 0.02);
  for (const g of groups) {
    if (kept.length === 0 || g.members.length >= minWeight) {
      kept.push(g);
      continue;
    }
    const nearest = kept.reduce((a, b) =>
      dist2(b, g) < dist2(a, g) ? b : a,
    );
    nearest.members.push(...g.members);
    nearest.down = nearest.members.reduce((s, m) => s + m.minutes.down, 0) / nearest.members.length;
    nearest.slow = nearest.members.reduce((s, m) => s + m.minutes.slow, 0) / nearest.members.length;
  }
  kept.sort((a, b) => b.members.length - a.members.length);
  const top = kept.slice(0, 4);

  const best = samples.reduce((a, b) => (b.logW > a.logW ? b : a), samples[0]);
  const topShare = top[0] ? top[0].members.length / samples.length : 0;

  return top.map((g, i) => {
    const rep = g.members.reduce((a, b) => (b.logW > a.logW ? b : a), g.members[0]);
    return {
      id: `hyp-${i}`,
      weight: g.members.length / samples.length,
      logGap: Math.max(0, best.logW - rep.logW),
      representative: rep,
      downSpans: g.spans,
      minutes: g.members.reduce(
        (acc, m) => {
          acc.run += m.minutes.run / g.members.length;
          acc.slow += m.minutes.slow / g.members.length;
          acc.down += m.minutes.down / g.members.length;
          return acc;
        },
        { run: 0, slow: 0, down: 0 } as Record<StateName, number>,
      ),
      downMin: g.down,
      slowMin: g.slow,
      stops: g.stops,
      diffuse: topShare < 0.45,
    };
  });
}

function dist2(a: { down: number; slow: number }, b: { down: number; slow: number }): number {
  return Math.abs(a.down - b.down) + Math.abs(a.slow - b.slow);
}

function mergeSpans(xs: { t0: number; t1: number }[]): { t0: number; t1: number }[] {
  const out: { t0: number; t1: number }[] = [];
  for (const x of xs) {
    const last = out[out.length - 1];
    if (last && x.t0 - last.t1 < 3 * 60_000) last.t1 = Math.max(last.t1, x.t1);
    else out.push({ ...x });
  }
  return out;
}

function dist(xs: number[]): { mean: number; lo: number; hi: number; samples: number[] } {
  if (!xs.length) return { mean: 0, lo: 0, hi: 0, samples: [] };
  const sorted = [...xs].sort((a, b) => a - b);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return {
    mean,
    lo: sorted[Math.floor(sorted.length * 0.1)],
    hi: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))],
    samples: xs,
  };
}

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const STATES_ALL = STATES;

export function spanStateLabel(s: SegmentKind): string {
  return s === 'product' ? '换型' : s === 'reset' ? '回零' : '正常';
}
