/**
 * Hidden-state inference for a cumulative counter that is read irregularly.
 *
 * Model
 * -----
 * The machine is in one of three states - running, slow, stopped. We never see
 * the path; for each pair of consecutive counter reads (a "segment") we see only
 * the cumulative gain. The latent variable per segment is therefore the *time
 * share* of the three states, and the observation is the gain:
 *
 *     gain ~ NegBinom( mean = dt * (f_run*r_run + f_slow*r_slow) )
 *
 * with a block-structured prior over shares and a tolerance for operator error.
 * Time shares cannot pin an event to a moment inside a segment, which is
 * deliberate: a long report interval gives a wide boundary range instead of a
 * fabricated timestamp.
 *
 * Several splits of one segment can explain the same gain equally well (a slow
 * hour and a half-stopped hour produce the same count). Those near-equivalent
 * paths are preserved by the path ensemble (paths.ts) and shown rather than
 * collapsed into a point.
 *
 * Count discontinuities (zeroing, product change) make the gain unknowable; those
 * segments fall back on prior + notes and are excluded from counted output.
 *
 * Emission rates are derived per product era from that product's own standard, so
 * a product change cannot inherit the previous standard's rates.
 */
import type {
  Analysis,
  BaselineAnalysis,
  CadencePrior,
  Constraint,
  Dataset,
  EraSummary,
  Hypothesis,
  InferredInterval,
  LossBucket,
  ProductRef,
  Segment,
  SegmentEvidence,
  StateName,
  Warning,
} from './types';
import { STATES } from './types';
import { buildSegGrid, mulberry32, PATH_SAMPLES, samplePaths } from './paths';

const MIN = 60_000;

export interface AnalyzeOptions {
  /** operator assertions that pin a window to "fully stopped" */
  constraints?: Constraint[];
}

/** Cycle inflation of the slow state relative to the standard. */
const SLOW_CYCLE_FACTOR = 2.6;
/** Overdispersion of the observed gain beyond counting noise. */
const NB_PHI = 0.13;
/** Operator timestamps are rounded to the minute and written a few minutes out. */
const MANUAL_JITTER_MIN = 3;
/** Probability that a manual downtime note is simply wrong about the interval. */
const MANUAL_SPURIOUS_P = 0.1;
/** Grid step over time shares. */
const SHARE_STEP = 0.05;
/** Mean spacing between stoppages; sets P(a report interval contains one). */
const DOWN_BLOCK_EVERY_MIN = 55;
/** Mean spacing between slow spells. */
const SLOW_BLOCK_EVERY_MIN = 45;
/** A fast line can beat its standard; a conflict is only claimed above this. */
const RUN_MAX_FACTOR = 1.6;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

export function analyze(dataset: Dataset, opts: AnalyzeOptions = {}): Analysis {
  const warnings: Warning[] = [];
  const segments = buildSegments(dataset, warnings);
  markStalls(segments, dataset);
  const priors = derivePriors(dataset);
  const constraints = normalizeConstraints(opts.constraints ?? [], segments, warnings);
  // A counter that undercounts by a steady few percent looks like permanent
  // downtime to a per-interval model. Estimate that bias once, up front, and
  // fold it into every emission and every blind-segment estimate.
  const counterBias = estimateCounterBias(segments, priors);
  if (counterBias < 0.99) {
    warnings.push({
      code: 'count-bias',
      text:
        `计数曲线的上包络约为标准的 ${(counterBias * 100).toFixed(0)}%，` +
        `更像是计数器漏计，而不是全长时段停机。已按该比例校正发射模型；无法校正的部分留在「未解释差额」。`,
    });
  }

  const posts = segments.map((seg) =>
    inferSegment(
      seg,
      priors.get(seg.era) ?? derivePrior(seg.product, seg.idealCycleMs),
      dataset,
      constraints,
      counterBias,
    ),
  );
  const rejectedNotes = checkConflicts(segments, dataset, constraints, priors, counterBias, warnings);

  const grids = segments.map((_, i) => buildSegGrid(segments[i], posts[i].grid, constraints));
  const paths = samplePaths(segments, grids, 0x5eed ^ segments.length, counterBias);
  const intervals = buildIntervals(paths.bins, paths.coverage, segments, dataset, constraints, rejectedNotes);
  const evidence = segments.map((seg, i) => toEvidence(seg, posts[i], paths.clusters));
  const totals = decomposeLoss(segments, posts, dataset, constraints);
  const eras = buildEraSummary(dataset, segments);

  const hypotheses: Hypothesis[] = paths.clusters.map((c, i) => ({
    id: c.id,
    label: describeCluster(c.downSpans, i),
    weight: c.weight,
    minutes: c.minutes,
    down: c.downSpans,
    logGap: c.logGap,
  }));

  return {
    segments,
    eras,
    intervals,
    evidence,
    buckets: totals.buckets,
    warnings,
    hypotheses,
    lossPosterior: {
      parts: paths.loss.mean,
      lo: paths.loss.lo,
      hi: paths.loss.hi,
      samples: paths.loss.samples,
    },
    horizonMin: segments.length ? (segments[segments.length - 1].t1 - segments[0].t0) / MIN : 0,
    idealParts: totals.idealParts,
    countedParts: totals.countedParts,
    blindParts: totals.blindParts,
    actualParts: totals.countedParts + totals.blindParts,
    lossParts: totals.lossParts,
    lossLo: totals.lossLo,
    lossHi: totals.lossHi,
    lossMin: totals.lossMin,
    unattributedMin: totals.unattributedMin,
    unattributedShare: totals.lossParts > 0 ? totals.unattributedParts / totals.lossParts : 0,
    constrained: constraints.length > 0,
    priorNote: summarizePriors(eras),
    modelVersion: 'taktlens-hmm 0.5',
  };
}

// ---------------------------------------------------------------------------
// Priors - one per product era, derived only from that standard
// ---------------------------------------------------------------------------

export function derivePrior(product: string, idealCycleMs: number): CadencePrior {
  const runMedian = 60_000 / idealCycleMs;
  return {
    source: 'product-standard',
    idealCycleMs,
    product,
    pRun: 0.78,
    pSlow: 0.1,
    pDown: 0.12,
    run: { median: runMedian, spread: 0.07 },
    slow: { median: runMedian / SLOW_CYCLE_FACTOR, spread: 0.16 },
    label: `${product} · 标准节拍 ${(idealCycleMs / 1000).toFixed(1)}s`,
  };
}

export function currentProduct(dataset: Dataset): ProductRef {
  const last = dataset.productChanges[dataset.productChanges.length - 1];
  return last ? { product: last.product, idealCycleMs: last.idealCycleMs } : dataset.initial;
}

export function derivePriors(dataset: Dataset): Map<number, CadencePrior> {
  const map = new Map<number, CadencePrior>();
  map.set(0, derivePrior(dataset.initial.product, dataset.initial.idealCycleMs));
  dataset.productChanges.forEach((c, i) => map.set(i + 1, derivePrior(c.product, c.idealCycleMs)));
  return map;
}

function eraRefs(dataset: Dataset): ProductRef[] {
  return [
    dataset.initial,
    ...dataset.productChanges.map((c) => ({ product: c.product, idealCycleMs: c.idealCycleMs })),
  ];
}

function eraAtTime(dataset: Dataset, t: number): number {
  let idx = 0;
  for (let i = 0; i < dataset.productChanges.length; i++) if (t >= dataset.productChanges[i].t) idx = i + 1;
  return idx;
}

function buildEraSummary(dataset: Dataset, segments: Segment[]): EraSummary[] {
  if (!segments.length) return [];
  const t0 = segments[0].t0;
  const t1 = segments[segments.length - 1].t1;
  const refs = eraRefs(dataset);
  const starts = [t0, ...dataset.productChanges.map((c) => c.t).filter((t) => t > t0 && t < t1)];
  const out: EraSummary[] = [];
  for (let i = 0; i < starts.length; i++) {
    const from = starts[i];
    const to = i + 1 < starts.length ? starts[i + 1] : t1;
    if (to <= from) continue;
    const era = eraAtTime(dataset, from + 1);
    const ref = refs[era] ?? refs[0];
    out.push({
      era,
      product: ref.product,
      idealCycleMs: ref.idealCycleMs,
      from,
      to,
      idealParts: ((to - from) / MIN) * (60_000 / ref.idealCycleMs),
      priorLabel: derivePrior(ref.product, ref.idealCycleMs).label,
    });
  }
  return out;
}

function summarizePriors(eras: EraSummary[]): string {
  if (!eras.length) return '';
  if (eras.length === 1) return `${eras[0].priorLabel}；发射参数由该标准推导`;
  return `${eras.map((e) => e.priorLabel).join(' → ')}；换型后按新标准重算发射参数（${eras.length} 套）`;
}

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

export function buildSegments(dataset: Dataset, warnings: Warning[] = []): Segment[] {
  const reports = dataset.reports;
  const refs = eraRefs(dataset);
  const segments: Segment[] = [];
  const changeTimes = dataset.productChanges.map((c) => c.t);

  for (let i = 0; i < reports.length - 1; i++) {
    const a = reports[i];
    const b = reports[i + 1];
    const dt = b.t - a.t;
    if (dt <= 0) {
      warnings.push({ code: 'non-monotonic', text: `上报 #${i + 1} 的时刻不晚于前一条，已跳过。` });
      continue;
    }
    const era = eraAtTime(dataset, b.t);
    const ref = refs[era] ?? refs[0];
    const notes: string[] = [];
    const changeInside = changeTimes.find((ct) => ct > a.t && ct <= b.t);
    let kind: Segment['kind'] = 'normal';
    let delta: number | null = b.cum - a.cum;

    if (changeInside) {
      kind = 'product';
      delta = null;
      notes.push('跨换型点：计数清零，段内产量不可拆分');
      const next = reports[i + 1];
      if (next && next.cum > 5) {
        warnings.push({
          code: 'change-without-reset',
          text: `换型后首条计数为 ${next.cum}，计数器未清零；换型窗口内产量不可知。`,
          segs: [segments.length],
        });
      }
    } else if (b.cum < a.cum) {
      kind = 'reset';
      delta = null;
      const dropRatio = (a.cum - b.cum) / Math.max(1, a.cum);
      notes.push(
        dropRatio > 0.5
          ? '累计计数回零（PLC 重启 / 班次清零）'
          : '计数小幅回落，按清零处理（也可能是计数修正）',
      );
    }

    const lowInfo = dt < 90_000;
    if (lowInfo) notes.push('间隔过短，计数取整噪声占主导');
    if (dt > 3 * 60 * MIN) notes.push(`间隔 ${(dt / (60 * MIN)).toFixed(1)} 小时，只能约束区间总量`);

    segments.push({
      id: segments.length,
      pair: i,
      era,
      product: ref.product,
      idealCycleMs: ref.idealCycleMs,
      t0: a.t,
      t1: b.t,
      dt,
      delta,
      kind,
      lowInfo,
      stall: false,
      notes,
    });
  }

  const long = segments.filter((s) => s.dt > 3 * 60 * MIN);
  if (long.length) {
    warnings.push({
      code: 'long-gap',
      text: `${long.length} 个上报间隔超过 3 小时；其间只有区间累计量可以约束状态，事件边界只能落到区间量级。`,
      segs: long.map((s) => s.id),
    });
  }
  const disc = segments.filter((s) => s.kind !== 'normal');
  if (disc.length) {
    warnings.push({
      code: 'discontinuity',
      text: `${disc.length} 段跨换型或回零，段内增量不可知；这些段不进入点估计，只给出范围。`,
      segs: disc.map((s) => s.id),
    });
  }
  if (dataset.downtime.length === 0) {
    warnings.push({ code: 'no-manual', text: '没有人工停机记录；停机边界完全由计数推断。' });
  }
  const zeroProd = segments.filter((s) => s.stall);
  if (zeroProd.length) {
    warnings.push({
      code: 'zero-gain',
      text:
        `${zeroProd.length} 个较长间隔内计数完全不动，且没有任何人工停机记录。它可能是真停机，也可能是计数器冻结后没有补记；` +
        `两种解释在计数上不可区分，这部分损失单列为「计数停滞」，不再算作已证实停机。`,
      segs: zeroProd.map((s) => s.id),
    });
  }
  return segments;
}

/**
 * Zero gain over a long stretch is the one pattern where the counts cannot tell
 * a stop from a broken counter. Mark it, but only when no operator note covers
 * the stretch - a noted stop is a stop, and a frozen counter usually has no note
 * because nobody knew it was frozen.
 */
function markStalls(segments: Segment[], dataset: Dataset): void {
  let i = 0;
  while (i < segments.length) {
    if (segments[i].kind !== 'normal' || segments[i].delta !== 0) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < segments.length && segments[j + 1].kind === 'normal' && segments[j + 1].delta === 0) j++;
    const t0 = segments[i].t0;
    const t1 = segments[j].t1;
    const noted = dataset.downtime.some((d) => d.end > t0 && d.start < t1);
    if (!noted && t1 - t0 >= 20 * MIN) {
      for (let k = i; k <= j; k++) {
        segments[k].stall = true;
        segments[k].notes.push('计数完全不动且无人工记录：停机与计数器冻结不可区分');
      }
    }
    i = j + 1;
  }
}

function normalizeConstraints(constraints: Constraint[], segments: Segment[], warnings: Warning[]): Constraint[] {
  if (!segments.length) return [];
  const lo = segments[0].t0;
  const hi = segments[segments.length - 1].t1;
  const out: Constraint[] = [];
  for (const c of constraints) {
    if (!(c.end > c.start)) {
      warnings.push({ code: 'bad-constraint', text: `约束「${c.label}」起止时间无效，已忽略。` });
      continue;
    }
    const a = Math.max(lo, c.start);
    const b = Math.min(hi, c.end);
    if (b - a < 30_000) {
      warnings.push({ code: 'bad-constraint', text: `约束「${c.label}」不在计数记录范围内，已忽略。` });
      continue;
    }
    out.push({ ...c, start: a, end: b });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Segment posterior over time shares
// ---------------------------------------------------------------------------

interface GridPoint {
  fr: number;
  fs: number;
  fd: number;
  logW: number;
  weight: number;
}

interface SegPosterior {
  mean: Record<StateName, number>;
  minutes: Record<StateName, number>;
  minutesBand: Record<StateName, [number, number]>;
  hypotheses: Hypothesis[];
  ambiguity: number;
  evidence: string;
  grid: GridPoint[];
}

/**
 * How much of the standard the counter actually registers at full speed. The
 * upper envelope of observed rates is the best evidence of "this line really was
 * running"; if even that sits below the standard, the gap is a counter bias, not
 * downtime spread across the whole day. Clamped hard, because a genuinely slow
 * day must not be explained away as a broken counter.
 */
function estimateCounterBias(segments: Segment[], priors: Map<number, CadencePrior>): number {
  const ratios: number[] = [];
  for (const seg of segments) {
    if (seg.kind !== 'normal' || seg.delta === null || seg.dt < 5 * MIN) continue;
    const prior = priors.get(seg.era);
    if (!prior) continue;
    const obs = seg.delta / (seg.dt / MIN);
    ratios.push(obs / prior.run.median);
  }
  if (ratios.length < 3) return 1;
  ratios.sort((a, b) => a - b);
  const p95 = ratios[Math.min(ratios.length - 1, Math.floor(ratios.length * 0.95))];
  // The envelope can exceed the standard (a fast line, or a fleet of short
  // intervals where rounding flatters the rate); only a shortfall is a bias.
  return Math.min(1, Math.max(0.8, p95));
}

function inferSegment(
  seg: Segment,
  prior: CadencePrior,
  dataset: Dataset,
  constraints: Constraint[],
  counterBias: number,
): SegPosterior {
  const T = seg.dt / MIN;
  const pinned = pinnedDownMinutes(seg, constraints);
  const overlap = overlapWithManual(seg, dataset);
  const free = Math.max(0, T - pinned);
  const hasCount = seg.delta !== null && seg.delta >= 0;

  const points: GridPoint[] = [];
  const steps = Math.round(1 / SHARE_STEP);
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; i + j <= steps; j++) {
      const fr = i / steps;
      const fs = j / steps;
      const fd = 1 - fr - fs;
      let logW = blockPrior(fr, fs, fd, T);
      if (hasCount) {
        const expected = free * (fr * prior.run.median + fs * prior.slow.median) * counterBias;
        logW += nbLogPmf(seg.delta as number, expected);
      } else {
        // no count evidence: the note is the only direct observation
        logW += manualLogLikelihood(fd * free + pinned, overlap);
      }
      points.push({ fr, fs, fd, logW, weight: 0 });
    }
  }
  normalizeLogWeights(points);

  const mean: Record<StateName, number> = { run: 0, slow: 0, down: 0 };
  const minutes: Record<StateName, number> = { run: 0, slow: 0, down: 0 };
  for (const p of points) {
    const r = p.fr * free;
    const s = p.fs * free;
    const d = p.fd * free + pinned;
    mean.run += p.weight * (r / T);
    mean.slow += p.weight * (s / T);
    mean.down += p.weight * (d / T);
    minutes.run += p.weight * r;
    minutes.slow += p.weight * s;
    minutes.down += p.weight * d;
  }
  const band = weightedBand(points, free, pinned);
  const hypotheses = topHypotheses(points, free, pinned);
  const ambiguity = 1 - Math.max(mean.run, mean.slow, mean.down);
  const evidence = describeEvidence(seg, overlap, pinned);

  return { mean, minutes, minutesBand: band, hypotheses, ambiguity, evidence, grid: points };
}

/**
 * Warnings for the cases where a note and the counts genuinely cannot both hold.
 * Only report segments that sit *entirely* inside the stated window: a note is
 * rounded to the minute and written a few minutes out, so production in a
 * partially covered segment may simply come from just outside it. One warning
 * per record, not per segment.
 */
function checkConflicts(
  segments: Segment[],
  dataset: Dataset,
  constraints: Constraint[],
  priors: Map<number, CadencePrior>,
  counterBias: number,
  warnings: Warning[],
): Set<string> {
  const rejected = new Set<string>();
  const fullyInside = (a: number, b: number) =>
    segments.filter((s) => s.delta !== null && s.t0 >= a && s.t1 <= b);
  const ceilingFor = (hit: Segment[]) => {
    const prior = priors.get(hit[0].era) ?? derivePrior(hit[0].product, hit[0].idealCycleMs);
    const mins = hit.reduce((acc, s) => acc + s.dt / MIN, 0);
    return mins * prior.run.median * counterBias * RUN_MAX_FACTOR + 3;
  };

  for (const c of constraints) {
    const hit = fullyInside(c.start, c.end);
    if (!hit.length) continue;
    const counted = hit.reduce((a, s) => a + (s.delta as number), 0);
    // A confirmed stop is a positive assertion that nothing was made. Only a
    // token allowance is left for boundary rounding - this is a much tighter
    // test than the one applied to an informal note.
    const ideal = hit.reduce((a, s) => a + (s.dt / MIN) * (60_000 / s.idealCycleMs), 0);
    const allowance = 0.05 * ideal + 3;
    if (counted > allowance) {
      warnings.push({
        code: 'constraint-conflict',
        text:
          `已确认停机「${c.label}」期间计数仍增加 ${Math.round(counted)} 件，而全停窗口内本应几乎为零。` +
          `冲突涉及计数段 #${hit.map((s) => s.id).join(', #')}；请复核这些段或该确认。`,
        segs: hit.map((s) => s.id),
      });
    }
  }

  for (const d of dataset.downtime) {
    if (constraints.some((c) => c.start < d.end && c.end > d.start)) continue;
    const hit = fullyInside(d.start + MANUAL_JITTER_MIN * MIN, d.end - MANUAL_JITTER_MIN * MIN);
    if (!hit.length) continue;
    const counted = hit.reduce((a, s) => a + (s.delta as number), 0);
    if (counted > ceilingFor(hit)) {
      rejected.add(d.id);
      warnings.push({
        code: 'manual-conflict',
        text:
          `人工停机记录「${d.note}」（${fmtClock(d.start)}–${fmtClock(d.end)}）称该窗口停机，` +
          `但窗口内计数增加 ${Math.round(counted)} 件，超过该时长的物理上限。` +
          `冲突涉及计数段 #${hit.map((s) => s.id).join(', #')}；按计数优先，该记录不用于压缩边界。`,
        segs: hit.map((s) => s.id),
      });
    }
  }
  return rejected;
}

function pinnedDownMinutes(seg: Segment, constraints: Constraint[]): number {
  let overlap = 0;
  for (const c of constraints) {
    const a = Math.max(seg.t0, c.start);
    const b = Math.min(seg.t1, c.end);
    if (b > a) overlap += (b - a) / MIN;
  }
  return Math.min(seg.dt / MIN, overlap);
}

function overlapWithManual(seg: Segment, dataset: Dataset): number {
  let m = 0;
  for (const d of dataset.downtime) {
    const a = Math.max(seg.t0, d.start);
    const b = Math.min(seg.t1, d.end);
    if (b > a) m += (b - a) / MIN;
  }
  return Math.min(seg.dt / MIN, m);
}

/**
 * Stops arrive as blocks, not as a drizzle: a report interval either holds none
 * of a stoppage or a sizeable share of it. A Dirichlet centred on the average
 * occupancy would smear a few percent of downtime into every clean interval and
 * make everything look slightly stopped - the exact failure this app exists to
 * avoid. So the prior is zero-inflated on the down and slow shares.
 */
function blockPrior(fr: number, fs: number, fd: number, T: number): number {
  const pAnyDown = 1 - Math.exp(-T / DOWN_BLOCK_EVERY_MIN);
  const pAnySlow = 1 - Math.exp(-T / SLOW_BLOCK_EVERY_MIN);
  const eps = 1e-3;
  // A share node stands for a cell of width SHARE_STEP, so the density is read at
  // the cell's centre. A Beta with both shapes above 1 is exactly zero at 1:
  // reading the node at face value would give "the entire interval ran slow" no
  // prior mass at all, which is an artifact of the grid, not a belief about the
  // line - and it would hide every slow spell that spans a whole report interval.
  const mid = (x: number) => (x <= eps ? 0 : x >= 1 - 1e-9 ? 1 - 0.5 * SHARE_STEP : x + 0.5 * SHARE_STEP);
  let lp = 0;
  if (fd <= eps) lp += Math.log(1 - pAnyDown);
  else lp += Math.log(pAnyDown) + Math.log(Math.max(eps, betaDensity(mid(fd), 1.2, 1.1)));
  if (fs <= eps) lp += Math.log(1 - pAnySlow);
  else lp += Math.log(pAnySlow) + Math.log(Math.max(eps, betaDensity(mid(fs), 1.3, 1.15)));
  if (fr < -eps) lp += -1e6;
  return lp;
}

/** Unnormalised Beta density; the constant cancels inside the posterior. */
function betaDensity(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0;
  return Math.pow(x, a - 1) * Math.pow(1 - x, b - 1);
}

/** Negative-binomial log-PMF: counting noise + rate overdispersion.
 *  pmf = C(k+r-1, k) p^r (1-p)^k with mean = r(1-p)/p. */
function nbLogPmf(k: number, mean: number): number {
  if (mean < 1e-6) return k === 0 ? 0 : -1e3;
  const phi = NB_PHI;
  const variance = mean + phi * phi * mean * mean;
  const r = (mean * mean) / Math.max(1e-9, variance - mean);
  const p = r / (r + mean);
  return logGamma(k + r) - logGamma(r) - logGamma(k + 1) + r * Math.log(p) + k * Math.log(1 - p);
}

function manualLogLikelihood(downMin: number, overlapMin: number): number {
  if (overlapMin <= 0) return 0;
  const deficit = Math.max(0, overlapMin - MANUAL_JITTER_MIN - downMin);
  const penalty = -0.5 * Math.pow(deficit / 3, 2);
  return logAddExp(Math.log(1 - MANUAL_SPURIOUS_P) + penalty, Math.log(MANUAL_SPURIOUS_P));
}

function logAddExp(a: number, b: number): number {
  const m = Math.max(a, b);
  if (!isFinite(m)) return m;
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

function normalizeLogWeights(points: GridPoint[]): void {
  if (!points.length) return;
  const max = points.reduce((m, p) => Math.max(m, p.logW), -Infinity);
  let sum = 0;
  for (const p of points) {
    p.weight = Math.exp(p.logW - max);
    sum += p.weight;
  }
  if (sum <= 0) {
    for (const p of points) p.weight = 1 / points.length;
    return;
  }
  for (const p of points) p.weight /= sum;
}

function weightedBand(points: GridPoint[], free: number, pinned: number): Record<StateName, [number, number]> {
  const acc: Record<StateName, [number, number]> = { run: [0, 0], slow: [0, 0], down: [0, 0] };
  for (const st of STATES) {
    const rows = points
      .map((p) => ({
        m: (st === 'run' ? p.fr : st === 'slow' ? p.fs : p.fd) * free + (st === 'down' ? pinned : 0),
        w: p.weight,
      }))
      .sort((a, b) => a.m - b.m);
    let cum = 0;
    let lo = rows[0]?.m ?? 0;
    let hi = rows[rows.length - 1]?.m ?? 0;
    for (const r of rows) {
      cum += r.w;
      if (cum >= 0.1) {
        lo = r.m;
        break;
      }
    }
    cum = 0;
    for (const r of rows) {
      cum += r.w;
      if (cum >= 0.9) {
        hi = r.m;
        break;
      }
    }
    acc[st] = [Math.max(0, lo), Math.max(0, lo, hi)];
  }
  return acc;
}

/** Coarse bins of the posterior over time shares: the near-equivalent splits. */
function topHypotheses(points: GridPoint[], free: number, pinned: number): Hypothesis[] {
  // Bins collect the weighted mean shares rather than one rounded representative.
  // Rounding fr and fs separately would let fr + fs exceed 1 and the split would
  // then describe more minutes than the segment has.
  const bins = new Map<string, { fr: number; fs: number; w: number; sr: number; ss: number; sd: number }>();
  for (const p of points) {
    const key = `${Math.round(p.fr * 10)}:${Math.round(p.fs * 10)}`;
    const b = bins.get(key) ?? { fr: 0, fs: 0, w: 0, sr: 0, ss: 0, sd: 0 };
    b.w += p.weight;
    b.sr += p.weight * p.fr;
    b.ss += p.weight * p.fs;
    b.sd += p.weight * p.fd;
    bins.set(key, b);
  }
  for (const b of bins.values()) {
    if (b.w > 0) {
      b.fr = b.sr / b.w;
      b.fs = b.ss / b.w;
    }
  }
  const sorted = [...bins.values()].sort((a, b) => b.w - a.w);
  const picked: typeof sorted = [];
  for (const b of sorted) {
    if (b.w < 0.03) break;
    if (picked.some((q) => Math.abs(q.fr - b.fr) + Math.abs(q.fs - b.fs) < 0.25)) continue;
    picked.push(b);
    if (picked.length >= 4) break;
  }
  return picked.map((b, i) => {
    const fd = b.w > 0 ? b.sd / b.w : Math.max(0, 1 - b.fr - b.fs);
    return {
      id: `split-${i}`,
      label: `运行 ${pct(b.fr)} · 低速 ${pct(b.fs)} · 停机 ${pct(fd)}`,
      weight: b.w,
      minutes: { run: b.fr * free, slow: b.fs * free, down: fd * free + pinned },
      down: [],
      logGap: 0,
    };
  });
}

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

function describeEvidence(seg: Segment, manualMin: number, pinnedMin: number): string {
  if (seg.kind === 'product') return '跨换型：计数清零，无增量可用';
  if (seg.kind === 'reset') return '计数回零：无增量可用';
  if (seg.delta === null) return '无可用增量';
  const perMin = seg.delta / (seg.dt / MIN);
  const bits = [`+${Math.round(seg.delta)} 件 / ${(seg.dt / MIN).toFixed(1)} 分 = ${perMin.toFixed(2)} 件/分`];
  if (manualMin > 0) bits.push(`人工记录覆盖 ${manualMin.toFixed(0)} 分`);
  if (pinnedMin > 0) bits.push(`已确认停机 ${pinnedMin.toFixed(0)} 分`);
  return bits.join(' · ');
}

// ---------------------------------------------------------------------------
// Intervals: contiguous runs of one dominant state, ranges from the ensemble
// ---------------------------------------------------------------------------

/** Stickiness of the state label between adjacent 2-minute bins, in log units.
 *  Without it the arg-max label flickers between run/slow/down inside one long
 *  ambiguous interval and every flicker becomes a fake event. */
const BIN_STICKINESS = 2.2;
/** How far a boundary may migrate into a neighbouring run before that run's own
 *  label would have to change. */
const MAX_BOUNDARY_EXTEND_BINS = 24;

type Bin = { t0: number; t1: number; p: Record<StateName, number> };
type Cov = { t0: number; t1: number; c: Record<StateName, number> };
interface Run {
  state: StateName;
  from: number;
  to: number;
}

function buildIntervals(
  bins: Bin[],
  coverage: Cov[],
  segments: Segment[],
  dataset: Dataset,
  constraints: Constraint[],
  rejectedNotes: Set<string> = new Set(),
): InferredInterval[] {
  if (!bins.length || !segments.length) return [];

  // 1. Viterbi over the bins: states are sticky, so a one-bin blip cannot split
  //    a stop into two intervals with a fake run between them.
  const labels = viterbi(bins.map((b) => b.p));
  let runs: Run[] = [];
  for (let i = 0; i < bins.length; i++) {
    const last = runs[runs.length - 1];
    if (last && last.state === labels[i]) last.to = i;
    else runs.push({ state: labels[i], from: i, to: i });
  }

  // 2. A short, uncertain blip is not an event. Fold it into the neighbouring
  //    run it most resembles, then relabel every run by its own mass. This is
  //    what stops a slow-vs-down stretch from being reported as four alternating
  //    intervals; the surviving interval carries the alternative instead.
  runs = mergeWeakRuns(runs, bins);
  runs = relabelRuns(runs, bins);

  // 3. Cuts between runs are shared: one range per junction, used as the end of
  //    the left interval and the start of the right one. Ranges therefore never
  //    overlap and the timeline is a partition even when every boundary is wide.
  const horizonLo = bins[0].t0;
  const horizonHi = bins[bins.length - 1].t1;
  const cuts: { lo: number; hi: number }[] = [{ lo: horizonLo, hi: horizonLo }];
  for (let k = 1; k < runs.length; k++) cuts.push(cutBetween(coverage, runs[k - 1], runs[k]));
  cuts.push({ lo: horizonHi, hi: horizonHi });

  // 3b. Operator anchors narrow a junction to the minute. They are applied to
  // the shared cut, never to one side, so a pinned boundary still partitions
  // the timeline: the end of one stretch and the start of the next stay equal.
  // A constraint is a direct assertion about the line, so it is anchored first
  // and for any run it touches - confirming a stretch the model called "slow"
  // must still work. Manual notes are weaker, and only anchor a run the model
  // already believes is a stop.
  const accepted = dataset.downtime.filter((d) => !rejectedNotes.has(d.id));
  const pinnedCut = new Set<number>();
  const pinAt = (k: number, t: number) => {
    cuts[k] = { lo: t, hi: t };
    pinnedCut.add(k);
  };
  const center = (k: number) => (cuts[k].lo + cuts[k].hi) / 2;
  for (const c of constraints) {
    // A confirmed stop has a start and an end; each belongs at a junction. Pick
    // the ordered pair of junctions whose guesses are closest to the two edges,
    // so both ends are pinned and the stretch between them is the confirmed one.
    let bestA = -1;
    let bestB = -1;
    let bestCost = Infinity;
    for (let a = 1; a < cuts.length - 1; a++) {
      for (let b = a + 1; b < cuts.length - 1; b++) {
        const cost = Math.abs(center(a) - c.start) + Math.abs(center(b) - c.end);
        if (cost < bestCost) {
          bestCost = cost;
          bestA = a;
          bestB = b;
        }
      }
    }
    if (bestA < 0) continue;
    // A confirmation retimes a junction; it does not delete one or invent one.
    // So an edge is pinned only where it falls between the neighbouring
    // junctions - the stretch the model already set aside as one interval. An
    // edge that would need a junction moved past its neighbour means the model
    // and the operator disagree about the shape of the day; the conflict check
    // reports that rather than the anchor rewriting it.
    const canPin = (k: number, edge: number) => edge > cuts[k - 1].hi && edge < cuts[k + 1].lo;
    if (canPin(bestA, c.start)) pinAt(bestA, c.start);
    if (canPin(bestB, c.end)) pinAt(bestB, c.end);
  }
  for (let i = 0; i < runs.length; i++) {
    if (runs[i].state !== 'down') continue;
    const a = anchorNear(cuts[i], [], accepted, horizonLo, horizonHi, 'start');
    if (a && !pinnedCut.has(i) && a.cut.lo !== a.cut.hi) cuts[i] = a.cut;
    const b = anchorNear(cuts[i + 1], [], accepted, horizonLo, horizonHi, 'end');
    if (b && !pinnedCut.has(i + 1) && b.cut.lo !== b.cut.hi) cuts[i + 1] = b.cut;
  }
  // Anchoring two junctions of one run can cross the other's range, and a pin
  // can land on the far side of a neighbour's guess. Repair the whole chain left
  // to right once, moving the unpinned side of any crossed pair, so the output
  // is a partition no matter how the anchors fell.
  for (let k = 1; k < cuts.length; k++) {
    if (cuts[k].lo >= cuts[k - 1].hi) continue;
    if (pinnedCut.has(k)) {
      cuts[k - 1] = { lo: Math.min(cuts[k - 1].lo, cuts[k].lo), hi: cuts[k].lo };
    } else if (pinnedCut.has(k - 1)) {
      cuts[k] = { lo: cuts[k - 1].hi, hi: Math.max(cuts[k].hi, cuts[k - 1].hi) };
    } else {
      const mid = (cuts[k - 1].hi + cuts[k].lo) / 2;
      cuts[k - 1] = { lo: cuts[k - 1].lo, hi: mid };
      cuts[k] = { lo: mid, hi: cuts[k].hi };
    }
  }

  const out: InferredInterval[] = [];
  for (let i = 0; i < runs.length; i++) {
    const run = runs[i];
    const cells = bins.slice(run.from, run.to + 1);
    const t0 = cells[0].t0;
    const t1 = cells[cells.length - 1].t1;
    const share = mean(cells.map((c) => c.p[run.state]));
    const { alternatives, expectedMin } = summarizeCells(cells, run.state);

    const startLo = cuts[i].lo;
    const startHi = cuts[i].hi;
    const endLo = cuts[i + 1].lo;
    const endHi = cuts[i + 1].hi;

    // How much of the ensemble places this state at the middle of the run. The
    // edges are covered by the cut ranges; the centre is what tells us whether
    // this stretch is a fact (nearly every path) or one of several readings.
    const centerBin = bins[Math.floor((run.from + run.to) / 2)];
    const pathShare = coverageFind(coverage, (centerBin.t0 + centerBin.t1) / 2)?.c[run.state] ?? 0;

    let constrained = false;
    if (run.state === 'down') {
      constrained =
        isPinned(cuts[i], constraints) ||
        isPinned(cuts[i + 1], constraints);
    }
    const manualIds =
      run.state === 'down'
        ? dataset.downtime.filter((d) => d.end > t0 - 6 * MIN && d.start < t1 + 6 * MIN).map((d) => d.id)
        : [];

    const ambiguous =
      !constrained &&
      (startHi - startLo > 2 * MIN ||
        endHi - endLo > 2 * MIN ||
        pathShare < 0.55 ||
        alternatives.length > 0 ||
        (run.state === 'down' && manualIds.length === 0));

    out.push({
      id: `iv-${i}`,
      state: run.state,
      kind: share >= 0.5 ? 'dominant' : 'suspect',
      alternatives,
      confidence: share,
      pathShare,
      startLo,
      startHi,
      endLo,
      endHi,
      expectedMin,
      segs: segments.filter((s) => s.t1 > t0 && s.t0 < t1).map((s) => s.id),
      manualIds,
      ambiguous,
      constrained,
    });
  }
  return out;
}

function isPinned(cut: { lo: number; hi: number }, constraints: Constraint[]): boolean {
  return constraints.some((c) => Math.abs(cut.lo - cut.hi) < 1 && (Math.abs(c.start - cut.lo) < MIN || Math.abs(c.end - cut.lo) < MIN));
}

/** Prefer an operator constraint, then a manual note, then leave the cut. */
function anchorNear(
  cut: { lo: number; hi: number },
  constraints: Constraint[],
  notes: { id: string; start: number; end: number }[],
  horizonLo: number,
  horizonHi: number,
  edge: 'start' | 'end',
): { cut: { lo: number; hi: number } } | null {
  const near = (t: number) => t >= cut.lo - 6 * MIN && t <= cut.hi + 6 * MIN;
  const c = constraints.find((k) => near(edge === 'start' ? k.start : k.end));
  if (c) {
    const t = edge === 'start' ? c.start : c.end;
    return { cut: { lo: t, hi: t } };
  }
  if (cut.lo === horizonLo || cut.hi === horizonHi) return null;
  const n = notes.find((d) => near(edge === 'start' ? d.start : d.end));
  if (!n) return null;
  const t = edge === 'start' ? n.start : n.end;
  const lo = Math.max(horizonLo, t - MANUAL_JITTER_MIN * MIN);
  const hi = Math.min(horizonHi, t + MANUAL_JITTER_MIN * MIN);
  return { cut: { lo, hi } };
}

function coverageFind(coverage: Cov[], t: number, tol = MIN): Cov | undefined {
  let best: Cov | undefined;
  let bestD = Infinity;
  for (const c of coverage) {
    if (t >= c.t0 - tol && t <= c.t1 + tol) {
      const d = Math.abs((c.t0 + c.t1) / 2 - t);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
  }
  return best;
}

function summarizeCells(
  cells: Bin[],
  state: StateName,
): { alternatives: { state: StateName; share: number }[]; expectedMin: number; duration: number } {
  const total = cells.reduce((s, c) => s + (c.t1 - c.t0), 0);
  const mass: Record<StateName, number> = { run: 0, slow: 0, down: 0 };
  let expectedMin = 0;
  for (const c of cells) {
    const dur = (c.t1 - c.t0) / MIN;
    for (const st of STATES) mass[st] += dur * c.p[st];
    expectedMin += dur * c.p[state];
  }
  const alternatives = STATES.filter((st) => st !== state)
    .map((st) => ({ state: st, share: total > 0 ? (mass[st] * MIN) / total : 0 }))
    .filter((a) => a.share >= 0.15)
    .sort((a, b) => b.share - a.share);
  return { alternatives, expectedMin, duration: total / MIN };
}

/** Fold short, low-confidence blips into a neighbour. A real stop survives this
 *  (it is confident); a flicker of the label inside an ambiguous stretch does
 *  not, because it carries neither duration nor conviction. */
function mergeWeakRuns(runs: Run[], bins: Bin[]): Run[] {
  const durOf = (r: Run) => (bins[r.to].t1 - bins[r.from].t0) / MIN;
  const confOf = (r: Run) => mean(bins.slice(r.from, r.to + 1).map((c) => c.p[r.state]));
  let list = [...runs];
  for (let guard = 0; guard < 16 && list.length > 1; guard++) {
    let idx = -1;
    let worst = Infinity;
    for (let i = 0; i < list.length; i++) {
      const r = list[i];
      // a run at the very edge has nowhere to fold into unless it is tiny
      const d = durOf(r);
      const c = confOf(r);
      const weak = d < 4 || (d < 12 && c < 0.62);
      if (!weak) continue;
      if (i === 0 && d > 3) continue;
      if (i === list.length - 1 && d > 3) continue;
      const score = d * (1.2 - c);
      if (score < worst) {
        worst = score;
        idx = i;
      }
    }
    if (idx < 0) break;
    const left = list[idx - 1];
    const right = list[idx + 1];
    let target: Run;
    if (!left) target = right;
    else if (!right) target = left;
    else {
      const lMass = confOf(left) * (left.to - left.from + 1);
      const rMass = confOf(right) * (right.to - right.from + 1);
      target = lMass >= rMass ? left : right;
    }
    const a = Math.min(list[idx].from, target.from);
    const b = Math.max(list[idx].to, target.to);
    const merged: Run = { state: target.state, from: a, to: b };
    const next: Run[] = [];
    for (const r of list) {
      if (r === target) next.push(merged);
      else if (r !== list[idx]) next.push(r);
    }
    list = next;
  }
  return list;
}

/** Relabel each run by the state with the most expected minutes, then join
 *  neighbours that ended up with the same label. */
function relabelRuns(runs: Run[], bins: Bin[]): Run[] {
  const labelled = runs.map((r) => {
    const mass: Record<StateName, number> = { run: 0, slow: 0, down: 0 };
    for (let i = r.from; i <= r.to; i++) {
      const dur = (bins[i].t1 - bins[i].t0) / MIN;
      for (const st of STATES) mass[st] += dur * bins[i].p[st];
    }
    return { ...r, state: argmaxState(mass) };
  });
  const out: Run[] = [];
  for (const r of labelled) {
    const last = out[out.length - 1];
    if (last && last.state === r.state) last.to = r.to;
    else out.push({ ...r });
  }
  return out;
}

function argmaxState(p: Record<StateName, number>): StateName {
  let best: StateName = 'run';
  for (const s of STATES) if (p[s] > p[best]) best = s;
  return best;
}

/**
 * One cut per junction. The range is where the boundary could plausibly sit:
 * as far left as the incoming state still has material presence, up to the
 * first bin where it is the majority. Migration is capped at the neighbouring
 * run's midpoint, because a boundary that moved further would require the
 * neighbour's own label to change - which it did not.
 */
function cutBetween(coverage: Cov[], left: Run, right: Run): { lo: number; hi: number } {
  const B = right.state;
  const c = right.from;
  const leftMid = Math.max(left.from, Math.floor((left.from + left.to) / 2));
  const rightMid = Math.min(right.to, Math.floor((right.from + right.to) / 2));
  const minL = Math.max(leftMid, c - MAX_BOUNDARY_EXTEND_BINS);
  const maxR = Math.min(rightMid, c + MAX_BOUNDARY_EXTEND_BINS);

  let cred = c;
  while (cred < maxR && coverage[cred].c[B] < 0.5) cred++;
  let lo = cred;
  while (lo > minL && coverage[lo - 1].c[B] >= 0.12) lo--;

  const loT = Math.max(coverage[0].t0, coverage[lo].t0);
  const hiT = Math.max(loT, coverage[cred].t0);
  return { lo: loT, hi: hiT };
}

function viterbi(ps: Record<StateName, number>[]): StateName[] {
  if (!ps.length) return [];
  const n = ps.length;
  const K = STATES.length;
  const dp: number[][] = Array.from({ length: n }, () => new Array(K).fill(-Infinity));
  const back: number[][] = Array.from({ length: n }, () => new Array(K).fill(0));
  const lp = (p: Record<StateName, number>, k: number) => Math.log(Math.max(1e-6, p[STATES[k]]));
  for (let k = 0; k < K; k++) dp[0][k] = lp(ps[0], k);
  for (let i = 1; i < n; i++) {
    for (let k = 0; k < K; k++) {
      let best = -Infinity;
      let arg = 0;
      for (let j = 0; j < K; j++) {
        const v = dp[i - 1][j] - (j === k ? 0 : BIN_STICKINESS);
        if (v > best) {
          best = v;
          arg = j;
        }
      }
      dp[i][k] = best + lp(ps[i], k);
      back[i][k] = arg;
    }
  }
  let bestK = 0;
  for (let k = 1; k < K; k++) if (dp[n - 1][k] > dp[n - 1][bestK]) bestK = k;
  const out = new Array<StateName>(n);
  out[n - 1] = STATES[bestK];
  let cur = bestK;
  for (let i = n - 1; i > 0; i--) {
    cur = back[i][cur];
    out[i - 1] = STATES[cur];
  }
  return out;
}

function mean(xs: number[]): number {
  if (!xs.length) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

// ---------------------------------------------------------------------------
// Segment evidence rows
// ---------------------------------------------------------------------------

function toEvidence(seg: Segment, post: SegPosterior, clusters: { downSpans: { t0: number; t1: number }[]; weight: number; representative: { minutes: Record<StateName, number>; logW: number } }[]): SegmentEvidence {
  const T = seg.dt / MIN;
  const step = 60_000 / seg.idealCycleMs;
  const idealParts = T * step;
  const actual = seg.delta ?? post.minutes.run * step + post.minutes.slow * step / SLOW_CYCLE_FACTOR;
  const loss = Math.max(0, idealParts - actual);
  const half = Math.max(1, post.minutesBand.down[1] - post.minutesBand.down[0]) * step;
  // every near-equivalent split inside this segment, so the drawer can show the
  // competing readings even when no ensemble cluster places a stop here.
  // These minutes are segment-local: they partition this segment's own span.
  const splits: Hypothesis[] = post.hypotheses.map((h, i) => ({
    id: `seg-${seg.id}-s${i}`,
    label:
      h.minutes.down >= 1
        ? `停机 ${Math.round(h.minutes.down)} 分 / 低速 ${Math.round(h.minutes.slow)} 分`
        : `低速 ${Math.round(h.minutes.slow)} 分 / 运行 ${Math.round(h.minutes.run)} 分`,
    weight: h.weight,
    minutes: h.minutes,
    down: [],
    logGap: 0,
  }));
  // only the path hypotheses that put a stop inside this segment are relevant
  const hyps = clusters
    .filter((c) => c.downSpans.some((d) => d.t1 > seg.t0 && d.t0 < seg.t1))
    .slice(0, 3)
    .map((c, i) => ({
      id: `seg-${seg.id}-h${i}`,
      label: `停机 ${c.downSpans.filter((d) => d.t1 > seg.t0 && d.t0 < seg.t1).map((d) => `${fmtClock(Math.max(seg.t0, d.t0))}–${fmtClock(Math.min(seg.t1, d.t1))}`).join('、')}`,
      weight: c.weight,
      minutes: c.representative.minutes,
      down: c.downSpans,
      logGap: 0,
    }));
  // Competing readings, one per state: several splits can name the same state,
  // and listing "停机 8%、停机 5%" twice reads as two different claims about it.
  const best = argmaxState(post.minutes);
  const altByState = new Map<StateName, number>();
  for (const h of splits) {
    const s = argmaxState(h.minutes);
    if (s === best) continue;
    altByState.set(s, (altByState.get(s) ?? 0) + h.weight);
  }
  const alternatives = [...altByState.entries()]
    .map(([state, share]) => ({ state, share }))
    .sort((a, b) => b.share - a.share);
  return {
    seg: seg.id,
    t0: seg.t0,
    t1: seg.t1,
    dtMin: T,
    delta: seg.delta,
    idealParts,
    lossParts: loss,
    lossLo: Math.max(0, loss - half * 0.8),
    lossHi: loss + half * 0.8 + idealParts * 0.04,
    p: post.mean,
    minutes: post.minutes,
    minutesBand: post.minutesBand,
    alternatives,
    splits,
    hypotheses: hyps.length ? hyps : splits,
    notes: [...seg.notes, post.evidence],
  };
}

// ---------------------------------------------------------------------------
// Loss decomposition
// ---------------------------------------------------------------------------

interface LossTotals {
  buckets: LossBucket[];
  idealParts: number;
  countedParts: number;
  blindParts: number;
  lossParts: number;
  lossLo: number;
  lossHi: number;
  lossMin: number;
  unattributedMin: number;
  unattributedParts: number;
}

const SLOW_RATIO = 1 / SLOW_CYCLE_FACTOR;

function decomposeLoss(
  segments: Segment[],
  posts: SegPosterior[],
  dataset: Dataset,
  constraints: Constraint[],
): LossTotals {
  let idealParts = 0;
  let countedParts = 0;
  let blindParts = 0;
  let recordedDown = 0;
  let unrecordedDown = 0;
  let slowLoss = 0;
  let changeover = 0;
  let normalLoss = 0;
  let unattributedMin = 0;
  // Bands are combined in quadrature, not by summing each segment's 10-90
  // width. Summing widths would assume every interval sits at its own edge at
  // once and would widen the day's range with the number of report intervals,
  // which is not what "uncertain" means here.
  let downVar = 0;
  let slowVar = 0;
  let blindLo = 0;
  let blindHi = 0;
  let stallLoss = 0;
  let stallHi = 0;

  for (const seg of segments) {
    const post = posts[seg.id];
    const T = seg.dt / MIN;
    const rIdeal = 60_000 / seg.idealCycleMs;
    const rSlow = rIdeal * SLOW_RATIO;
    const ideal = T * rIdeal;
    idealParts += ideal;

    if (seg.kind !== 'normal' || seg.delta === null) {
      const est = post.minutes.run * rIdeal + post.minutes.slow * rSlow;
      blindParts += est;
      const coLoss = Math.max(0, ideal - est);
      changeover += coLoss;
      blindLo += Math.max(0, ideal - post.minutesBand.run[1] * rIdeal - post.minutesBand.slow[1] * rSlow);
      blindHi += coLoss;
      unattributedMin += post.minutes.down;
      continue;
    }

    const counted = seg.delta as number;
    countedParts += counted;
    // signed: a segment can beat its standard, and that surplus must offset loss
    // elsewhere or the buckets will not sum to the counted-versus-standard gap
    const loss = ideal - counted;
    normalLoss += loss;

    // A long zero-gain stretch is not evidence of a stop; the counter could be
    // frozen. Keep it out of the stop buckets entirely.
    if (seg.stall) {
      stallLoss += Math.max(0, loss);
      stallHi += Math.max(0, loss);
      unattributedMin += post.minutes.down;
      continue;
    }

    const segLoss = Math.max(0, loss);
    let dLoss = post.minutes.down * rIdeal;
    let sLoss = post.minutes.slow * (rIdeal - rSlow);
    if (dLoss + sLoss > segLoss) {
      const k = segLoss / Math.max(1e-9, dLoss + sLoss);
      dLoss *= k;
      sLoss *= k;
    }
    downVar += Math.pow(((post.minutesBand.down[1] - post.minutesBand.down[0]) / 2) * rIdeal, 2);
    slowVar += Math.pow(((post.minutesBand.slow[1] - post.minutesBand.slow[0]) / 2) * (rIdeal - rSlow), 2);

    const overlap = overlapWithManual(seg, dataset);
    const pinned = pinnedDownMinutes(seg, constraints);
    const support = Math.min(post.minutes.down, Math.max(overlap, pinned));
    const rec = post.minutes.down > 0 ? (support / post.minutes.down) * dLoss : 0;
    recordedDown += rec;
    unrecordedDown += Math.max(0, dLoss - rec);
    slowLoss += sLoss;

    if (overlap <= 0 && pinned <= 0) unattributedMin += post.minutes.down;
  }

  // Closure: the headline loss is exact arithmetic (standard minus counted
  // minus blind). The buckets are apportionments of it, so whatever the state
  // model cannot claim becomes the residual. If the line out-produced its
  // standard, the residual goes negative in the arithmetic and is floored to
  // zero for display; the buckets then sum to the floored headline.
  const stallPart = Math.max(0, stallLoss);
  const totalLoss = Math.max(0, idealParts - countedParts - blindParts);
  const explained = recordedDown + unrecordedDown + slowLoss + changeover + stallPart;
  if (explained > totalLoss && explained > 1e-6) {
    const k = totalLoss / explained;
    recordedDown *= k;
    unrecordedDown *= k;
    slowLoss *= k;
    changeover *= k;
    stallLoss *= k;
  }
  const resid = totalLoss - (recordedDown + unrecordedDown + slowLoss + changeover + stallLoss);
  const downHalf = Math.sqrt(downVar);
  const slowHalf = Math.sqrt(slowVar);

  const buckets: LossBucket[] = [
    {
      key: 'down-recorded',
      label: '停机 · 有记录或已确认',
      parts: recordedDown,
      lo: Math.max(0, recordedDown - downHalf),
      hi: recordedDown + downHalf,
      unattributed: recordedDown > 0 ? 0.12 : 0,
      detail: '人工停机记录或已确认区间支持；边界仍有分钟级不确定。',
    },
    {
      key: 'down-unrecorded',
      label: '停机 · 无人工记录',
      parts: unrecordedDown,
      lo: 0,
      hi: unrecordedDown + downHalf,
      unattributed: 1,
      detail: '计数显示有停顿，但人工记录没有写到；可能是漏报，也可能是换型准备。',
    },
    {
      key: 'slow',
      label: '低速损失',
      parts: slowLoss,
      lo: Math.max(0, slowLoss - slowHalf),
      hi: slowLoss + slowHalf,
      unattributed: 0.6,
      detail: '低速与短暂停顿在区间累计量上难以区分，这部分主要是范围。',
    },
    {
      key: 'changeover',
      label: '换型 / 回零窗口',
      parts: changeover,
      lo: Math.max(0, Math.min(blindLo, changeover)),
      hi: Math.max(changeover, blindHi),
      unattributed: 0.55,
      detail: '计数清零，产量不可知；只有人工记录与先验能给出时间范围。',
    },
    {
      key: 'stall',
      label: '计数停滞 · 无法归因',
      parts: stallLoss,
      lo: 0,
      hi: Math.max(stallLoss, stallHi),
      unattributed: 1,
      detail:
        '计数完全不动且无人工记录：可能是真停机，也可能是计数器冻结。模型不替车间把两者混为一谈。',
    },
  ];

  const lossParts = Math.max(0, idealParts - countedParts - blindParts);
  const finalBuckets: LossBucket[] = [
    ...buckets,
    {
      key: 'residual',
      label: '未解释差额',
      parts: Math.max(0, resid),
      lo: 0,
      hi: Math.max(0, resid),
      unattributed: 1,
      detail:
        '理想产量减计数产量，再减去模型可解释的停机、低速与换型窗口。可能来自漏计、废品、计数误差，或没有记录的短停。',
    },
  ];

  // The total loss against the counter is exact arithmetic, but it is only as
  // good as the counter. Widen the range by (a) the buckets whose attribution is
  // still open, in quadrature, and (b) a stated counting-fidelity band on the
  // counted output. A range that collapses to a point would claim a precision
  // the raw reads do not have.
  const open = finalBuckets.filter((b) => b.unattributed >= 0.5);
  const bucketVar = open.reduce((s, b) => s + Math.pow(Math.max(0, b.hi - b.lo) / 2, 2), 0);
  const countVar = Math.pow(0.015 * countedParts, 2);
  const spread = Math.sqrt(bucketVar + countVar);
  const lossLo = Math.max(0, lossParts - spread);
  const lossHi = lossParts + spread;
  const partsPerMin = 60_000 / avgIdealCycle(segments);

  return {
    buckets: finalBuckets,
    idealParts,
    countedParts,
    blindParts,
    lossParts,
    lossLo,
    lossHi,
    lossMin: lossParts / partsPerMin,
    unattributedMin,
    unattributedParts: buckets.reduce((s, b) => s + b.parts * b.unattributed, 0),
  };
}

function avgIdealCycle(segments: Segment[]): number {
  let num = 0;
  let den = 0;
  for (const s of segments) {
    num += s.dt * s.idealCycleMs;
    den += s.dt;
  }
  return den > 0 ? num / den : 60_000;
}

// ---------------------------------------------------------------------------
// Deterministic baseline (the method this replaces)
// ---------------------------------------------------------------------------

export function baseline(dataset: Dataset): BaselineAnalysis {
  const warnings: Warning[] = [];
  const segments = buildSegments(dataset, warnings);
  const blindSpots: { seg: number; reason: string }[] = [];
  let idealParts = 0;
  let countedParts = 0;
  let lossParts = 0;
  let lossMin = 0;
  let blindIdeal = 0;
  for (const seg of segments) {
    const T = seg.dt / MIN;
    const rIdeal = 60_000 / seg.idealCycleMs;
    idealParts += T * rIdeal;
    if (seg.delta === null) {
      blindIdeal += T * rIdeal;
      blindSpots.push({
        seg: seg.id,
        reason: seg.kind === 'product' ? '换型清零：增量未知，基线无法分摊' : '计数回零：增量未知',
      });
      continue;
    }
    countedParts += seg.delta;
    const rate = seg.delta / T;
    if (rate < rIdeal * 0.98) {
      lossParts += T * rIdeal - seg.delta;
      lossMin += T * (1 - rate / rIdeal);
    }
  }
  return {
    lossParts,
    lossLo: lossParts * 0.8,
    lossHi: lossParts * 1.2,
    lossMin,
    idealParts,
    countedParts,
    blindParts: 0,
    blindSpots,
    note: `确定性基线把每个增量均匀摊到间隔上，不推断隐状态。${blindSpots.length} 段因清零无法估计，未计入损失，涉及标准产量约 ${Math.round(blindIdeal)} 件。`,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function describeCluster(downSpans: { t0: number; t1: number }[], idx: number): string {
  if (idx === 0) return '最可信路径';
  // say where it differs most from the best path: a shifted start
  if (!downSpans.length) return '无停机路径';
  return `另一条等价路径 · ${downSpans.length} 段停机`;
}

function logGamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function fmtClock(t: number): string {
  const d = new Date(t);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

export { PATH_SAMPLES, mulberry32 };
