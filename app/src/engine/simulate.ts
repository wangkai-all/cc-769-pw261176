import type { CountReport, Dataset, ProductChange, StatedDowntime, StateName, SyntheticTruth } from './types';
import { logNormal, mulberry32 } from './prng';

interface TruthSpan {
  state: StateName;
  t0: number;
  t1: number;
}

export interface ScenarioInput {
  seed: number;
  /** absolute start, epoch ms */
  start: number;
  /** total horizon, minutes */
  minutes: number;
  /** typical report cadence, minutes */
  reportEvery: number;
  initial: { product: string; idealCycleMs: number };
  /** product changes: minute offsets into the horizon */
  changes: { atMin: number; product: string; idealCycleMs: number }[];
  /** true state schedule, laid end to end then stretched to the horizon */
  schedule: [StateName, number][];
  /** extra counter zeroings at these minute offsets (shift rollover, PLC restart) */
  extraResetsAtMin?: number[];
  /** stuck-counter window [fromMin, toMin]: the read repeats its last value */
  frozenCounterAtMin?: [number, number];
  /** indexes of true down spans whose manual record is missing */
  dropDowntime?: number[];
  /** a manual record for a stretch that was in fact running slow */
  spuriousDowntime?: { atMin: number; minutes: number };
  /** fraction of produced parts the counter fails to register (undercount) */
  undercountP?: number;
  /** report times to drop entirely (missing flush) */
  dropReportsAtMin?: number[];
  /** drop every report inside this minute window: a real reporting outage */
  dropReportsRange?: [number, number];
  /** multiply the standard rate to get the planted actual run rate */
  runRateScale?: number;
  /** multiply the standard rate to get the planted actual slow rate */
  slowRateScale?: number;
}

const MIN = 60_000;

/** Ideal cycle for a product in the era list (initial is index 0). */
function cycleFor(input: ScenarioInput, era: number): number {
  if (era === 0) return input.initial.idealCycleMs;
  const c = input.changes[era - 1];
  return c ? c.idealCycleMs : input.initial.idealCycleMs;
}

/**
 * Simulate a line and record only what the plant actually keeps: a cumulative
 * counter read irregularly, sparse manual downtime notes, and product-change
 * bookkeeping. The hidden state path is returned separately as ground truth and
 * is never handed to the inference engine.
 */
export function simulateScenario(input: ScenarioInput): Dataset {
  const rng = mulberry32(input.seed);
  const t0 = input.start;
  const tEnd = t0 + input.minutes * MIN;

  // ---- hidden state path --------------------------------------------------
  const spans: TruthSpan[] = [];
  let cur = t0;
  const specMin = input.schedule.reduce((s, [, d]) => s + d, 0);
  const stretch = input.minutes / specMin;
  for (const [state, durMin] of input.schedule) {
    const dt = durMin * MIN * stretch;
    if (dt <= 0) continue;
    spans.push({ state, t0: Math.round(cur), t1: Math.round(cur + dt) });
    cur += dt;
  }
  if (spans.length) spans[spans.length - 1].t1 = tEnd;

  // ---- product eras -------------------------------------------------------
  const changes = [...input.changes].sort((a, b) => a.atMin - b.atMin);
  const changeTimes = changes.map((c) => t0 + c.atMin * MIN);
  const eraStarts = [t0, ...changeTimes];
  const eraEnds = [...changeTimes, tEnd];
  const eraAt = (t: number): number => {
    let idx = 0;
    for (let i = 0; i < changeTimes.length; i++) if (t >= changeTimes[i]) idx = i + 1;
    return idx;
  };
  const productOf = (era: number) => (era === 0 ? input.initial.product : changes[era - 1]?.product ?? input.initial.product);

  // ---- lattice production -------------------------------------------------
  const STEP = 10_000;
  const pts: { t: number; parts: number; counted: number; era: number }[] = [];
  const trueMinutes: Record<StateName, number> = { run: 0, slow: 0, down: 0 };
  const eraTrueParts = new Array(eraStarts.length).fill(0) as number[];
  const runScale = input.runRateScale ?? 1;
  const slowScale = input.slowRateScale ?? 1;
  const undercountP = input.undercountP ?? 0;

  for (let t = t0; t < tEnd; t += STEP) {
    const span = spans.find((s) => t >= s.t0 && t < s.t1) ?? spans[spans.length - 1];
    const state: StateName = span ? span.state : 'down';
    const era = eraAt(t);
    const ideal = cycleFor(input, era);
    trueMinutes[state] += STEP / MIN;
    let parts = 0;
    if (state === 'run') {
      const cycle = logNormal(rng, ideal / runScale, 0.06);
      parts = Math.max(0, Math.round((STEP / cycle) * (1 + (rng() - 0.5) * 0.06)));
    } else if (state === 'slow') {
      const cycle = logNormal(rng, (ideal * 2.05) / slowScale, 0.18);
      // stop-start micro-jams: a fifth of the ticks produce nothing
      parts = rng() < 0.22 ? 0 : Math.max(0, Math.round(STEP / cycle));
    }
    let counted = parts;
    if (undercountP > 0) {
      counted = 0;
      for (let k = 0; k < parts; k++) if (rng() >= undercountP) counted++;
    }
    eraTrueParts[era] += parts;
    pts.push({ t, parts, counted, era });
  }

  // ---- counter epochs -----------------------------------------------------
  const resetTimes = [...changeTimes, ...(input.extraResetsAtMin ?? []).map((m) => t0 + m * MIN)].sort(
    (a, b) => a - b,
  );
  const epochAt = (t: number) => {
    let n = 0;
    for (const rt of resetTimes) if (t >= rt) n++;
    return n;
  };
  const freeze = input.frozenCounterAtMin
    ? { a: t0 + input.frozenCounterAtMin[0] * MIN, b: t0 + input.frozenCounterAtMin[1] * MIN }
    : undefined;

  // cumulative counter at time t; a frozen read holds its value until the
  // freeze ends, then snaps back to the true cumulative
  const cumAt = (t: number): number => {
    const e = epochAt(t);
    let total = 0;
    let frozenValue: number | null = null;
    for (const p of pts) {
      if (p.t >= t) break;
      if (epochAt(p.t) !== e) continue;
      if (freeze && p.t >= freeze.a && p.t < freeze.b) {
        if (frozenValue === null) frozenValue = total;
        continue;
      }
      total += p.counted;
    }
    if (freeze && t >= freeze.a && t < freeze.b) return frozenValue ?? total;
    return total;
  };

  // ---- report schedule ----------------------------------------------------
  const reports: CountReport[] = [];
  const droppedTimes = new Set((input.dropReportsAtMin ?? []).map((m) => Math.round(t0 + m * MIN)));
  const dropRange = input.dropReportsRange
    ? [t0 + input.dropReportsRange[0] * MIN, t0 + input.dropReportsRange[1] * MIN]
    : null;
  const push = (tt: number, cum: number) => {
    const at = Math.round(tt);
    if (droppedTimes.has(at)) return;
    if (dropRange && at > dropRange[0] && at < dropRange[1]) return;
    const prev = reports[reports.length - 1];
    if (prev && at - prev.t < 45_000) return;
    if (at > tEnd) return;
    reports.push({ t: at, cum });
  };
  push(t0, 0);
  let t = t0 + 3 * MIN;
  let lastEpoch = 0;
  while (t < tEnd) {
    const e = epochAt(t);
    if (e !== lastEpoch) {
      // the counter reloads at zero; a read lands shortly after each reset
      lastEpoch = e;
      const rt = resetTimes.filter((r) => r <= t).pop() ?? t;
      push(rt + 50_000 + rng() * 40_000, 0);
    }
    push(t, cumAt(t));
    const outage = rng() < 0.07 ? 3 + rng() * 4 : 1;
    t += input.reportEvery * MIN * outage * (0.55 + rng() * 1.0);
  }
  const last = reports[reports.length - 1];
  if (last.t < tEnd - 30_000) push(tEnd - 20_000, cumAt(tEnd - 20_000));

  // ---- manual downtime records -------------------------------------------
  const downtime: StatedDowntime[] = [];
  const downSpans = spans.filter((s) => s.state === 'down');
  const droppedIds = new Set(input.dropDowntime ?? []);
  downSpans.forEach((s, i) => {
    if (droppedIds.has(i)) return;
    const dur = s.t1 - s.t0;
    if (dur < 90_000) return; // not worth a note
    const chunks = dur > 40 * MIN ? Math.max(1, Math.round(dur / (75 * MIN))) : 1;
    for (let k = 0; k < chunks; k++) {
      const a = s.t0 + (dur * k) / chunks;
      const b = s.t0 + (dur * (k + 1)) / chunks;
      // operators round to the minute and are a few minutes early/late
      const jitterA = (rng() - 0.5) * 6 * MIN;
      const jitterB = (rng() - 0.5) * 6 * MIN;
      downtime.push({
        id: `dt-${i}-${k}`,
        start: Math.round((a + jitterA) / MIN) * MIN,
        end: Math.round((b + jitterB) / MIN) * MIN,
        note: ['液压站压力低', '缺料，等配送', '换型夹具卡滞', '下游堵料', '刀具更换'][Math.floor(rng() * 5)],
        source: 'manual',
      });
    }
  });
  if (input.spuriousDowntime) {
    const a = t0 + input.spuriousDowntime.atMin * MIN;
    downtime.push({
      id: 'dt-spurious',
      start: a,
      end: a + input.spuriousDowntime.minutes * MIN,
      note: '操作员记为停机，实际低速运行',
      source: 'manual',
    });
  }
  downtime.sort((a, b) => a.start - b.start);

  // ---- truth summary ------------------------------------------------------
  const totalParts = pts.reduce((s, p) => s + p.parts, 0);
  const idealParts = Math.round(
    eraStarts.reduce((acc, s, i) => acc + (((eraEnds[i] - s) / MIN) * 60_000) / cycleFor(input, i), 0),
  );
  const truth: SyntheticTruth = {
    states: spans.map((s) => ({ state: s.state, t0: s.t0, t1: s.t1 })),
    minutes: { run: r1(trueMinutes.run), slow: r1(trueMinutes.slow), down: r1(trueMinutes.down) },
    parts: totalParts,
    idealParts,
    lossParts: idealParts - totalParts,
    eras: eraStarts.map((s, i) => ({
      product: productOf(i),
      t0: s,
      t1: eraEnds[i],
      idealCycleMs: cycleFor(input, i),
      parts: eraTrueParts[i],
      idealParts: Math.round((((eraEnds[i] - s) / MIN) * 60_000) / cycleFor(input, i)),
    })),
    injects: buildInjects(input),
  };

  const productChanges: ProductChange[] = changes.map((c) => ({
    t: t0 + c.atMin * MIN,
    product: c.product,
    idealCycleMs: c.idealCycleMs,
    note: '换型',
  }));

  return {
    id: `sim-${input.seed}`,
    name: `合成样例 · ${frTime(t0)}`,
    note: buildNote(input, downSpans.length),
    initial: input.initial,
    reports,
    productChanges,
    downtime,
    truth,
  };
}

const r1 = (x: number) => Math.round(x * 10) / 10;
const frTime = (t: number) => {
  const d = new Date(t);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};

function buildInjects(input: ScenarioInput): string[] {
  const out: string[] = [];
  if (input.extraResetsAtMin?.length) out.push(`计数回零 ×${input.extraResetsAtMin.length}`);
  if (input.frozenCounterAtMin) out.push('计数器冻结');
  if (input.dropReportsRange) out.push('上报中断');
  if (input.undercountP) out.push(`漏计 ${Math.round(input.undercountP * 100)}%`);
  if (input.dropDowntime?.length) out.push(`停机漏报 ×${input.dropDowntime.length}`);
  if (input.spuriousDowntime) out.push('停机误报');
  if (input.changes.length) out.push(`换产品 ×${input.changes.length}`);
  return out;
}

function buildNote(input: ScenarioInput, downSpanCount: number): string {
  const bits = [`${input.minutes} 分钟`, `上报≈${input.reportEvery} 分`, `停机 ${downSpanCount} 段`];
  if (input.changes.length) bits.push(`${input.changes.length} 次换型`);
  return bits.join(' · ');
}
