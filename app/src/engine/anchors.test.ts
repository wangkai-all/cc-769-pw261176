import { describe, expect, it } from 'vitest';
import { analyze } from './infer';
import { SAMPLES, compareSample, sampleById } from '../samples';
import type { Constraint } from './types';

const MIN = 60_000;
const D = (h: number, m: number) => Date.UTC(2026, 10, 18, h, m);

/** A partition check that also holds while the model is being bent by
 *  operator input: an interval can never end before it starts, and each
 *  interval must begin exactly where the previous one ended. */
function assertPartition(a: ReturnType<typeof analyze>) {
  for (let i = 0; i < a.intervals.length; i++) {
    const iv = a.intervals[i];
    expect(iv.startLo).toBeLessThanOrEqual(iv.startHi);
    expect(iv.startHi).toBeLessThanOrEqual(iv.endLo);
    expect(iv.endLo).toBeLessThanOrEqual(iv.endHi);
    if (i > 0) {
      expect(iv.startLo).toBe(a.intervals[i - 1].endLo);
      expect(iv.startHi).toBe(a.intervals[i - 1].endHi);
    }
  }
}

describe('constraint anchoring', () => {
  it('pins both edges on every sample', () => {
    for (const s of SAMPLES) {
      const base = analyze(s.dataset);
      const target = base.intervals.find((i) => i.state !== 'run');
      if (!target) continue;
      const c: Constraint = { id: 'pin', kind: 'downtime', start: target.startHi, end: target.endLo, label: '确认' };
      const a = analyze(s.dataset, { constraints: [c] });
      const pinned = a.intervals.find((i) => i.constrained);
      expect(pinned).toBeDefined();
      expect(pinned!.startLo).toBe(c.start);
      expect(pinned!.endLo).toBe(c.end);
      expect(pinned!.endHi).toBe(c.end);
      assertPartition(a);
    }
  });

  it('can confirm a stretch the model first called slow', () => {
    const s = sampleById('hotsplit');
    const base = analyze(s.dataset);
    const slow = base.intervals.find((i) => i.state === 'slow')!;
    expect(slow).toBeDefined();
    const a = analyze(s.dataset, {
      constraints: [{ id: 'k', kind: 'downtime', start: slow.startHi, end: slow.endLo, label: '其实是停机' }],
    });
    const pinned = a.intervals.find((i) => i.constrained)!;
    expect(pinned.state).toBe('down');
    assertPartition(a);
  });

  it('stays a partition even with two confirmations whose edges cross', () => {
    const s = sampleById('steady');
    const a = analyze(s.dataset, {
      constraints: [
        { id: 'a', kind: 'downtime', start: D(7, 40), end: D(7, 50), label: 'A' },
        { id: 'b', kind: 'downtime', start: D(10, 30), end: D(10, 40), label: 'B' },
      ],
    });
    assertPartition(a);
    expect(a.intervals.filter((i) => i.constrained).length).toBeGreaterThanOrEqual(2);
  });

  it('reports rather than rewrites a confirmation that lands on a wall of production', () => {
    const s = sampleById('steady');
    const a = analyze(s.dataset, {
      constraints: [{ id: 'bad', kind: 'downtime', start: D(8, 30), end: D(9, 30), label: '错误确认' }],
    });
    expect(a.warnings.some((w) => w.code === 'constraint-conflict')).toBe(true);
    assertPartition(a);
  });
});

describe('slow spells are not absorbed into downtime', () => {
  it('finds a slow interval where the line was planted slow', () => {
    for (const id of ['steady', 'hotsplit', 'sparse-manual']) {
      const s = sampleById(id);
      const a = analyze(s.dataset);
      const slowMin = a.intervals.filter((i) => i.state === 'slow').reduce((x, i) => x + i.expectedMin, 0);
      expect(slowMin).toBeGreaterThan(10);
    }
  });

  it('keeps the slow interval adjacent to a stop on the ambiguous sample', () => {
    const a = analyze(sampleById('hotsplit').dataset);
    const slow = a.intervals.find((i) => i.state === 'slow');
    expect(slow).toBeDefined();
    // 45 minutes slow vs 30 minutes stopped produce nearly the same count
    const alt = slow!.alternatives.find((x) => x.state === 'down');
    expect(alt).toBeDefined();
    expect(alt!.share).toBeGreaterThan(0.2);
  });
});

describe('bands combine honestly across intervals', () => {  it('the day range does not grow with the number of report intervals', () => {
    const a = analyze(sampleById('steady').dataset);
    const half = (a.lossHi - a.lossLo) / 2;
    // summing per-segment widths would put this near the whole loss
    expect(half).toBeLessThan(0.5 * a.lossParts);
    expect(half).toBeGreaterThan(0.02 * a.lossParts);
  });

  it('every interval is either stable, or says it is a range', () => {
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      for (const iv of a.intervals) {
        const wide = iv.startHi - iv.startLo > 2 * MIN || iv.endHi - iv.endLo > 2 * MIN;
        if (wide) expect(iv.ambiguous).toBe(true);
      }
    }
  });

  it('draws every per-segment split as a real partition of that segment', () => {
    // the drawer renders these as widths; a split that sums past the segment
    // would overflow its track, one that sums short would leave a gap
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      for (const e of a.evidence) {
        for (const h of e.splits) {
          const sum = h.minutes.run + h.minutes.slow + h.minutes.down;
          expect(Math.abs(sum - e.dtMin)).toBeLessThan(1e-6 * Math.max(1, e.dtMin));
        }
      }
    }
  });

  it('never reports a negative duration', () => {
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      for (const e of a.evidence) {
        for (const st of ['run', 'slow', 'down'] as const) {
          expect(e.minutes[st]).toBeGreaterThanOrEqual(0);
          expect(e.minutesBand[st][0]).toBeGreaterThanOrEqual(0);
          expect(e.minutesBand[st][1]).toBeGreaterThanOrEqual(e.minutesBand[st][0]);
        }
      }
    }
  });

  it('names a competing reading at most once per state', () => {
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      for (const iv of a.intervals) {
        const states = iv.alternatives.map((x) => x.state);
        expect(new Set(states).size).toBe(states.length);
        for (const x of iv.alternatives) expect(x.state).not.toBe(iv.state);
      }
    }
  });
});

describe('loss buckets after the prior change', () => {
  it('still closes exactly and reports a non-degenerate range', () => {
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      const sum = a.buckets.reduce((x, b) => x + b.parts, 0);
      expect(Math.abs(sum - a.lossParts)).toBeLessThan(1e-6);
      expect(a.lossHi).toBeGreaterThan(a.lossLo);
    }
  });

  it('keeps the model within the same order as the planted loss', () => {
    for (const s of SAMPLES) {
      const c = compareSample(s);
      const rel = Math.abs(c.modelErr.lossParts) / Math.max(1, c.truth.lossParts);
      expect(rel).toBeLessThan(0.7);
    }
  });

  it('the frozen-counter window is not billed as proven downtime', () => {
    const a = analyze(sampleById('noisy').dataset);
    const stall = a.buckets.find((b) => b.key === 'stall')!;
    const recorded = a.buckets.find((b) => b.key === 'down-recorded')!;
    expect(stall.parts).toBeGreaterThan(0);
    expect(recorded.parts).toBeLessThan(0.5 * a.lossParts);
  });
});
