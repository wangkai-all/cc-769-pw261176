import { describe, expect, it } from 'vitest';
import { analyze, baseline, buildSegments } from './infer';
import { SAMPLES, compareSample, sampleById } from '../samples';
import type { Constraint } from './types';

const MIN = 60_000;

describe('determinism', () => {
  it('same dataset yields the same analysis twice', () => {
    const s = sampleById('missing');
    const a = analyze(s.dataset);
    const b = analyze(s.dataset);
    expect(b.intervals.map((i) => [i.id, i.state, i.startLo, i.endLo])).toEqual(
      a.intervals.map((i) => [i.id, i.state, i.startLo, i.endLo]),
    );
    expect(b.lossParts).toBe(a.lossParts);
    expect(b.hypotheses.map((h) => h.weight)).toEqual(a.hypotheses.map((h) => h.weight));
  });

  it('constraint order does not change the result', () => {
    const s = sampleById('steady');
    const c1: Constraint = { id: 'c1', kind: 'downtime', start: Date.UTC(2026, 10, 18, 7, 38), end: Date.UTC(2026, 10, 18, 8, 0), label: 'A' };
    const c2: Constraint = { id: 'c2', kind: 'downtime', start: Date.UTC(2026, 10, 18, 10, 48), end: Date.UTC(2026, 10, 18, 10, 58), label: 'B' };
    const a = analyze(s.dataset, { constraints: [c1, c2] });
    const b = analyze(s.dataset, { constraints: [c2, c1] });
    expect(b.lossParts).toBe(a.lossParts);
  });
});

describe('segments and counter discontinuities', () => {
  it('a reset is not differenced across', () => {
    const s = sampleById('reset');
    const segs = buildSegments(s.dataset);
    const resets = segs.filter((g) => g.kind === 'reset');
    expect(resets.length).toBeGreaterThanOrEqual(2);
    for (const r of resets) expect(r.delta).toBeNull();
  });

  it('a product change segment is blind and carries no delta', () => {
    const s = sampleById('changeover');
    const analysis = analyze(s.dataset);
    const crossing = analysis.segments.filter((g) => g.kind === 'product');
    expect(crossing.length).toBeGreaterThanOrEqual(1);
    for (const g of crossing) expect(g.delta).toBeNull();
    expect(analysis.blindParts).toBeGreaterThan(0);
  });

  it('every segment resolves to the product in force at its end', () => {
    const s = sampleById('changeover');
    const segs = buildSegments(s.dataset);
    for (const g of segs) {
      const expected = g.t1 >= s.dataset.productChanges[0].t ? s.dataset.productChanges[0].product : s.dataset.initial.product;
      expect(g.product).toBe(expected);
    }
  });
});

describe('no fabricated precision', () => {
  it('does not smear downtime into clean intervals', () => {
    const a = analyze(sampleById('steady').dataset);
    // the long clean run at the end must stay a single run interval
    const tail = a.intervals.filter((i) => i.endLo > Date.UTC(2026, 10, 18, 12, 0));
    expect(tail).toHaveLength(1);
    expect(tail[0].state).toBe('run');
    expect(tail[0].confidence).toBeGreaterThan(0.9);
  });

  it('intervals form a partition: no overlaps, no gaps', () => {
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      for (let i = 1; i < a.intervals.length; i++) {
        const prev = a.intervals[i - 1];
        const cur = a.intervals[i];
        expect(cur.startLo).toBe(prev.endLo);
        expect(cur.startHi).toBe(prev.endHi);
        expect(cur.startLo).toBeLessThanOrEqual(cur.startHi);
        expect(cur.endLo).toBeGreaterThanOrEqual(cur.startHi);
        expect(cur.endHi).toBeGreaterThanOrEqual(cur.endLo);
      }
    }
  });

  it('wide boundaries are flagged rather than given a timestamp', () => {
    const a = analyze(sampleById('missing').dataset);
    const ambiguous = a.intervals.filter((i) => i.ambiguous);
    expect(ambiguous.length).toBeGreaterThan(0);
    for (const i of ambiguous) expect(i.startLo).not.toBe(i.endHi - 1);
  });

  it('keeps several near-equivalent readings on the ambiguous sample', () => {
    const a = analyze(sampleById('hotsplit').dataset);
    const withAlternative = a.intervals.filter((i) => i.alternatives.length > 0);
    expect(withAlternative.length).toBeGreaterThan(0);
    const top = withAlternative[0].alternatives[0];
    expect(top.share).toBeGreaterThan(0.15);
    expect(['run', 'slow', 'down']).toContain(top.state);
  });
});

describe('loss accounting', () => {
  it('decomposition sums to the counted-versus-standard loss', () => {
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      const sum = a.buckets.reduce((x, b) => x + b.parts, 0);
      const target = Math.max(0, a.idealParts - a.countedParts - a.blindParts);
      expect(Math.abs(sum - target)).toBeLessThan(Math.max(2, target * 0.001));
    }
  });

  it('reports a range, not a point, for every sample', () => {
    for (const s of SAMPLES) {
      const a = analyze(s.dataset);
      expect(a.lossHi).toBeGreaterThan(a.lossLo);
      expect(a.lossLo).toBeLessThanOrEqual(a.lossParts);
      expect(a.lossHi).toBeGreaterThanOrEqual(a.lossParts);
    }
  });

  it('names the unattributable remainder instead of hiding it', () => {
    const a = analyze(sampleById('sparse-manual').dataset);
    expect(a.unattributedShare).toBeGreaterThan(0.2);
    const stall = a.buckets.find((b) => b.key === 'stall');
    expect(stall).toBeDefined();
    expect(stall!.unattributed).toBe(1);
  });

  it('does not book a fully frozen counter as proven downtime', () => {
    const a = analyze(sampleById('noisy').dataset);
    const stall = a.buckets.find((b) => b.key === 'stall')!;
    expect(stall.parts).toBeGreaterThan(0);
    const residual = a.buckets.find((b) => b.key === 'residual')!;
    expect(residual.unattributed).toBe(1);
  });
});

describe('constraints and conflicts', () => {
  const steadyDown = () => {
    const s = sampleById('steady');
    return {
      s,
      start: Date.UTC(2026, 10, 18, 7, 40),
      end: Date.UTC(2026, 10, 18, 8, 2),
    };
  };

  it('a confirmed stop pins the boundary and clears ambiguity', () => {
    const { s, start, end } = steadyDown();
    const c: Constraint = { id: 'c', kind: 'downtime', start, end, label: '确认停机' };
    const a = analyze(s.dataset, { constraints: [c] });
    const hit = a.intervals.find((i) => i.state === 'down' && i.startLo < end && i.endHi > start);
    expect(hit).toBeDefined();
    expect(hit!.constrained).toBe(true);
    expect(hit!.startLo).toBe(start);
    expect(hit!.startHi).toBe(start);
    expect(hit!.endLo).toBe(end);
    expect(hit!.endHi).toBe(end);
    expect(hit!.ambiguous).toBe(false);
  });

  it('a confirmation is a constraint on the recomputation, not a decoration', () => {
    const { s, start, end } = steadyDown();
    const before = analyze(s.dataset);
    const after = analyze(s.dataset, { constraints: [{ id: 'c', kind: 'downtime', start, end, label: '确认停机' }] });
    const overlapping = (e: { t0: number; t1: number }) => e.t1 > start && e.t0 < end;
    const downBefore = before.evidence.filter(overlapping).reduce((x, e) => x + e.minutes.down, 0);
    const downAfter = after.evidence.filter(overlapping).reduce((x, e) => x + e.minutes.down, 0);
    // the window is pinned to 22 confirmed minutes; the model cannot leave 10
    expect(downAfter).toBeGreaterThanOrEqual(21.5);
    expect(downAfter).toBeGreaterThan(downBefore);
    expect(after.constrained).toBe(true);
  });

  it('flags a conflict and names the count segments involved', () => {
    const s = sampleById('steady');
    // claim a 60-minute stop across a stretch that visibly produced parts
    const bad: Constraint = {
      id: 'bad',
      kind: 'downtime',
      start: Date.UTC(2026, 10, 18, 8, 30),
      end: Date.UTC(2026, 10, 18, 9, 30),
      label: '错误确认',
    };
    const a = analyze(s.dataset, { constraints: [bad] });
    const warn = a.warnings.find((w) => w.code === 'constraint-conflict');
    expect(warn).toBeDefined();
    expect(warn!.segs!.length).toBeGreaterThan(0);
    expect(warn!.text).toContain('段 #');
  });

  it('ignores out-of-range or inverted constraints and says so', () => {
    const s = sampleById('steady');
    const a = analyze(s.dataset, {
      constraints: [{ id: 'x', kind: 'downtime', start: Date.UTC(2030, 0, 1), end: Date.UTC(2030, 0, 2), label: '未来' }],
    });
    expect(a.warnings.some((w) => w.code === 'bad-constraint')).toBe(true);
    expect(a.intervals.every((i) => !i.constrained)).toBe(true);
  });
});

describe('product standard change', () => {
  it('does not reuse the old era emission parameters', () => {
    const a = analyze(sampleById('changeover').dataset);
    expect(a.eras).toHaveLength(2);
    expect(a.eras[0].idealCycleMs).not.toBe(a.eras[1].idealCycleMs);
    expect(a.priorNote).toContain('2 套');
    // same minute count, different standard -> different ideal output per era
    const perMin = a.eras.map((e) => e.idealParts / ((e.to - e.from) / MIN));
    expect(perMin[0]).toBeLessThan(perMin[1]);
  });

  it('gives duplicate products different eras when the standard changes', () => {
    const s = sampleById('changeover');
    const ds = {
      ...s.dataset,
      productChanges: [{ ...s.dataset.productChanges[0], product: s.dataset.initial.product, idealCycleMs: 900 }],
    };
    const a = analyze(ds);
    expect(a.eras).toHaveLength(2);
    expect(a.eras[1].idealCycleMs).toBe(900);
  });
});

describe('model versus the deterministic baseline', () => {
  it('scores every sample against its planted truth', () => {
    for (const s of SAMPLES) {
      const c = compareSample(s);
      expect(Number.isFinite(c.model.lossParts)).toBe(true);
      expect(Number.isFinite(c.base.lossParts)).toBe(true);
      expect(c.truth.parts).toBeGreaterThan(0);
    }
  });

  it('sees downtime the baseline averages away on the missing-report day', () => {
    const c = compareSample(sampleById('missing'));
    const down = c.model.intervals.filter((i) => i.state === 'down').reduce((s, i) => s + i.expectedMin, 0);
    expect(down).toBeGreaterThan(0.5 * c.truth.minutes.down);
    expect(Math.abs(c.modelErr.lossParts)).toBeLessThan(Math.abs(c.baselineErr.lossParts) + 400);
  });

  it('baseline cannot attribute blind segments and says so', () => {
    const b = baseline(sampleById('reset').dataset);
    expect(b.blindSpots.length).toBeGreaterThan(0);
    expect(b.note).toContain('无法估计');
  });
});

describe('warnings surface the damage', () => {
  it('missing reports are called out', () => {
    const a = analyze(sampleById('missing').dataset);
    expect(a.warnings.some((w) => w.code === 'long-gap')).toBe(true);
  });

  it('resets and changeovers are called out as blind', () => {
    const a = analyze(sampleById('changeover').dataset);
    expect(a.warnings.some((w) => w.code === 'discontinuity')).toBe(true);
  });

  it('a counter that undercounts is not reported as all-day downtime', () => {
    const a = analyze(sampleById('noisy').dataset);
    const downShare = a.intervals.filter((i) => i.state === 'down').reduce((s, i) => s + i.expectedMin, 0) / a.horizonMin;
    expect(downShare).toBeLessThan(0.3);
    expect(a.warnings.some((w) => w.code === 'count-bias')).toBe(true);
  });
});
