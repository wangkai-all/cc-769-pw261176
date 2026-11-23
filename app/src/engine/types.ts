/**
 * TaktLens domain model.
 * Timestamps are epoch ms. Durations are ms unless the name ends in `Min`.
 * Counters are whole parts.
 */

export type StateName = 'run' | 'slow' | 'down';
export const STATES: StateName[] = ['run', 'slow', 'down'];
export const STATE_LABEL: Record<StateName, string> = {
  run: '运行',
  slow: '低速',
  down: '停机',
};

export interface CountReport {
  t: number;
  cum: number;
}

export interface ProductChange {
  t: number;
  product: string;
  idealCycleMs: number;
  note?: string;
}

export interface ProductRef {
  product: string;
  idealCycleMs: number;
}

export interface StatedDowntime {
  id: string;
  start: number;
  end: number;
  note: string;
  /** manual logs are sparse and occasionally late - never treated as complete */
  source: 'manual';
}

export interface SyntheticTruth {
  states: { state: StateName; t0: number; t1: number }[];
  minutes: Record<StateName, number>;
  parts: number;
  idealParts: number;
  lossParts: number;
  eras: {
    product: string;
    t0: number;
    t1: number;
    idealCycleMs: number;
    parts: number;
    idealParts: number;
  }[];
  injects: string[];
}

export interface Dataset {
  id: string;
  name: string;
  note: string;
  /** the standard in force before the first product change */
  initial: ProductRef;
  reports: CountReport[];
  productChanges: ProductChange[];
  downtime: StatedDowntime[];
  /** only on synthetic samples, for the verification panel */
  truth?: SyntheticTruth;
}

export type SegmentKind = 'normal' | 'reset' | 'product';

/** A pair of consecutive count reports, resolved to one product era. */
export interface Segment {
  id: number;
  /** index of the report pair this segment came from */
  pair: number;
  era: number;
  product: string;
  idealCycleMs: number;
  t0: number;
  t1: number;
  dt: number;
  /**
   * parts gained over the segment; null when the pair straddles a product
   * change or a reset, so no single gain can be attributed to one rate
   */
  delta: number | null;
  kind: SegmentKind;
  /** interval too short for count quantisation to say much */
  lowInfo: boolean;
  /** zero gain over a long stretch with no note: a stop and a stalled counter
   *  are indistinguishable from the counts alone */
  stall: boolean;
  /** reasons this segment carries an unresolvable gain */
  notes: string[];
}

export interface RatePrior {
  /** parts per minute, median of a log-normal */
  median: number;
  /** sigma in log space */
  spread: number;
}

export interface CadencePrior {
  source: 'product-standard';
  /** the standard this prior was derived from */
  idealCycleMs: number;
  /** the product this prior was derived for; a change invalidates it */
  product: string;
  pRun: number;
  pSlow: number;
  pDown: number;
  run: RatePrior;
  slow: RatePrior;
  label: string;
}

/** One coherent hidden path over the horizon, kept when it explains the counts
 *  about as well as the best path. */
export interface Hypothesis {
  id: string;
  label: string;
  /** share of posterior mass landing on this cluster */
  weight: number;
  /** total minutes per state along this path */
  minutes: Record<StateName, number>;
  /** down spans as this path places them (only down; enough to compare) */
  down: { t0: number; t1: number }[];
  /** lower likelihood than the best path, in log units */
  logGap: number;
}

export interface InferredInterval {
  id: string;
  state: StateName;
  /** dominant = the model's best label for this stretch;
   *  suspect = plausibly this state but not the majority */
  kind: 'dominant' | 'suspect';
  /** competing readings of the same stretch that the counts cannot separate,
   *  e.g. 45 minutes of slow running vs. 30 minutes stopped. Empty when the
   *  counts pick one state clearly. */
  alternatives: { state: StateName; share: number }[];
  /** posterior mass of the dominant label across its cells, 0..1 */
  confidence: number;
  /** share of sampled paths that contain this stretch at all */
  pathShare: number;
  /** plausible range for the start of the stretch */
  startLo: number;
  startHi: number;
  /** plausible range for the end of the stretch */
  endLo: number;
  endHi: number;
  expectedMin: number;
  segs: number[];
  manualIds: string[];
  /** boundaries cannot be dated closer than this range */
  ambiguous: boolean;
  /** an operator constraint pinned this stretch */
  constrained: boolean;
}

export interface SegmentEvidence {
  seg: number;
  t0: number;
  t1: number;
  dtMin: number;
  delta: number | null;
  idealParts: number;
  /** parts loss vs. the era standard, expected and plausible range */
  lossParts: number;
  lossLo: number;
  lossHi: number;
  /** posterior state mass over the segment */
  p: Record<StateName, number>;
  minutes: Record<StateName, number>;
  minutesBand: Record<StateName, [number, number]>;
  /** the near-equivalent splits of this segment; empty when the counts pin one */
  alternatives: { state: StateName; share: number }[];
  /** every near-equivalent time split of this one segment, minutes within it */
  splits: Hypothesis[];
  /** the handful of near-equivalent paths that survive the counts */
  hypotheses: Hypothesis[];
  /** plain-language account of what the counts do and do not prove */
  notes: string[];
}

export interface LossBucket {
  key: string;
  label: string;
  /** loss parts, expected and plausible range */
  parts: number;
  lo: number;
  hi: number;
  /** fraction of the bucket whose attribution / timing is still uncertain, 0..1 */
  unattributed: number;
  detail: string;
}

export interface Constraint {
  id: string;
  kind: 'downtime';
  /** interval the operator asserted to be fully stopped */
  start: number;
  end: number;
  label: string;
}

export interface Warning {
  code: string;
  text: string;
  segs?: number[];
}

export interface EraSummary {
  era: number;
  product: string;
  idealCycleMs: number;
  from: number;
  to: number;
  idealParts: number;
  priorLabel: string;
}

export interface Analysis {
  segments: Segment[];
  eras: EraSummary[];
  intervals: InferredInterval[];
  evidence: SegmentEvidence[];
  buckets: LossBucket[];
  warnings: Warning[];
  /** the near-equivalent hidden paths, best first */
  hypotheses: Hypothesis[];
  /** sample spread of the loss estimate, parts */
  lossPosterior: { parts: number; lo: number; hi: number; samples: number[] };
  /** wall minutes covered by the count record */
  horizonMin: number;
  /** ideal output under each era's own standard */
  idealParts: number;
  /** counted parts: sum of the deltas that could be resolved */
  countedParts: number;
  /** parts in reset / changeover segments, estimated from the hidden state */
  blindParts: number;
  /** counted + blind */
  actualParts: number;
  lossParts: number;
  lossLo: number;
  lossHi: number;
  /** expected minutes of lost capacity, top-down from parts */
  lossMin: number;
  /** posterior down minutes with no manual record to back them */
  unattributedMin: number;
  /** unattributed parts as a share of total loss */
  unattributedShare: number;
  constrained: boolean;
  priorNote: string;
  modelVersion: string;
}

/** Deterministic baseline: no hidden state, spread counts evenly over each interval. */
export interface BaselineAnalysis {
  lossParts: number;
  lossLo: number;
  lossHi: number;
  lossMin: number;
  idealParts: number;
  countedParts: number;
  blindParts: number;
  blindSpots: { seg: number; reason: string }[];
  note: string;
}
