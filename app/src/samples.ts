import { simulateScenario, type ScenarioInput } from './engine/simulate';
import { analyze, baseline } from './engine/infer';
import type { Dataset } from './engine/types';

const START = Date.UTC(2026, 10, 18, 6, 0, 0); // 06:00 local-ish, fixed for reproducibility

export interface Sample {
  id: string;
  title: string;
  blurb: string;
  dataset: Dataset;
}

/**
 * Seven synthetic days. Every one has a known hidden state path; the counters
 * are generated from it and then damaged in the ways a real plant damages them.
 * The verification panel scores the model and the deterministic baseline
 * against that path.
 */
function make(id: string, title: string, blurb: string, input: ScenarioInput): Sample {
  return { id, title, blurb, dataset: simulateScenario({ ...input, start: START }) };
}

export const SAMPLES: Sample[] = [
  make('steady', '基准日：正常三班', '一个干净的 8 小时班次：运行、两次短停、一段慢速。用来校准读数。', {
    seed: 1001,
    start: START,
    minutes: 480,
    reportEvery: 8,
    initial: { product: 'A-220 壳体', idealCycleMs: 1420 },
    changes: [],
    schedule: [
      ['run', 100],
      ['down', 22],
      ['run', 120],
      ['slow', 45],
      ['down', 12],
      ['run', 181],
    ],
    undercountP: 0,
  }),

  make('missing', '缺报：长时间无读数', '排产系统有 6 小时没有上报，只有首尾两个累计数。停机被藏在一个大区间里。', {
    seed: 1002,
    start: START,
    minutes: 600,
    reportEvery: 9,
    initial: { product: 'A-220 壳体', idealCycleMs: 1420 },
    changes: [],
    schedule: [
      ['run', 140],
      ['down', 95],
      ['run', 160],
      ['slow', 60],
      ['down', 55],
      ['run', 90],
    ],
    dropReportsRange: [200, 370],
    undercountP: 0.01,
  }),

  make('reset', '回零：中班清零', '班次切换时计数器回零，之后又发生两次 PLC 重启。计数差不能直接相减。', {
    seed: 1003,
    start: START,
    minutes: 540,
    reportEvery: 7,
    initial: { product: 'B-17 端盖', idealCycleMs: 980 },
    changes: [],
    schedule: [
      ['run', 130],
      ['down', 18],
      ['run', 110],
      ['down', 35],
      ['slow', 50],
      ['down', 20],
      ['run', 177],
    ],
    extraResetsAtMin: [180, 395],
    undercountP: 0.005,
  }),

  make('changeover', '换产品：标准节拍改变', '中途从 1.42 s 换到 0.98 s 的产品。旧发射参数必须作废，换型窗口产量不可知。', {
    seed: 1004,
    start: START,
    minutes: 600,
    reportEvery: 8,
    initial: { product: 'A-220 壳体', idealCycleMs: 1420 },
    changes: [{ atMin: 260, product: 'C-6 支架', idealCycleMs: 980 }],
    schedule: [
      ['run', 150],
      ['down', 30],
      ['run', 70],
      ['down', 40],
      ['run', 130],
      ['slow', 60],
      ['down', 25],
      ['run', 95],
    ],
    undercountP: 0.008,
  }),

  make('noisy', '脏数据：漏计 + 冻结 + 误报', '计数器漏掉 4% 的件，有一小时读数冻结，操作员还把一段低速误记成停机。', {
    seed: 1005,
    start: START,
    minutes: 600,
    reportEvery: 6,
    initial: { product: 'D-3 齿轮', idealCycleMs: 1150 },
    changes: [],
    schedule: [
      ['run', 120],
      ['down', 25],
      ['slow', 70],
      ['down', 30],
      ['run', 210],
      ['down', 20],
      ['run', 125],
    ],
    frozenCounterAtMin: [310, 370],
    spuriousDowntime: { atMin: 145, minutes: 40 },
    undercountP: 0.04,
    dropDowntime: [1],
  }),

  make('sparse-manual', '人工记录稀疏', '大部分停机没有任何人工记录。只能靠计数差证明有停顿，但时间边界很宽。', {
    seed: 1006,
    start: START,
    minutes: 720,
    reportEvery: 14,
    initial: { product: 'E-9 法兰', idealCycleMs: 2600 },
    changes: [],
    schedule: [
      ['run', 200],
      ['down', 70],
      ['run', 120],
      ['slow', 90],
      ['down', 45],
      ['run', 195],
    ],
    dropDowntime: [0, 2],
    undercountP: 0.015,
  }),

  make('hotsplit', '近等价路径：慢速与停机不分', '一段 90 分钟里只有两个读数。45 分钟慢速与 30 分钟停机加 15 分钟运行给出几乎相同的计数。', {
    seed: 1007,
    start: START,
    minutes: 300,
    reportEvery: 30,
    initial: { product: 'F-2 轴套', idealCycleMs: 1750 },
    changes: [],
    schedule: [
      ['run', 70],
      ['slow', 45],
      ['down', 30],
      ['run', 155],
    ],
    undercountP: 0,
  }),
];

export function sampleById(id: string): Sample {
  return SAMPLES.find((s) => s.id === id) ?? SAMPLES[0];
}

export interface Comparison {
  sample: Sample;
  model: ReturnType<typeof analyze>;
  base: ReturnType<typeof baseline>;
  truth: {
    minutes: Record<string, number>;
    parts: number;
    idealParts: number;
    lossParts: number;
  };
  /** how far the model's down+slow minutes are from the planted path */
  minutesErr: { down: number; slow: number };
  baselineErr: { lossParts: number };
  modelErr: { lossParts: number };
}

export function compareSample(sample: Sample): Comparison {
  const model = analyze(sample.dataset);
  const base = baseline(sample.dataset);
  const truth = sample.dataset.truth!;
  const downModel = model.intervals
    .filter((i) => i.state === 'down')
    .reduce((s, i) => s + i.expectedMin, 0);
  const slowModel = model.intervals
    .filter((i) => i.state === 'slow')
    .reduce((s, i) => s + i.expectedMin, 0);
  return {
    sample,
    model,
    base,
    truth: {
      minutes: truth.minutes as unknown as Record<string, number>,
      parts: truth.parts,
      idealParts: truth.idealParts,
      lossParts: truth.lossParts,
    },
    minutesErr: {
      down: downModel - truth.minutes.down,
      slow: slowModel - truth.minutes.slow,
    },
    baselineErr: { lossParts: base.lossParts - truth.lossParts },
    modelErr: { lossParts: model.lossParts - truth.lossParts },
  };
}
