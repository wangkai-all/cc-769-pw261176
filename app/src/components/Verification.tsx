import type { Analysis, Dataset } from '../engine/types';
import type { BaselineAnalysis } from '../engine/types';
import { fmtMin, fmtNum, fmtSigned, pct } from '../ui/format';

interface Props {
  analysis: Analysis;
  base: BaselineAnalysis;
  dataset: Dataset;
}

/** Only rendered for synthetic samples: the whole point is that the reader can
 *  see how far the inference drifts from a known state path. */
export function Verification({ analysis, base, dataset }: Props) {
  const truth = dataset.truth;
  if (!truth) return null;

  const modelDown = analysis.intervals.filter((i) => i.state === 'down').reduce((s, i) => s + i.expectedMin, 0);
  const modelSlow = analysis.intervals.filter((i) => i.state === 'slow').reduce((s, i) => s + i.expectedMin, 0);

  const truthLoss = Math.max(0, truth.idealParts - truth.parts);
  const modelErr = analysis.lossParts - truthLoss;
  const baseErr = base.lossParts - truthLoss;
  const scale = Math.max(1, truth.idealParts);
  const openShare = analysis.unattributedShare;

  const residual = analysis.buckets.find((b) => b.key === 'residual');
  const stall = analysis.buckets.find((b) => b.key === 'stall');
  const unrecorded = analysis.buckets.find((b) => b.key === 'down-unrecorded');

  // one shared minute scale for the whole group, so the three bars compare
  const minuteMax = Math.max(1, modelDown, modelSlow, truth.minutes.down, truth.minutes.slow) * 1.1;

  return (
    <div className="compare">
      <div className="compare-card card">
        <h4>对照已知状态路径</h4>
        <div className="c-sub">同一份计数，分别交给模型与确定性基线，再和埋入的真实状态比。</div>
        <div className="tick-key">
          <span className="truth-tick-sample" aria-hidden="true" />
          竖线 = 真实损失，同一条轴上对比
        </div>
        <BarPair label="真实损失" value={truthLoss} max={scale} color="var(--ink-3)" truth={truthLoss} />
        <BarPair label="模型估计" value={analysis.lossParts} max={scale} color="var(--down)" truth={truthLoss} />
        <BarPair label="确定性基线" value={base.lossParts} max={scale} color="var(--slow)" truth={truthLoss} />
        <p className="small-note" style={{ marginTop: 10 }}>
          模型偏差 <span className="num">{fmtSigned(modelErr)}</span> 件，基线偏差{' '}
          <span className="num">{fmtSigned(baseErr)}</span> 件。真实损失是理想产量减去实际出件；真实出件里有漏计，
          所以“真实损失”本身也带着计数误差，模型不可能恰好落在它上面。
        </p>
        <div className="subhead">停机时长</div>
        <div className="mini-bars">
          <MiniPair label="模型停机" value={modelDown} max={minuteMax} color="var(--down)" />
          <MiniPair label="真实停机" value={truth.minutes.down} max={minuteMax} color="var(--ink-3)" tick={truth.minutes.down} />
          <MiniPair label="模型低速" value={modelSlow} max={minuteMax} color="var(--slow)" />
        </div>
      </div>

      <div className="compare-card card">
        <h4>哪些损失仍然无法归因</h4>
        <div className="c-sub">这不是误差，是读数本身不给的信息。</div>
        <div className="mini-bars">
          <Row label="无人工记录的停机" value={unrecorded?.parts ?? 0} max={scale} />
          <Row label="计数停滞 / 冻结" value={stall?.parts ?? 0} max={scale} />
          <Row label="未解释差额" value={residual?.parts ?? 0} max={scale} />
        </div>
        <p className="small-note" style={{ marginTop: 10 }}>
          约 <span className="num">{pct(openShare, 0)}</span> 的损失（
          <span className="num">{fmtNum(analysis.unattributedMin)}</span> 分钟停机当量）不能落到一条人工记录上。
          {stall && stall.parts > 0
            ? '其中计数停滞的部分只能说明“产出低于标准”，分不清是停机、换型准备，还是计数器本身在漏。'
            : '其中无记录停机的部分只能说明“产出低于标准”，分不清是停机、换型准备，还是计数器本身在漏。'}
        </p>
        <div className="subhead">基线做不到的事</div>
        <ul className="note-list">
          {base.blindSpots.slice(0, 3).map((b) => (
            <li key={b.seg}>
              段 #{b.seg}：{b.reason}
            </li>
          ))}
          <li>把区间增量平均到每一分钟，等于假设一分钟都没停；停机被摊平成“略低于标准”。</li>
        </ul>
      </div>

      <div className="compare-card card">
        <h4>埋入的损伤</h4>
        <div className="c-sub">{dataset.note}</div>
        <div className="chips" style={{ marginTop: 0 }}>
          {truth.injects.map((x) => (
            <span className="chip" key={x}>
              {x}
            </span>
          ))}
        </div>
        <div className="subhead">模型自报的读数问题</div>
        <ul className="note-list">
          {analysis.warnings.length === 0 && <li>没有触发读数质量告警。</li>}
          {analysis.warnings.slice(0, 5).map((w, i) => (
            <li key={i}>{w.text}</li>
          ))}
        </ul>
        <p className="small-note" style={{ marginTop: 10 }}>
          真实理想产量 <span className="num">{fmtNum(truth.idealParts)}</span>，
          真实出件 <span className="num">{fmtNum(truth.parts)}</span>，
          真实停机 <span className="num">{fmtMin(truth.minutes.down)}</span>。
        </p>
      </div>
    </div>
  );
}

function BarPair({ label, value, max, color, truth }: { label: string; value: number; max: number; color: string; truth: number }) {
  return (
    <div className="bar-pair">
      <span>{label}</span>
      <div className="track">
        <div className="fill" style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: color }} />
        <div className="truth-tick" style={{ left: `${Math.min(100, (truth / max) * 100)}%` }} />
      </div>
      <span className="num" style={{ textAlign: 'right' }}>
        {fmtNum(value)}
      </span>
    </div>
  );
}

function MiniPair({
  label,
  value,
  max,
  color,
  tick,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  tick?: number;
}) {
  return (
    <div className="mini-bar">
      <span>{label}</span>
      <div className="track">
        <div
          className="fill"
          style={{ width: `${Math.min(100, (value / Math.max(1, max)) * 100)}%`, background: color }}
        />
        {tick !== undefined && (
          <div className="truth-tick" style={{ left: `${Math.min(100, (tick / Math.max(1, max)) * 100)}%` }} />
        )}
      </div>
      <span className="num" style={{ textAlign: 'right' }}>
        {fmtMin(value)}
      </span>
    </div>
  );
}

function Row({ label, value, max }: { label: string; value: number; max: number }) {
  return (
    <div className="mini-bar">
      <span style={{ fontSize: 11.5 }}>{label}</span>
      <div className="track">
        <div
          className="fill"
          style={{
            width: `${Math.min(100, (value / max) * 100)}%`,
            background: 'var(--ink-3)',
            backgroundImage: 'repeating-linear-gradient(45deg, transparent 0 3px, var(--hatch-line) 3px 5px)',
          }}
        />
      </div>
      <span className="num" style={{ textAlign: 'right' }}>
        {fmtNum(value)}
      </span>
    </div>
  );
}
