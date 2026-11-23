import { useMemo, useState } from 'react';
import type { Analysis, LossBucket } from '../engine/types';
import { BUCKET_VAR, fmtNum, fmtRange, pct } from '../ui/format';

type View = 'chart' | 'table';

const HATCHED = new Set(['down-unrecorded', 'stall', 'residual']);

interface Props {
  analysis: Analysis;
}

function bucketFill(b: LossBucket): string {
  return BUCKET_VAR[b.key] ?? 'var(--ink-3)';
}

export function LossBreakdown({ analysis }: Props) {
  const [view, setView] = useState<View>('chart');
  const buckets = analysis.buckets;
  const total = Math.max(1, buckets.reduce((s, b) => s + b.parts, 0));

  const rows = useMemo(
    () =>
      buckets
        .map((b) => ({
          ...b,
          share: b.parts / total,
        }))
        .sort((a, b) => b.parts - a.parts),
    [buckets, total],
  );

  return (
    <div className="breakdown">
      <div className="stack-card card">
        <div className="section-head" style={{ marginBottom: 12 }}>
          <h2>损失构成</h2>
          <span className="sub">
            合计 <span className="num">{fmtNum(analysis.lossParts)}</span> 件 ·
            区间 <span className="num">{fmtRange(analysis.lossLo, analysis.lossHi)}</span> 件
          </span>
          <span className="spacer" style={{ flex: 1 }} />
          <span className="segmented" role="group" aria-label="视图切换">
            <button aria-pressed={view === 'chart'} onClick={() => setView('chart')}>
              图
            </button>
            <button aria-pressed={view === 'table'} onClick={() => setView('table')}>
              表
            </button>
          </span>
        </div>

        {view === 'chart' ? (
          <>
            <div className="stackbar" role="img" aria-label="按原因拆解的产能损失，各段为占比">
              {rows
                .filter((b) => b.parts > 0)
                .map((b) => (
                  <div
                    key={b.key}
                    className="seg"
                    style={{
                      flexGrow: Math.max(b.parts, 1e-6),
                      flexBasis: 0,
                      minWidth: 2,
                      background: bucketFill(b),
                    }}
                    title={`${b.label}：${fmtNum(b.parts)} 件（${pct(b.share, 1)}）`}
                  >
                    {HATCHED.has(b.key) && <div className="hatch" />}
                  </div>
                ))}
            </div>
            <div className="bucket-list">
              {rows.map((b) => (
                <div className="bucket-row" key={b.key}>
                  <span
                    className="swatch"
                    style={{
                      background: bucketFill(b),
                      backgroundImage: HATCHED.has(b.key)
                        ? 'repeating-linear-gradient(45deg, transparent 0 3px, var(--hatch-line) 3px 5px)'
                        : undefined,
                      marginTop: 3,
                    }}
                  />
                  <div>
                    <div className="b-name">
                      {b.label}
                      {b.unattributed >= 0.99 && <span className="uncertain-tag">无法归因</span>}
                      {b.unattributed > 0 && b.unattributed < 0.99 && (
                        <span className="uncertain-tag">{pct(b.unattributed, 0)} 待定</span>
                      )}
                    </div>
                    <div className="b-detail">{b.detail}</div>
                  </div>
                  <div className="b-value">
                    <span className="num">{fmtNum(b.parts)}</span>
                    <span className="pct">{pct(b.share, 1)}</span>
                    <div className="b-detail num" style={{ textAlign: 'right' }}>
                      {fmtRange(b.lo, b.hi)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <p className="small-note" style={{ marginTop: 12 }}>
              六个分区严格加总到 <span className="num">{fmtNum(analysis.lossParts)}</span> 件：理想产量减去可解析的计数产量，
              再减去不可解析窗口的估计产量。分段宽度只表示份额，不表示置信度；斜纹段是归因仍未闭合的部分。
            </p>
          </>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <caption className="sr-only">产能损失分项表</caption>
              <thead>
                <tr>
                  <th>分项</th>
                  <th className="num">件</th>
                  <th className="num">占比</th>
                  <th className="num">可信范围</th>
                  <th className="num">待定比例</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((b) => (
                  <tr key={b.key}>
                    <td>{b.label}</td>
                    <td className="num">{fmtNum(b.parts)}</td>
                    <td className="num">{pct(b.share, 1)}</td>
                    <td className="num">{fmtRange(b.lo, b.hi)}</td>
                    <td className="num">{pct(b.unattributed, 0)}</td>
                    <td style={{ whiteSpace: 'normal', minWidth: 240 }}>{b.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card stack-card">
        <div className="section-head" style={{ marginBottom: 10 }}>
          <h2>产量对账</h2>
        </div>
        <div className="mini-bars">
          <MiniBar label="理想产量" value={analysis.idealParts} max={analysis.idealParts} color="var(--ink-3)" />
          <MiniBar label="计数产量" value={analysis.countedParts} max={analysis.idealParts} color="var(--run)" />
          <MiniBar
            label="不可解析"
            value={analysis.blindParts}
            max={analysis.idealParts}
            color="var(--slow)"
            note="回零/换型窗口的估计"
          />
          <MiniBar label="产能损失" value={analysis.lossParts} max={analysis.idealParts} color="var(--down)" note="含未解释差额" />
        </div>
        <div className="subhead">损失换算</div>
        <p className="small-note">
          约 <span className="num">{fmtNum(analysis.lossMin)}</span> 分钟的标准产能，
          其中 <span className="num">{fmtNum(analysis.unattributedMin)}</span> 分钟的停机没有人工记录可背书，
          占损失 <span className="num">{pct(analysis.unattributedShare, 0)}</span>。按件计，这部分被留在带「无法归因」标签的分项里。
        </p>
        <div className="subhead">读数质量</div>
        <p className="small-note">
          覆盖 <span className="num">{fmtNum(analysis.horizonMin)}</span> 分钟、
          <span className="num">{analysis.segments.length}</span> 个计数段。
          计数器回零后的段不参与差值计算；换型跨段只保留一个不可知窗口。
        </p>
      </div>
    </div>
  );
}

function MiniBar({
  label,
  value,
  max,
  color,
  note,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  note?: string;
}) {
  return (
    <div className="mini-bar">
      <span>{label}</span>
      <div className="track">
        <div className="fill" style={{ width: `${Math.min(100, (value / Math.max(1, max)) * 100)}%`, background: color }} />
      </div>
      <span className="num" style={{ textAlign: 'right' }} title={note}>
        {fmtNum(value)}
      </span>
    </div>
  );
}
