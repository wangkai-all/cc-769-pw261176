import { useEffect, useMemo, useState } from 'react';
import type { Analysis, Constraint, Dataset, InferredInterval, SegmentEvidence } from '../engine/types';
import { fmtClock, fmtMin, fmtNum, fmtRange, fmtWindow, pct, STATE_LABEL, STATE_SOFT, STATE_VAR } from '../ui/format';

export interface DrawerProps {
  analysis: Analysis;
  dataset: Dataset;
  interval: InferredInterval;
  constraints: Constraint[];
  onClose: () => void;
  onConfirm: (start: number, end: number) => void;
  onRemoveConstraint: (id: string) => void;
  onFocusSeg: (seg: number) => void;
}

/** Epoch ms for HH:MM on the day nearest `near`. A shift can run past midnight,
 *  and "01:20" typed for the end of a 23:00 start belongs to the next day; the
 *  model's own boundary is the best guide to which day the operator means. */
function atClock(near: number, hhmm: string): number | null {
  const m = /^(\d{1,2}):(\d{2})$/.exec(hhmm.trim());
  if (!m) return null;
  const h = Number(m[1]);
  const min = Number(m[2]);
  if (h > 23 || min > 59) return null;
  let best: number | null = null;
  for (const shift of [-1, 0, 1]) {
    const d = new Date(near);
    d.setDate(d.getDate() + shift);
    d.setHours(h, min, 0, 0);
    const t = d.getTime();
    if (best === null || Math.abs(t - near) < Math.abs(best - near)) best = t;
  }
  return best;
}

/** the minute a time input should show for a range: the tight end if pinned,
 *  otherwise the start of the solid core, rounded to the minute */
function toInput(t: number): string {
  return `${String(new Date(t).getHours()).padStart(2, '0')}:${String(new Date(t).getMinutes()).padStart(2, '0')}`;
}

export function IntervalDrawer({
  analysis,
  dataset,
  interval: iv,
  constraints,
  onClose,
  onConfirm,
  onRemoveConstraint,
  onFocusSeg,
}: DrawerProps) {
  const [startStr, setStartStr] = useState(() => toInput(iv.startHi));
  const [endStr, setEndStr] = useState(() => toInput(Math.max(iv.startHi, iv.endLo)));
  // The form is seeded from the interval it was opened on. It must not re-seed
  // when the recomputation shifts that interval's range - the operator's typed
  // times are theirs until they close the drawer or reset them.
  const [seedKey] = useState(() => `${iv.id}:${iv.startHi}:${iv.endLo}`);

  useEffect(() => {
    setStartStr(toInput(iv.startHi));
    setEndStr(toInput(Math.max(iv.startHi, iv.endLo)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seedKey]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const start = atClock(iv.startHi, startStr);
  const end = atClock(iv.endLo, endStr);
  const invalid = start === null || end === null || end <= start;

  const evidence = useMemo(() => {
    const ids = new Set(iv.segs);
    return analysis.evidence.filter((e) => ids.has(e.seg));
  }, [analysis.evidence, iv.segs]);

  const total = evidence.reduce(
    (acc, e) => {
      for (const s of ['run', 'slow', 'down'] as const) {
        acc[s] += e.minutes[s];
      }
      return acc;
    },
    { run: 0, slow: 0, down: 0 } as Record<'run' | 'slow' | 'down', number>,
  );
  const totalMin = Math.max(1e-9, total.run + total.slow + total.down);

  const crossRefs = constraints.filter((c) => c.end > iv.startLo && c.start < iv.endHi);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="区间证据" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <h3>
              {STATE_LABEL[iv.state]} · {fmtClock(iv.startLo)}–{fmtClock(iv.endHi)}
            </h3>
            <div className="small-note" style={{ marginTop: 4 }}>
              {fmtWindow(iv.startLo, iv.startHi, iv.endLo, iv.endHi)} ·{' '}
              {iv.ambiguous ? '边界只能给到范围' : '边界稳定到分钟'}
            </div>
          </div>
          <span className="spacer" />
          <button className="icon-btn" onClick={onClose} aria-label="关闭">
            ✕
          </button>
        </div>

        <div className="meta-grid">
          <div>
            <div className="m-label">状态后验占比</div>
            <div className="m-value num">{pct(iv.confidence, 1)}</div>
          </div>
          <div>
            <div className="m-label">路径支持率</div>
            <div className="m-value num">{pct(iv.pathShare, 0)}</div>
          </div>
          <div>
            <div className="m-label">预计时长</div>
            <div className="m-value num">{fmtMin(iv.expectedMin)}</div>
          </div>
          <div>
            <div className="m-label">约束</div>
            <div className="m-value">{iv.constrained ? '人工确认已钉住' : '无'}</div>
          </div>
        </div>

        <div className="subhead">区间内状态分布</div>
        <div className="mini-bars">
          {(['run', 'slow', 'down'] as const).map((s) => (
            <div className="mini-bar" key={s}>
              <span>{STATE_LABEL[s]}</span>
              <div className="track">
                <div className="fill" style={{ width: `${(total[s] / totalMin) * 100}%`, background: STATE_VAR[s] }} />
              </div>
              <span className="num" style={{ textAlign: 'right' }}>{fmtMin(total[s])}</span>
            </div>
          ))}
        </div>

        <div className="subhead" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span>其他读法</span>
        </div>
        {iv.alternatives.length === 0 ? (
          <p className="small-note">
            计数把这一个区间压到了单一状态：{STATE_LABEL[iv.state]}。它仍然只是一个读法，不是此刻的观测。
          </p>
        ) : (
          <div className="note-list" style={{ paddingLeft: 0, listStyle: 'none', margin: 0 }}>
            {iv.alternatives.map((a) => (
              <div key={a.state} className="mini-bar">
                <span>{STATE_LABEL[a.state]}</span>
                <div className="track">
                  <div className="fill" style={{ width: `${a.share * 100}%`, background: STATE_VAR[a.state], opacity: 0.75 }} />
                </div>
                <span className="num" style={{ textAlign: 'right' }} title="这种读法在区间内占的后验权重">
                  {pct(a.share, 0)}
                </span>
              </div>
            ))}
            <p className="small-note">
              这些读法在区间累计量上给出几乎相同的似然。模型不在这两者之间假装能分辨。
            </p>
          </div>
        )}

        <div className="subhead">
          原始计数支持 · {evidence.length} 段
        </div>
        <div className="seg-evidence">
          {evidence.map((e) => (
            <EvidenceCard key={e.seg} e={e} onFocus={() => onFocusSeg(e.seg)} />
          ))}
          {evidence.length === 0 && <p className="small-note">这个区间没有完整的计数段（全部落在回零或换型窗口里）。</p>}
        </div>

        <div className="subhead">这一段的近等价读法</div>
        <p className="small-note" style={{ margin: '-4px 0 8px' }}>
          每个计数段进来时只带一个累计增量。下面这些时间分配给出几乎相同的似然，模型不挑一个当真；
          条宽是该段自身时长的分配，不是全天分钟数。
        </p>
        <div className="split-bars">
          {evidence.flatMap((e) =>
            e.splits.slice(0, 2).map((h, i) => {
              const segMin = Math.max(1e-9, e.dtMin);
              return (
                <div className="split-row" key={`${e.seg}-${h.id}`} title={h.label}>
                  <span className="split-key num">
                    段#{e.seg} · 读法 {i + 1}
                  </span>
                  <div className="split-track" aria-label={`段 ${e.seg} 读法 ${i + 1}，${h.label}，权重 ${pct(h.weight, 0)}`}>
                    {(['run', 'slow', 'down'] as const).map((s) => {
                      const w = (h.minutes[s] / segMin) * 100;
                      if (w < 0.5) return null;
                      return (
                        <div
                          key={s}
                          style={{
                            width: `${w}%`,
                            background: STATE_VAR[s],
                            height: '100%',
                            opacity: i === 0 ? 0.95 : 0.5,
                          }}
                        />
                      );
                    })}
                  </div>
                  <span className="split-val num">
                    {pct(h.weight, 0)}
                  </span>
                </div>
              );
            }),
          )}
        </div>
        <p className="small-note" style={{ marginTop: 6 }}>
          同一段的条目共用一个 100% 宽；颜色是运行 / 低速 / 停机的分配，右侧是该读法在段内的权重。
        </p>

        {crossRefs.length > 0 && (
          <>
            <div className="subhead">区间内的约束 / 人工记录</div>
            <ul className="note-list">
              {crossRefs.map((c) => (
                <li key={c.id}>
                  {c.label}：{fmtClock(c.start)}–{fmtClock(c.end)}
                </li>
              ))}
              {iv.manualIds.map((id) => {
                const d = dataset.downtime.find((x) => x.id === id);
                return d ? (
                  <li key={id}>
                    人工记录「{d.note}」：{fmtClock(d.start)}–{fmtClock(d.end)}（只作锚点，不当真值）
                  </li>
                ) : null;
              })}
            </ul>
          </>
        )}

        <div className="confirm-box">
          <h4>确认这一段是停机？</h4>
          <p>
            确认会被当作硬约束重新计算全天：时间校到分钟，边界不再模糊。若计数显示这段时间在出件，
            系统不会顺从确认，而是指出冲突的计数段。
          </p>
          <div className="confirm-row">
            <label className="field">
              开始
              <input type="time" value={startStr} onChange={(e) => setStartStr(e.target.value)} />
            </label>
            <label className="field">
              结束
              <input type="time" value={endStr} onChange={(e) => setEndStr(e.target.value)} />
            </label>
            <button
              className="primary"
              disabled={invalid}
              onClick={() => {
                if (start !== null && end !== null && end > start) onConfirm(start, end);
              }}
            >
              确认停机
            </button>
            <button className="secondary" onClick={() => { setStartStr(toInput(iv.startHi)); setEndStr(toInput(Math.max(iv.startHi, iv.endLo))); }}>
              重置为模型范围
            </button>
          </div>
          {invalid && <p className="small-note" style={{ marginTop: 8, color: 'var(--critical)' }}>结束必须晚于开始。</p>}
          {constraints.length > 0 && (
            <div className="chips">
              {constraints.map((c) => (
                <span className="chip" key={c.id}>
                  停机 {fmtClock(c.start)}–{fmtClock(c.end)}
                  <button onClick={() => onRemoveConstraint(c.id)} aria-label={`移除约束 ${c.label}`}>
                    ✕
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

function EvidenceCard({ e, onFocus }: { e: SegmentEvidence; onFocus: () => void }) {
  const rate = e.delta !== null && e.dtMin > 0 ? e.delta / e.dtMin : null;
  const std = e.dtMin > 0 ? e.idealParts / e.dtMin : 0;
  return (
    <div className="seg-ev">
      <div className="sev-head">
        <span className="num">
          段 #{e.seg} · {fmtClock(e.t0)}–{fmtClock(e.t1)} · {fmtNum(e.dtMin)} 分
        </span>
        <button className="ghost" style={{ padding: '1px 8px', fontSize: 11 }} onClick={onFocus}>
          在时间带上定位
        </button>
      </div>
      <div className="mini-bars" style={{ marginTop: 8 }}>
        {(['run', 'slow', 'down'] as const).map((s) => {
          const [lo, hi] = e.minutesBand[s];
          return (
            <div className="mini-bar" key={s}>
              <span>{STATE_LABEL[s]}</span>
              <div className="track">
                <div
                  className="fill"
                  style={{
                    width: `${(hi / Math.max(1e-9, e.dtMin)) * 100}%`,
                    background: STATE_SOFT[s],
                    position: 'relative',
                  }}
                >
                  <span
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: `${(e.minutes[s] / Math.max(1e-9, hi)) * 100}%`,
                      background: STATE_VAR[s],
                      borderRadius: 3,
                    }}
                  />
                </div>
              </div>
              <span className="num" style={{ textAlign: 'right' }}>
                {fmtNum(lo)}–{fmtNum(hi)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="sev-note num">
        {e.delta === null ? (
          <>增量不可解析（回零或换型跨段）</>
        ) : (
          <>
            计数 +{fmtNum(e.delta)} 件 / {fmtNum(e.dtMin)} 分 = {fmtNum(rate ?? 0, 1)} 件/分
            {' · '}标准 {fmtNum(std, 1)} 件/分
          </>
        )}
      </div>
      <div className="sev-note">
        对标准损失 <span className="num">{fmtNum(e.lossParts)}</span> 件（
        <span className="num">{fmtRange(e.lossLo, e.lossHi)}</span>），状态后验{' '}
        {(['run', 'slow', 'down'] as const)
          .map((s) => `${STATE_LABEL[s]} ${pct(e.p[s], 0)}`)
          .join(' / ')}
      </div>
      {e.alternatives.length > 0 && (
        <div className="sev-note">
          另一种读法：{e.alternatives.map((a) => `${STATE_LABEL[a.state]} ${pct(a.share, 0)}`).join('、')}
        </div>
      )}      {e.notes.map((n, i) => (
        <div className="sev-note" key={i}>
          · {n}
        </div>
      ))}
    </div>
  );
}
