import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Analysis, Dataset, InferredInterval } from '../engine/types';
import { STATES } from '../engine/types';
import { fmtClock, fmtMin, fmtNum, pct, STATE_LABEL, STATE_VAR } from '../ui/format';

const MIN = 60_000;

/** Ink chosen per fill, not per theme: the dark-theme run colour is light, so a
 *  white label on it would fall to 2.4:1. The tokens carry the validated pair. */
const LABEL_ON = {
  run: 'var(--on-run)',
  slow: 'var(--on-slow)',
  down: 'var(--on-down)',
} as const;

export interface TimelineProps {
  analysis: Analysis;
  dataset: Dataset;
  selectedId: string | null;
  onSelect: (iv: InferredInterval) => void;
  showPaths: boolean;
  /** a count segment to trace across every lane, from the evidence drawer */
  focusedSeg?: number | null;
}

interface Tip {
  x: number;
  y: number;
  title: string;
  rows: string[];
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(1000);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(Math.max(680, w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, width };
}

export function Timeline({ analysis, dataset, selectedId, onSelect, showPaths, focusedSeg = null }: TimelineProps) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip | null>(null);

  const t0 = analysis.segments[0]?.t0 ?? 0;
  const t1 = analysis.segments[analysis.segments.length - 1]?.t1 ?? t0 + 1;
  const span = Math.max(1, t1 - t0);

  // the traced segment: its own pair of report timestamps, so the highlight is
  // an observation, not an inferred edge
  const segFocus = useMemo(
    () => (focusedSeg === null ? null : analysis.segments.find((s) => s.id === focusedSeg) ?? null),
    [analysis.segments, focusedSeg],
  );

  const padL = 70;
  const padR = 18;
  const plotW = Math.max(120, width - padL - padR);
  const x = useCallback((t: number) => padL + ((t - t0) / span) * plotW, [padL, plotW, span, t0]);

  const stateH = 46;
  const countH = 70;
  const manualH = 20;
  const hypH = showPaths ? 16 * 4 + 10 : 0;
  const axisH = 26;
  const laneGap = 14;

  const stateY = 6;
  const countY = stateY + stateH + laneGap;
  const manualY = countY + countH + laneGap;
  const hypY = manualY + manualH + laneGap;
  const axisY = hypY + hypH + 8;
  const H = axisY + axisH;

  const hours = useMemo(() => {
    const out: number[] = [];
    const start = new Date(t0);
    start.setMinutes(0, 0, 0);
    const step = span > 8 * 60 * MIN ? 2 * 60 * MIN : 60 * MIN;
    let t = start.getTime();
    if (t < t0) t += step;
    for (; t <= t1; t += step) out.push(t);
    return out;
  }, [t0, t1, span]);

  // cumulative count, split at every discontinuity so a reset is not drawn as a
  // cliff that pretends to be production
  const countPath = useMemo(() => {
    const groups: { t: number; cum: number }[][] = [];
    let cur: { t: number; cum: number }[] = [];
    let prev: { t: number; cum: number } | null = null;
    for (const r of dataset.reports) {
      if (prev && r.cum < prev.cum) {
        if (cur.length) groups.push(cur);
        cur = [];
      }
      cur.push(r);
      prev = r;
    }
    if (cur.length) groups.push(cur);
    return groups;
  }, [dataset]);

  const cumMax = useMemo(
    () => Math.max(1, ...dataset.reports.map((r) => r.cum)),
    [dataset],
  );

  const downSpans = analysis.hypotheses.slice(0, 3);

  const showTip = (e: React.MouseEvent, title: string, rows: string[]) => {
    setTip({ x: e.clientX, y: e.clientY, title, rows });
  };

  return (
    <div className="timeline-card card" ref={ref}>
      <div className="legend">
        {STATES.map((s) => (
          <span className="key" key={s}>
            <span className="swatch" style={{ background: STATE_VAR[s] }} />
            {STATE_LABEL[s]}
          </span>
        ))}
        <span className="key">
          <span className="swatch hatch" style={{ background: 'var(--ink-3)' }} />
          边界不确定（斜纹 = 可能落点，实色 = 模型断言）
        </span>
        <span className="key">
          <span className="swatch" style={{ background: 'var(--ink-3)', height: 2, width: 16 }} />
          累计计数
        </span>
      </div>

      {/* role=group, not img: the bands inside are focusable buttons, and an
          img role would hide them from assistive tech */}
      <svg className="tl-svg" viewBox={`0 0 ${width} ${H}`} role="group" aria-label="推断出的运行、低速与停机时间带">
        <defs>
          <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="currentColor" strokeWidth="2" opacity="0.75" />
          </pattern>
        </defs>

        {/* grid */}
        {hours.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={stateY} y2={axisY - 4} className="tl-grid" />
        ))}

        {/* row labels */}
        <text className="tl-rowlabel" x={0} y={stateY + 16}>状态推断</text>
        <text className="tl-rowlabel" x={0} y={countY + 12}>累计计数</text>
        <text className="tl-rowlabel" x={0} y={manualY + 13}>人工停机</text>
        {showPaths && <text className="tl-rowlabel" x={0} y={hypY + 12}>等价路径</text>}

        {/* state bands */}
        {analysis.intervals.map((iv) => {
          const coreL = x(iv.startHi);
          const coreR = x(iv.endLo);
          const fuzzyL = x(iv.startLo);
          const fuzzyR = x(iv.endHi);
          const selected = iv.id === selectedId;
          const label = STATE_LABEL[iv.state];
          const w = coreR - coreL;
          return (
            <g
              key={iv.id}
              className="tl-band"
              role="button"
              tabIndex={0}
              aria-label={`${label} ${fmtClock(iv.startLo)} 到 ${fmtClock(iv.endHi)}`}
              onClick={() => onSelect(iv)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(iv);
                }
              }}
              onMouseMove={(e) =>
                showTip(e, `${label} · ${fmtClock(iv.startLo)}–${fmtClock(iv.endHi)}`, [
                  `状态占比 ${pct(iv.confidence, 0)} · 路径支持 ${pct(iv.pathShare, 0)}`,
                  `预计时长 ${fmtMin(iv.expectedMin)}`,
                  iv.alternatives.length
                    ? `另一读法 ${iv.alternatives.map((a) => `${STATE_LABEL[a.state]} ${pct(a.share, 0)}`).join('、')}`
                    : '计数支持单一读法',
                  iv.constrained ? '已由人工确认钉住边界' : iv.ambiguous ? '边界为范围，不能给出分钟级时刻' : '边界稳定',
                ])
              }
              onMouseLeave={() => setTip(null)}
            >
              {/* uncertainty wings: hatched, never a solid claim */}
              {fuzzyL < coreL - 0.5 && (
                <g style={{ color: STATE_VAR[iv.state] }}>
                  <rect x={fuzzyL} y={stateY} width={coreL - fuzzyL} height={stateH} fill={STATE_VAR[iv.state]} opacity={0.16} />
                  <rect x={fuzzyL} y={stateY} width={coreL - fuzzyL} height={stateH} fill="url(#hatch)" />
                </g>
              )}
              {fuzzyR > coreR + 0.5 && (
                <g style={{ color: STATE_VAR[iv.state] }}>
                  <rect x={coreR} y={stateY} width={fuzzyR - coreR} height={stateH} fill={STATE_VAR[iv.state]} opacity={0.16} />
                  <rect x={coreR} y={stateY} width={fuzzyR - coreR} height={stateH} fill="url(#hatch)" />
                </g>
              )}
              <rect
                className="tl-fill"
                x={coreL}
                y={stateY}
                width={Math.max(1.5, w)}
                height={stateH}
                rx={iv.state === 'run' ? 2 : 3}
                fill={STATE_VAR[iv.state]}
                opacity={iv.kind === 'suspect' ? 0.72 : 1}
              />
              {selected && (
                <rect
                  className="tl-selected"
                  x={fuzzyL - 1}
                  y={stateY - 3}
                  width={Math.max(3, fuzzyR - fuzzyL + 2)}
                  height={stateH + 6}
                  rx={4}
                />
              )}
              {w > 34 && (
                <text
                  className="tl-label"
                  x={coreL + w / 2}
                  y={stateY + stateH / 2 + 3.5}
                  textAnchor="middle"
                  style={{ fill: LABEL_ON[iv.state] }}
                >
                  {label}
                </text>
              )}
            </g>
          );
        })}

        {/* count lane: one polyline per counter epoch */}
        {countPath.map((g, gi) => {
          if (g.length < 2) return null;
          const pts = g.map((r) => `${x(r.t)},${countY + countH - (r.cum / cumMax) * (countH - 6)}`).join(' ');
          return <polyline key={gi} points={pts} fill="none" stroke="var(--ink-3)" strokeWidth={1.5} strokeLinejoin="round" />;
        })}
        {dataset.reports.map((r, i) => (
          <circle
            key={i}
            cx={x(r.t)}
            cy={countY + countH - (r.cum / cumMax) * (countH - 6)}
            r={1.8}
            fill="var(--surface-raised)"
            stroke="var(--ink-3)"
            strokeWidth={1}
          />
        ))}
        <line x1={padL} x2={padL + plotW} y1={countY + countH} y2={countY + countH} stroke="var(--border)" strokeWidth={1} />
        <text className="tl-note" x={padL + plotW} y={countY + 9} textAnchor="end">
          峰值 {fmtNum(cumMax)} 件 · 断口为清零/换型
        </text>

        {/* manual downtime */}
        <line x1={padL} x2={padL + plotW} y1={manualY + manualH / 2} y2={manualY + manualH / 2} stroke="var(--border)" />
        {dataset.downtime.map((d) => (
          <rect
            key={d.id}
            x={x(d.start)}
            y={manualY + 3}
            width={Math.max(2, x(d.end) - x(d.start))}
            height={manualH - 6}
            rx={2}
            fill="var(--down)"
            opacity={0.55}
            onMouseMove={(e) => showTip(e, d.note, [`${fmtClock(d.start)}–${fmtClock(d.end)}`, '人工记录：仅作为边界锚点，不作为真值'])}
            onMouseLeave={() => setTip(null)}
          />
        ))}

        {/* alternative paths */}
        {showPaths &&
          downSpans.map((h, i) => (
            <g key={h.id}>
              <text className="tl-note" x={0} y={hypY + 12 + i * 16} fontSize={9.5}>
                路径 {i + 1}
              </text>
              <rect
                x={padL}
                y={hypY + 4 + i * 16}
                width={plotW}
                height={10}
                fill="var(--surface-sunken)"
                rx={2}
              />
              {h.down.map((d, k) => (
                <rect
                  key={k}
                  x={x(d.t0)}
                  y={hypY + 4 + i * 16}
                  width={Math.max(1.5, x(d.t1) - x(d.t0))}
                  height={10}
                  fill="var(--down)"
                  opacity={0.35 + 0.5 * h.weight}
                  rx={1.5}
                  onMouseMove={(e) =>
                    showTip(e, `等价路径 ${i + 1}`, [
                      `权重 ${pct(h.weight, 0)}`,
                      `停机 ${fmtMin(h.minutes.down)} · 低速 ${fmtMin(h.minutes.slow)}`,
                      `${fmtClock(d.t0)}–${fmtClock(d.t1)}`,
                    ])
                  }
                  onMouseLeave={() => setTip(null)}
                />
              ))}
            </g>
          ))}

        {/* traced count segment: an observed pair of timestamps, drawn across
            every lane so the reader can see what the drawer is talking about */}
        {segFocus && (
          <g className="tl-focus" pointerEvents="none">
            <rect
              x={x(segFocus.t0)}
              y={stateY - 2}
              width={Math.max(1, x(segFocus.t1) - x(segFocus.t0))}
              height={axisY - stateY + 2}
              fill="var(--ink)"
              opacity={0.06}
            />
            <line x1={x(segFocus.t0)} x2={x(segFocus.t0)} y1={stateY - 2} y2={axisY} className="tl-focus-edge" />
            <line x1={x(segFocus.t1)} x2={x(segFocus.t1)} y1={stateY - 2} y2={axisY} className="tl-focus-edge" />
          </g>
        )}

        {/* axis */}
        {hours.map((t) => (
          <text key={t} className="tl-axis" x={x(t)} y={axisY + 12} textAnchor="middle">
            {fmtClock(t)}
          </text>
        ))}
        <line x1={padL} x2={padL + plotW} y1={axisY} y2={axisY} stroke="var(--border-strong)" />
        <text className="tl-axis" x={0} y={axisY + 12}>
          {fmtClock(t0)} 起
        </text>
      </svg>

      {tip && (
        <div className="tooltip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 280), top: tip.y + 14 }}>
          <div className="tt-title">{tip.title}</div>
          {tip.rows.map((r, i) => (
            <div className="tt-row" key={i}>
              {r}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
