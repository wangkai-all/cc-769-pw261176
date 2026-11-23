import { useEffect, useMemo, useState } from 'react';
import { analyze, baseline } from './engine/infer';
import type { Constraint, InferredInterval } from './engine/types';
import { SAMPLES, sampleById } from './samples';
import { Timeline } from './components/Timeline';
import { LossBreakdown } from './components/LossBreakdown';
import { IntervalDrawer } from './components/IntervalDrawer';
import { Verification } from './components/Verification';
import { fmtClock, fmtMin, fmtNum, fmtRange, pct, STATE_LABEL, STATE_VAR } from './ui/format';

type Theme = 'light' | 'dark';

function initialTheme(): Theme {
  const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('taktlens.theme') : null;
  if (saved === 'light' || saved === 'dark') return saved;
  return typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

const WARN_ICON: Record<string, string> = {
  'constraint-conflict': '!',
  'manual-conflict': '!',
  'bad-constraint': '!',
  'long-gap': '△',
  discontinuity: '△',
  'count-bias': '△',
  'zero-gain': '△',
  'no-manual': '·',
  'non-monotonic': '·',
  'change-without-reset': '·',
};

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [sampleId, setSampleId] = useState('steady');
  const [constraints, setConstraints] = useState<Constraint[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [focusedSeg, setFocusedSeg] = useState<number | null>(null);
  const [showPaths, setShowPaths] = useState(false);
  const [timelineView, setTimelineView] = useState<'chart' | 'table'>('chart');
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('taktlens.theme', theme);
    } catch {
      /* private mode: the toggle still works for this session */
    }
  }, [theme]);

  useEffect(() => {
    if (!flash) return;
    const h = setTimeout(() => setFlash(null), 4200);
    return () => clearTimeout(h);
  }, [flash]);

  const sample = sampleById(sampleId);
  const dataset = sample.dataset;

  const analysis = useMemo(() => analyze(dataset, { constraints }), [dataset, constraints]);
  const base = useMemo(() => baseline(dataset), [dataset]);

  const selected: InferredInterval | null = useMemo(
    () => analysis.intervals.find((i) => i.id === selectedId) ?? null,
    [analysis.intervals, selectedId],
  );

  const intervalForSeg = (seg: number): InferredInterval | undefined =>
    analysis.intervals.find((i) => i.segs.includes(seg));

  const pickSample = (id: string) => {
    setSampleId(id);
    setConstraints([]);
    setSelectedId(null);
    setFocusedSeg(null);
    setFlash(null);
  };

  const openInterval = (id: string) => {
    setSelectedId(id);
    setFocusedSeg(null);
  };

  const confirm = (start: number, end: number) => {
    const c: Constraint = {
      id: `c-${start}-${end}`,
      kind: 'downtime',
      start,
      end,
      label: `确认停机 ${fmtClock(start)}–${fmtClock(end)}`,
    };
    setConstraints((prev) => [...prev.filter((x) => x.id !== c.id), c]);
    setSelectedId(null);
    setFlash(`已把 ${fmtClock(start)}–${fmtClock(end)} 作为停机约束，全天重新推断。`);
  };

  const removeConstraint = (id: string) => {
    const c = constraints.find((x) => x.id === id);
    setConstraints((prev) => prev.filter((x) => x.id !== id));
    setSelectedId(null);
    if (c) setFlash(`已移除约束 ${fmtClock(c.start)}–${fmtClock(c.end)}，恢复无约束推断。`);
  };

  const downShare = analysis.intervals
    .filter((i) => i.state === 'down')
    .reduce((s, i) => s + i.expectedMin, 0);

  // Domain for the loss-range track. Both measures share one axis: the window
  // (path ensemble) and the tick (reconciliation) use actual part counts, not
  // each measure's own max, so the two are comparable by eye.
  const rangeWindow = useMemo(() => {
    const p = analysis.lossPosterior;
    const lo = Math.min(p.lo, analysis.lossParts);
    const hi = Math.max(p.hi, analysis.lossParts);
    // pad the domain so a zero-width window is not glued to an edge, and so the
    // scale labels stay to whole parts rather than fractions of one
    const pad = Math.max(1, (hi - lo) * 0.08, hi * 0.04);
    const dom0 = Math.max(0, lo - pad);
    const dom1 = hi + pad;
    const pos = (v: number) => ((v - dom0) / Math.max(1e-9, dom1 - dom0)) * 100;
    const tight = p.hi - p.lo < 1;
    return {
      dom0,
      dom1,
      tick: pos(analysis.lossParts),
      left: pos(p.lo),
      width: tight ? 0 : Math.max(1, pos(p.hi) - pos(p.lo)),
      tight,
    };
  }, [analysis.lossParts, analysis.lossPosterior]);

  const conflicts = analysis.warnings.filter((w) => w.code === 'constraint-conflict');

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>TaktLens</h1>
          <span className="tag">节拍推断与产能损失 · 只看累计计数</span>
        </div>
        <span className="spacer" />
        <button className="ghost" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
          {theme === 'light' ? '深色' : '浅色'}
        </button>
      </header>

      <section className="section" aria-labelledby="samples-h">
        <div className="section-head">
          <h2 id="samples-h">合成样例</h2>
          <span className="sub">每份都有已知的真实状态路径，用来检验推断偏差，不是用来调参的。</span>
        </div>
        <div className="picker" role="group" aria-label="选择样例">
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              className="pick"
              aria-pressed={s.id === sampleId}
              onClick={() => pickSample(s.id)}
            >
              <span className="pick-title">{s.title}</span>
              <span className="pick-blurb">{s.blurb}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="section" aria-label="关键指标">
        <div className="kpis">
          <div className="kpi">
            <span className="k-label">产能损失（对标准）</span>
            <div className="k-value">
              {fmtNum(analysis.lossParts)} <small>件</small>
            </div>
            <div className="k-band">
              可信范围 {fmtRange(analysis.lossLo, analysis.lossHi)} 件
            </div>
            <div className="k-note">≈ {fmtMin(analysis.lossMin)} 标准产能 · 六项严格加总</div>
          </div>
          <div className="kpi">
            <span className="k-label">无法归因的损失</span>
            <div className="k-value">
              {pct(analysis.unattributedShare, 0)}
            </div>
            <div className="k-band">停机当量 {fmtNum(analysis.unattributedMin)} 分钟</div>
            <div className="k-note">无人工记录、计数器停滞、未解释差额</div>
          </div>
          <div className="kpi">
            <span className="k-label">推断停机</span>
            <div className="k-value">
              {fmtMin(downShare)} <small>共 {analysis.intervals.filter((i) => i.state === 'down').length} 段</small>
            </div>
            <div className="k-note">
              {analysis.intervals.filter((i) => i.ambiguous).length} 个区间边界只能给范围
            </div>
          </div>
          <div className="kpi">
            <span className="k-label">读数覆盖</span>
            <div className="k-value">
              {fmtNum(analysis.segments.length)} <small>段</small>
            </div>
            <div className="k-note">
              {fmtNum(analysis.horizonMin)} 分钟 · {dataset.reports.length} 次上报 ·{' '}
              {analysis.blindParts > 0 ? `${fmtNum(analysis.blindParts)} 件不可解析` : '无不可解析窗口'}
            </div>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="timeline-h">
        <div className="section-head">
          <h2 id="timeline-h">状态时间带</h2>
          <span className="sub">
            实色是模型断言；斜纹是边界可能落点。点任一区间查看原始计数支持。
          </span>
          <span className="spacer" style={{ flex: 1 }} />
          {focusedSeg !== null && (
            <span className="chip" style={{ alignSelf: 'center' }}>
              已定位段 #{focusedSeg}
              <button onClick={() => setFocusedSeg(null)} aria-label="取消定位">
                ✕
              </button>
            </span>
          )}
          <span className="toggle-row">
            <button className="ghost" aria-pressed={showPaths} onClick={() => setShowPaths((v) => !v)}>
              {showPaths ? '隐藏等价路径' : '显示等价路径'}
            </button>
            <span className="segmented" role="group" aria-label="时间带视图">
              <button aria-pressed={timelineView === 'chart'} onClick={() => setTimelineView('chart')}>
                图
              </button>
              <button aria-pressed={timelineView === 'table'} onClick={() => setTimelineView('table')}>
                表
              </button>
            </span>
          </span>
        </div>

        {flash && (
          <div className="warn" style={{ borderLeftColor: 'var(--good)', marginBottom: 10 }}>
            <span className="w-icon" style={{ color: 'var(--good)' }}>
              ↻
            </span>
            <span className="w-text">{flash}</span>
            <span />
          </div>
        )}

        {timelineView === 'chart' ? (
          <Timeline
            analysis={analysis}
            dataset={dataset}
            selectedId={selectedId}
            onSelect={(iv) => openInterval(iv.id)}
            showPaths={showPaths}
            focusedSeg={focusedSeg}
          />
        ) : (
          <div className="card" style={{ padding: 16 }}>
            <div className="table-wrap">
              <table className="data">
                <caption className="sr-only">推断区间表</caption>
                <thead>
                  <tr>
                    <th>区间</th>
                    <th>状态</th>
                    <th>开始（范围）</th>
                    <th>结束（范围）</th>
                    <th className="num">预计</th>
                    <th className="num">后验</th>
                    <th className="num">路径支持</th>
                    <th>其他读法</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.intervals.map((iv) => (
                    <tr
                      key={iv.id}
                      onClick={() => openInterval(iv.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td className="num">#{iv.id.replace('iv-', '')}</td>
                      <td>
                        <span
                          className="swatch"
                          style={{ background: STATE_VAR[iv.state], marginRight: 6, verticalAlign: -1 }}
                        />
                        {STATE_LABEL[iv.state]}
                        {iv.constrained && <span className="uncertain-tag" style={{ color: 'var(--ink-2)' }}>已确认</span>}
                        {iv.ambiguous && <span className="uncertain-tag">边界不定</span>}
                      </td>
                      <td className="num">{iv.startLo === iv.startHi ? fmtClock(iv.startLo) : `${fmtClock(iv.startLo)}–${fmtClock(iv.startHi)}`}</td>
                      <td className="num">{iv.endLo === iv.endHi ? fmtClock(iv.endLo) : `${fmtClock(iv.endLo)}–${fmtClock(iv.endHi)}`}</td>
                      <td className="num">{fmtNum(iv.expectedMin)} 分</td>
                      <td className="num">{pct(iv.confidence, 0)}</td>
                      <td className="num">{pct(iv.pathShare, 0)}</td>
                      <td>{iv.alternatives.length ? iv.alternatives.map((a) => `${STATE_LABEL[a.state]} ${pct(a.share, 0)}`).join('、') : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="section" aria-labelledby="loss-h">
        <div className="section-head">
          <h2 id="loss-h">产能损失拆解</h2>
          <span className="sub">理想产量 − 计数产量 − 不可解析窗口 = 损失。分项是对这一个差额的归因，不是另一套总数。</span>
        </div>
        <LossBreakdown analysis={analysis} />
      </section>

      <section className="section" aria-labelledby="warn-h">
        <div className="section-head">
          <h2 id="warn-h">读数问题与冲突</h2>
          <span className="sub">
            {analysis.constrained ? `${constraints.length} 条人工约束已参与重算` : '当前没有人工约束'}
          </span>
        </div>
        <div className="warn-strip">
          {analysis.warnings.length === 0 && <p className="small-note">这次读数没有触发告警。</p>}
          {analysis.warnings.map((w, i) => {
            const conflict = w.code === 'constraint-conflict' || w.code === 'manual-conflict';
            return (
              <div className={`warn${conflict ? ' conflict' : ''}`} key={i}>
                <span className="w-icon" aria-hidden="true">
                  {WARN_ICON[w.code] ?? '△'}
                </span>
                <span className="w-text">{w.text}</span>
                <span className="w-segs">
                  {w.segs?.slice(0, 6).map((seg) => (
                    <button
                      key={seg}
                      onClick={() => {
                        const iv = intervalForSeg(seg);
                        if (iv) openInterval(iv.id);
                      }}
                    >
                      段 #{seg}
                    </button>
                  ))}
                </span>
              </div>
            );
          })}
        </div>
        {conflicts.length > 0 && (
          <p className="small-note" style={{ marginTop: 8 }}>
            冲突不等于确认失败：确认作为约束生效，但模型同时报告与之矛盾的计数段。点段号可查看该段的分段证据。
          </p>
        )}
        {constraints.length > 0 && (
          <div className="chips">
            {constraints.map((c) => (
              <span className="chip" key={c.id}>
                {c.label}
                <button onClick={() => removeConstraint(c.id)} aria-label={`移除 ${c.label}`}>
                  ✕
                </button>
              </span>
            ))}
          </div>
        )}
      </section>

      <section className="section" aria-labelledby="verify-h">
        <div className="section-head">
          <h2 id="verify-h">对照真实状态</h2>
          <span className="sub">模型、确定性基线与埋入的真实路径并排；偏差和无法归因的部分都写在明面上。</span>
        </div>
        <Verification analysis={analysis} base={base} dataset={dataset} />
      </section>

      <section className="section" aria-labelledby="prior-h">
        <div className="section-head">
          <h2 id="prior-h">先验与产出来源</h2>
          <span className="sub">{analysis.priorNote}</span>
        </div>
        <div className="card" style={{ padding: 16 }}>
          <div className="table-wrap">
            <table className="data">
              <caption className="sr-only">按产品期的标准节拍与先验</caption>
              <thead>
                <tr>
                  <th>期</th>
                  <th>产品</th>
                  <th className="num">标准节拍</th>
                  <th className="num">时段</th>
                  <th className="num">理想产量</th>
                  <th>发射先验</th>
                </tr>
              </thead>
              <tbody>
                {analysis.eras.map((e) => (
                  <tr key={e.era}>
                    <td className="num">#{e.era + 1}</td>
                    <td>{e.product}</td>
                    <td className="num">{(e.idealCycleMs / 1000).toFixed(2)} s</td>
                    <td className="num">{fmtClock(e.from)}–{fmtClock(e.to)}</td>
                    <td className="num">{fmtNum(e.idealParts)}</td>
                    <td>{e.priorLabel}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small-note" style={{ marginTop: 10 }}>
            每期的发射参数只由该期的标准节拍推出。换产品时旧参数整体作废，不会沿用上一期的节拍；
            换型那一刻的时间窗口产量不可知，计入“不可解析”，不摊到前后任一期。
            模型版本 <span className="num">{analysis.modelVersion}</span>。
          </p>
          <div className="subhead">路径集合给出的损失范围</div>
          <div
            className="range-track"
            role="img"
            aria-label={`路径集合的损失范围 ${fmtRange(analysis.lossPosterior.lo, analysis.lossPosterior.hi)} 件，对账口径 ${fmtNum(analysis.lossParts)} 件`}
          >
            {rangeWindow.width > 0 ? (
              <div
                className="range-window"
                style={{
                  left: `${rangeWindow.left}%`,
                  width: `${rangeWindow.width}%`,
                  background: 'var(--ink-3)',
                }}
              />
            ) : (
              <div
                className="range-point"
                style={{ left: `${rangeWindow.left}%` }}
                title={`路径集合一致给出 ${fmtNum(analysis.lossPosterior.parts)} 件`}
              />
            )}
            <div
              className="range-tick"
              style={{ left: `${rangeWindow.tick}%` }}
              title="对账口径的损失"
            />
          </div>
          <div className="range-scale num" aria-hidden="true">
            <span>{fmtNum(rangeWindow.dom0)}</span>
            <span>{fmtNum(rangeWindow.dom1)}</span>
          </div>
          <p className="small-note" style={{ marginTop: 6 }}>
            {rangeWindow.tight ? (
              <>
                这份读数没有回零或换型窗口，路径集合里的{' '}
                <span className="num">{analysis.lossPosterior.samples.length}</span> 条采样都落在同一个损失上（
                <span className="num">{fmtNum(analysis.lossPosterior.parts)}</span> 件），所以范围缩成一个点。
              </>
            ) : (
              <>
                路径集合（<span className="num">{analysis.lossPosterior.samples.length}</span> 条采样）给出的损失
                <span className="num"> {fmtRange(analysis.lossPosterior.lo, analysis.lossPosterior.hi)} </span> 件。
              </>
            )}
            {' '}实线刻度是对账口径的 <span className="num">{fmtNum(analysis.lossParts)}</span> 件。
            两者口径不同：路径集合按状态分配推算产量，对账口径用实际计数增量，所以不要求重合，只要求同量级。
          </p>
        </div>
      </section>

      {selected && (
        <IntervalDrawer
          analysis={analysis}
          dataset={dataset}
          interval={selected}
          constraints={constraints}
          onClose={() => {
            setSelectedId(null);
            setFocusedSeg(null);
          }}
          onConfirm={confirm}
          onRemoveConstraint={removeConstraint}
          onFocusSeg={(seg) => {
            setFocusedSeg(seg);
            setSelectedId(null);
            document.getElementById('timeline-h')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }}
        />
      )}
    </div>
  );
}
