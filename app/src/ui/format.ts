import type { StateName } from '../engine/types';

export const STATE_LABEL: Record<StateName, string> = {
  run: '运行',
  slow: '低速',
  down: '停机',
};

export const STATE_VAR: Record<StateName, string> = {
  run: 'var(--run)',
  slow: 'var(--slow)',
  down: 'var(--down)',
};

export const STATE_SOFT: Record<StateName, string> = {
  run: 'var(--run-soft)',
  slow: 'var(--slow-soft)',
  down: 'var(--down-soft)',
};

export const BUCKET_VAR: Record<string, string> = {
  'down-recorded': 'var(--down)',
  'down-unrecorded': 'var(--down)',
  slow: 'var(--slow)',
  changeover: 'var(--ink-3)',
  stall: 'var(--ink-2)',
  residual: 'var(--residual)',
};

export function fmtClock(t: number): string {
  const d = new Date(t);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

export function fmtClockSec(t: number): string {
  const d = new Date(t);
  return `${fmtClock(t)}:${String(d.getSeconds()).padStart(2, '0')}`;
}

export function fmtNum(x: number, digits = 0): string {
  return x.toLocaleString('zh-CN', { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function fmtSigned(x: number, digits = 0): string {
  const s = fmtNum(Math.abs(x), digits);
  if (Math.abs(x) < Math.pow(10, -digits) / 2) return s;
  return `${x < 0 ? '−' : '+'}${s}`;
}

export function fmtMin(x: number): string {
  if (x >= 60) {
    const h = Math.floor(x / 60);
    const m = Math.round(x - h * 60);
    return m ? `${h} 小时 ${m} 分` : `${h} 小时`;
  }
  return `${Math.round(x)} 分`;
}

export function fmtRange(lo: number, hi: number, digits = 0): string {
  return `${fmtNum(lo, digits)}–${fmtNum(hi, digits)}`;
}

export function fmtWindow(startLo: number, startHi: number, endLo: number, endHi: number): string {
  const start = startLo === startHi ? fmtClock(startLo) : `${fmtClock(startLo)}–${fmtClock(startHi)}`;
  const end = endLo === endHi ? fmtClock(endLo) : `${fmtClock(endLo)}–${fmtClock(endHi)}`;
  return `${start} → ${end}`;
}

export function pct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits)}%`;
}
