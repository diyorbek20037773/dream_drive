import { useMemo } from 'react';

import type { ApiError } from '../api/client';
import type { AnyReasonCode } from '../api/types';
import type { Dict } from '../i18n';
import { useI18n } from '../i18n';

const pad2 = (value: number) => String(value).padStart(2, '0');

export interface Formatters {
  /** Fixed-precision number in the UI locale; null/undefined/NaN render as "not measured". */
  num: (value: number | null | undefined, digits?: number) => string;
  int: (value: number | null | undefined) => string;
  pct: (ratio: number | null | undefined, digits?: number) => string;
  bytes: (value: number | null | undefined) => string;
  mb: (bytes: number) => string;
  /** Seconds as m:ss (or h:mm:ss). */
  clock: (seconds: number | null | undefined) => string;
  /** Local date and time, YYYY-MM-DD HH:MM:SS. */
  dateTime: (iso: string | null | undefined) => string;
  time: (iso: string | null | undefined) => string;
  notMeasured: string;
}

export function makeFormatters(t: Dict): Formatters {
  const locale = t.meta.numberLocale;
  const cache = new Map<number, Intl.NumberFormat>();
  const nf = (digits: number) => {
    let format = cache.get(digits);
    if (!format) {
      format = new Intl.NumberFormat(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits });
      cache.set(digits, format);
    }
    return format;
  };
  const valid = (value: number | null | undefined): value is number =>
    typeof value === 'number' && Number.isFinite(value);
  const notMeasured = t.common.notMeasured;

  return {
    notMeasured,
    num: (value, digits = 1) => (valid(value) ? nf(digits).format(value) : notMeasured),
    int: (value) => (valid(value) ? nf(0).format(Math.round(value)) : notMeasured),
    pct: (ratio, digits = 1) => (valid(ratio) ? `${nf(digits).format(ratio * 100)} %` : notMeasured),
    bytes: (value) => {
      if (!valid(value)) return notMeasured;
      if (value < 1024) return `${nf(0).format(value)} B`;
      if (value < 1024 * 1024) return `${nf(1).format(value / 1024)} KB`;
      if (value < 1024 ** 3) return `${nf(1).format(value / 1024 ** 2)} MB`;
      return `${nf(2).format(value / 1024 ** 3)} GB`;
    },
    mb: (bytes) => nf(0).format(bytes / (1024 * 1024)),
    clock: (seconds) => {
      if (!valid(seconds) || seconds < 0) return notMeasured;
      const total = Math.floor(seconds);
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      return h > 0 ? `${h}:${pad2(m)}:${pad2(s)}` : `${m}:${pad2(s)}`;
    },
    dateTime: (iso) => {
      if (!iso) return '—';
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return '—';
      return (
        `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())} ` +
        `${pad2(date.getHours())}:${pad2(date.getMinutes())}:${pad2(date.getSeconds())}`
      );
    },
    time: (iso) => {
      if (!iso) return '—';
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return '—';
      return `${pad2(date.getHours())}:${pad2(date.getMinutes())}:${pad2(date.getSeconds())}`;
    },
  };
}

export function useFormat(): Formatters {
  const { t } = useI18n();
  return useMemo(() => makeFormatters(t), [t]);
}

/** Seconds between two ISO timestamps (end defaults to `nowMs`). */
export function elapsedSeconds(startIso: string | null | undefined, endIso: string | null | undefined, nowMs: number) {
  if (!startIso) return null;
  const start = Date.parse(startIso);
  const end = endIso ? Date.parse(endIso) : nowMs;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return Math.max(0, (end - start) / 1000);
}

export function shortHash(value: string | null | undefined, length = 12): string {
  if (!value) return '—';
  return value.length > length ? value.slice(0, length) : value;
}

export function errorText(t: Dict, error: ApiError): string {
  return t.errors[error.code] ?? error.message;
}

export function reasonText(t: Dict, code: string): string {
  return t.reasons[code as AnyReasonCode] ?? code;
}

/** Capability reasons are machine strings when known (e.g. model_missing); otherwise the API text is shown as-is. */
export function unavailableText(t: Dict, reason: string | null | undefined): string {
  if (!reason) return t.common.notAvailable;
  return t.unavailable[reason] ?? reason;
}

export function outcomeText(t: Dict, outcome: string | null | undefined): string {
  if (!outcome) return '—';
  return t.outcomes[outcome] ?? outcome;
}

export function stageText(t: Dict, stage: string | null | undefined): string {
  if (!stage) return '—';
  return t.stages[stage] ?? stage;
}

/** Humanizes an unknown metric key (snake_case) when no translation exists. */
export function humanizeKey(key: string): string {
  const text = key.replace(/_/g, ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}
