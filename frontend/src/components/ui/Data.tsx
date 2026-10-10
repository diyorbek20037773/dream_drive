import { Check, Copy } from 'lucide-react';
import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';

import { useI18n } from '../../i18n';

export interface KeyValueItem {
  label: ReactNode;
  value: ReactNode;
  testId?: string;
}

export function KeyValueList({ items, className = '' }: { items: KeyValueItem[]; className?: string }) {
  return (
    <dl className={`divide-y divide-line ${className}`}>
      {items.map((item, index) => (
        <div key={index} className="flex items-baseline justify-between gap-4 py-2 text-sm">
          <dt className="shrink-0 text-muted">{item.label}</dt>
          <dd className="num min-w-0 text-right font-medium break-words text-ink" data-testid={item.testId}>
            {item.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** One measured value. `value` null renders the explicit "not measured" text, never 0. */
export function MetricTile({
  label,
  value,
  unit,
  hint,
  testId,
}: {
  label: string;
  value: string | null;
  unit?: string;
  hint?: ReactNode;
  testId?: string;
}) {
  const { t } = useI18n();
  return (
    <div className="rounded-control border border-line bg-surface p-4" data-testid={testId}>
      <p className="text-xs font-medium text-muted">{label}</p>
      {value === null || value === t.common.notMeasured ? (
        <p className="mt-1 text-sm font-medium text-muted italic" data-measured="false">
          {t.common.notMeasured}
        </p>
      ) : (
        <p className="num mt-1 text-xl font-semibold text-ink" data-measured="true">
          {value}
          {unit && <span className="ml-1 text-sm font-medium text-muted">{unit}</span>}
        </p>
      )}
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </div>
  );
}

/** Shortened hash with a copy-to-clipboard button; the full value is in the title and copied text. */
export function CopyText({ value, display, testId }: { value: string; display?: string; testId?: string }) {
  const { t } = useI18n();
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  useEffect(() => {
    if (state === 'idle') return;
    const handle = setTimeout(() => setState('idle'), 2000);
    return () => clearTimeout(handle);
  }, [state]);
  const copy = () => {
    if (!navigator.clipboard) {
      setState('failed');
      return;
    }
    navigator.clipboard.writeText(value).then(
      () => setState('copied'),
      () => setState('failed'),
    );
  };
  const label = state === 'copied' ? t.common.copied : state === 'failed' ? t.common.copyFailed : t.common.copy;
  return (
    <span className="inline-flex max-w-full items-center gap-1.5">
      <code className="truncate font-mono text-[13px]" title={value} data-testid={testId}>
        {display ?? value}
      </code>
      <button
        type="button"
        onClick={copy}
        className="inline-flex size-7 shrink-0 items-center justify-center rounded-lg text-muted hover:bg-canvas hover:text-ink"
        aria-label={`${label}: ${value}`}
        title={label}
      >
        {state === 'copied' ? <Check className="size-4 text-success" aria-hidden /> : <Copy className="size-4" aria-hidden />}
      </button>
      <span className="sr-only" aria-live="polite">
        {state !== 'idle' ? label : ''}
      </span>
    </span>
  );
}

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
  title?: string;
  testId?: string;
}

/** Radio-group styled as a segmented control (arrow keys move between options). */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  size = 'md',
  onDark = false,
}: {
  label: string;
  value: T;
  options: SegmentOption<T>[];
  onChange: (value: T) => void;
  size?: 'sm' | 'md';
  onDark?: boolean;
}) {
  const enabled = options.filter((option) => !option.disabled);
  const move = (delta: number) => {
    const index = enabled.findIndex((option) => option.value === value);
    const next = enabled[(index + delta + enabled.length) % enabled.length];
    if (next) onChange(next.value);
  };
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={`inline-flex rounded-control border p-0.5 ${onDark ? 'border-white/15 bg-white/5' : 'border-line bg-canvas'}`}
    >
      {options.map((option) => {
        const selected = option.value === value;
        const base = size === 'sm' ? 'h-7 px-2.5 text-xs' : 'h-8 px-3 text-sm';
        const colors = onDark
          ? selected
            ? 'bg-white text-night'
            : 'text-white/80 hover:text-white'
          : selected
            ? 'bg-surface text-ink shadow-card'
            : 'text-muted hover:text-ink';
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            tabIndex={selected ? 0 : -1}
            disabled={option.disabled}
            title={option.title}
            data-testid={option.testId}
            onClick={() => onChange(option.value)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
                event.preventDefault();
                move(1);
              } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
                event.preventDefault();
                move(-1);
              }
            }}
            className={`rounded-[10px] font-medium whitespace-nowrap transition-colors disabled:opacity-40 ${base} ${colors}`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
