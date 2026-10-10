import { LoaderCircle } from 'lucide-react';
import type { ReactNode } from 'react';

import type { Tone } from '../../lib/tones';

export type { Tone };

const TONES: Record<Tone, string> = {
  neutral: 'bg-canvas text-muted border-line',
  info: 'bg-accent/8 text-accent-strong border-accent/20',
  success: 'bg-success/8 text-success border-success/25',
  warning: 'bg-warning/8 text-warning border-warning/25',
  danger: 'bg-danger/8 text-danger border-danger/25',
  cyan: 'bg-cyan/8 text-cyan-ink border-cyan/25',
};
const DOTS: Record<Tone, string> = {
  neutral: 'bg-muted',
  info: 'bg-accent',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-danger',
  cyan: 'bg-cyan',
};

export function Pill({
  tone = 'neutral',
  children,
  dot = false,
  className = '',
  title,
  testId,
  onDark = false,
}: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  className?: string;
  title?: string;
  testId?: string;
  /** Light text variant for the dark media panels (keeps the tone in the dot only). */
  onDark?: boolean;
}) {
  const colors = onDark ? 'border-white/20 bg-white/10 text-white' : TONES[tone];
  return (
    <span
      title={title}
      data-testid={testId}
      className={`inline-flex max-w-full items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium whitespace-nowrap ${colors} ${className}`}
    >
      {dot && <span aria-hidden className={`size-1.5 shrink-0 rounded-full ${DOTS[tone]}`} />}
      <span className="truncate">{children}</span>
    </span>
  );
}

export function Spinner({ className = 'size-4', label }: { className?: string; label?: string }) {
  return (
    <LoaderCircle
      className={`animate-spin motion-reduce:animate-none ${className}`}
      aria-hidden={label ? undefined : true}
      aria-label={label}
      role={label ? 'status' : undefined}
    />
  );
}

export function ProgressBar({
  value,
  label,
  tone = 'info',
  className = '',
  testId,
}: {
  /** 0..1, or null for an indeterminate bar. */
  value: number | null;
  label: string;
  tone?: Tone;
  className?: string;
  testId?: string;
}) {
  const fill = tone === 'danger' ? 'bg-danger' : tone === 'success' ? 'bg-success' : tone === 'warning' ? 'bg-warning' : 'bg-accent';
  const pct = value === null ? null : Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
      data-testid={testId}
      className={`relative h-2 w-full overflow-hidden rounded-full bg-line ${className}`}
    >
      {pct === null ? (
        <div className={`dd-indeterminate absolute inset-y-0 left-0 w-2/5 rounded-full ${fill}`} />
      ) : (
        <div className={`h-full rounded-full transition-[width] duration-300 ${fill}`} style={{ width: `${pct}%` }} />
      )}
    </div>
  );
}
