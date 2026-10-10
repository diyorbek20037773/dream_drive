/** Domain-specific building blocks shared by several pages. */
import { Activity, Cpu, Gauge, PlugZap, ScanEye } from 'lucide-react';

import type { Metric } from '../api/types';
import { useI18n } from '../i18n';
import { useApp } from '../state/app';
import { LoadingBlock } from './ui/Feedback';
import { metricLabel } from '../lib/evaluation';
import { errorText, reasonText, useFormat } from '../lib/format';
import type { Tone } from '../lib/tones';
import { Pill } from './ui/Status';

const SOURCE_TONE: Record<string, Tone> = {
  learned_model: 'info',
  synthetic_rgb_cv: 'cyan',
  classical_lane_fallback: 'warning',
  unavailable: 'neutral',
};

/** Perception source pill. Always shows the English source identifier next to the translated name. */
export function SourcePill({ source, extra, testId }: { source: string; extra?: string; testId?: string }) {
  const { t } = useI18n();
  const name = t.sources.perception[source as keyof typeof t.sources.perception] ?? source;
  return (
    <Pill tone={SOURCE_TONE[source] ?? 'neutral'} dot title={name} testId={testId}>
      <span className="font-mono">{source}</span>
      {extra ? ` · ${extra}` : ''}
    </Pill>
  );
}

export function ReasonList({ codes, testId, dense = false }: { codes: readonly string[]; testId?: string; dense?: boolean }) {
  const { t } = useI18n();
  if (codes.length === 0) return <p className="text-sm text-muted">—</p>;
  return (
    <ul className={dense ? 'space-y-1' : 'space-y-2'} data-testid={testId}>
      {codes.map((code) => (
        <li key={code} className="text-sm" data-reason-code={code}>
          <span className="text-ink">{reasonText(t, code)}</span>{' '}
          <code className="rounded bg-canvas px-1 py-0.5 font-mono text-[11px] break-all text-muted">{code}</code>
        </li>
      ))}
    </ul>
  );
}

export function ProvenanceLegend({ compact = false }: { compact?: boolean }) {
  const { t } = useI18n();
  const items = [
    { icon: <Cpu className="size-4" />, tone: 'text-accent', title: t.provenance.learned, id: 'learned_model', body: t.provenance.learnedBody },
    { icon: <ScanEye className="size-4" />, tone: 'text-cyan-ink', title: t.provenance.synthetic, id: 'synthetic_rgb_cv', body: t.provenance.syntheticBody },
    { icon: <Gauge className="size-4" />, tone: 'text-muted', title: t.provenance.telemetry, id: 'ego_telemetry', body: t.provenance.telemetryBody },
    { icon: <Activity className="size-4" />, tone: 'text-warning', title: t.provenance.truth, id: 'evaluation_truth', body: t.provenance.truthBody },
  ];
  return (
    <ul className={compact ? 'grid gap-3 sm:grid-cols-2' : 'grid gap-4 sm:grid-cols-2'} data-testid="provenance-legend">
      {items.map((item) => (
        <li key={item.id} className="flex items-start gap-3">
          <span className={`mt-0.5 shrink-0 ${item.tone}`} aria-hidden>
            {item.icon}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold">
              {item.title} <code className="ml-1 font-mono text-[11px] font-normal text-muted">{item.id}</code>
            </p>
            <p className="text-sm text-muted">{item.body}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

const METRIC_TONE: Record<Metric['status'], Tone> = { measured: 'success', not_evaluated: 'neutral', unsupported: 'warning' };

/** Metrics exactly as reported: value only when measured; otherwise the status and the API's note verbatim. */
export function MetricTable({ metrics, testId }: { metrics: Record<string, Metric>; testId?: string }) {
  const { t } = useI18n();
  const f = useFormat();
  const entries = Object.entries(metrics);
  if (entries.length === 0) return <p className="text-sm text-muted">{t.evaluation.noMetrics}</p>;
  return (
    <div className="scroll-x -mx-1" data-testid={testId}>
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead>
          <tr className="border-b border-line text-xs text-muted">
            <th scope="col" className="px-1 py-2 font-medium">{t.evaluation.metricName}</th>
            <th scope="col" className="px-1 py-2 text-right font-medium">{t.evaluation.value}</th>
            <th scope="col" className="px-1 py-2 font-medium">{t.evaluation.unit}</th>
            <th scope="col" className="px-1 py-2 text-right font-medium">{t.evaluation.samples}</th>
            <th scope="col" className="px-1 py-2 font-medium">{t.evaluation.status}</th>
            <th scope="col" className="px-1 py-2 font-medium">{t.evaluation.note}</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, metric]) => (
            <tr key={name} className="border-b border-line/70 align-top last:border-0" data-metric={name} data-status={metric.status}>
              <th scope="row" className="px-1 py-2 font-medium">
                {metricLabel(name, t)}
                <code className="block font-mono text-[11px] font-normal text-muted">{name}</code>
              </th>
              <td className="num px-1 py-2 text-right font-semibold">
                {metric.status === 'measured' && metric.value !== null ? (
                  f.num(metric.value, Math.abs(metric.value) >= 100 ? 1 : 3)
                ) : (
                  <span className="font-normal text-muted italic">{t.common.notMeasured}</span>
                )}
              </td>
              <td className="px-1 py-2 text-muted">{metric.unit}</td>
              <td className="num px-1 py-2 text-right">{f.int(metric.sample_count)}</td>
              <td className="px-1 py-2">
                <Pill tone={METRIC_TONE[metric.status]}>{t.states.metric[metric.status]}</Pill>
              </td>
              <td className="px-1 py-2 text-muted">{metric.note || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Placeholder for session-scoped data: "service offline" when no session could be created, otherwise loading. */
export function SessionPending() {
  const { t } = useI18n();
  const { sessionError } = useApp();
  if (!sessionError) return <LoadingBlock />;
  // The page-level banner carries the details and the retry button; cards only state why they are empty.
  return (
    <p className="flex items-center gap-2 py-2 text-sm text-muted" data-testid="session-pending-error" data-error-code={sessionError.code}>
      <PlugZap className="size-4 shrink-0 text-danger" aria-hidden />
      <span>
        <span className="font-medium text-ink">{t.common.serviceOffline}.</span> {errorText(t, sessionError)}
      </span>
    </p>
  );
}

export function LimitationsList({ items, testId }: { items: readonly string[]; testId?: string }) {
  if (items.length === 0) return <p className="text-sm text-muted">—</p>;
  return (
    <ul className="list-disc space-y-1 pl-5 text-sm text-ink/90" data-testid={testId}>
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
}
