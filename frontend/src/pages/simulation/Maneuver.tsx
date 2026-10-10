/**
 * "What is the car doing and why": the current maneuver (classified on the backend from the applied command and ego
 * telemetry), the decision basis (inputs the safety supervisor / FSM / controller evaluated for the same frame) and
 * the maneuver history. All values come from Run.live / SSE / the run report; nothing is derived or invented here.
 */
import { ArrowRight, History, ListChecks } from 'lucide-react';

import type { DecisionBasisItem, LaneStatus, LiveManeuver, QualityStatus } from '../../api/types';
import { Card, CardHeader } from '../../components/ui/Card';
import { Pill } from '../../components/ui/Status';
import type { Dict } from '../../i18n/uz';
import { useI18n } from '../../i18n';
import type { Formatters } from '../../lib/format';
import { reasonText, useFormat } from '../../lib/format';
import type { ManeuverChange } from '../../lib/maneuver';
import { BASIS_SOURCE_TONE, COMPARATOR_SYMBOL, MANEUVER_ICON, MANEUVER_TONE, TONE_BOX } from '../../lib/maneuver';

/** Large current-maneuver indicator. `compact` is the strip under the camera image. */
export function ManeuverIndicator({
  maneuver,
  frameId,
  compact = false,
  testId,
}: {
  maneuver: LiveManeuver;
  frameId: number;
  compact?: boolean;
  testId?: string;
}) {
  const { t } = useI18n();
  const f = useFormat();
  const Icon = MANEUVER_ICON[maneuver.primary];
  const tone = MANEUVER_TONE[maneuver.primary];
  const label = t.maneuver.primary[maneuver.primary];
  return (
    <div
      className={`flex items-center gap-3 rounded-card border motion-safe:transition-colors motion-safe:duration-300 ${TONE_BOX[tone]} ${compact ? 'px-3 py-2' : 'p-3'}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
      aria-label={`${t.maneuver.current}: ${label}`}
      data-testid={testId}
      data-maneuver={maneuver.primary}
      data-frame-id={frameId}
    >
      <span
        className={`flex shrink-0 items-center justify-center rounded-xl bg-surface/80 ${compact ? 'size-10' : 'size-14'}`}
        aria-hidden
      >
        <Icon className={compact ? 'size-6' : 'size-8'} strokeWidth={2.5} />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[11px] font-semibold tracking-wide text-muted uppercase">{t.maneuver.current}</p>
        <p className={`leading-tight font-semibold ${compact ? 'text-lg' : 'text-2xl'}`} data-testid={testId ? `${testId}-label` : undefined}>
          {label}
        </p>
        <p className="num mt-0.5 text-xs text-muted">
          {`${t.maneuver.longitudinal[maneuver.longitudinal]} · ${t.maneuver.lateral[maneuver.lateral]} · ${t.maneuver.since(f.num(maneuver.since_sim_time_s, 2))}`}
        </p>
      </div>
      <span className="num shrink-0 self-start rounded-full bg-surface/80 px-2 py-0.5 font-mono text-[11px] text-muted">
        {t.maneuver.cameraFrame(frameId)}
      </span>
    </div>
  );
}

/** Measured signals the maneuver label was classified from. */
export function ManeuverSignals({ maneuver }: { maneuver: LiveManeuver }) {
  const { t } = useI18n();
  const f = useFormat();
  return (
    <p className="text-xs text-muted" data-testid="maneuver-signals">
      {t.maneuver.accel}: <span className="num">{maneuver.accel_mps2 === null ? '—' : `${f.num(maneuver.accel_mps2, 2)} m/s²`}</span>
      {' · '}
      {t.maneuver.yawRate}: <span className="num">{maneuver.yaw_rate_radps === null ? '—' : `${f.num(maneuver.yaw_rate_radps, 3)} rad/s`}</span>
      {' · '}
      {t.maneuver.signal[maneuver.signal_source]}
    </p>
  );
}

function digitsFor(item: DecisionBasisItem, value: number): number {
  if (item.unit === 's' || item.unit === '1/m') return 3;
  if (item.unit) return 2;
  return Number.isInteger(value) ? 0 : 2;
}

function formatValue(t: Dict, f: Formatters, item: DecisionBasisItem): string {
  const value = item.value;
  if (value === null) return t.maneuver.none;
  if (typeof value === 'boolean') return value ? t.maneuver.yes : t.maneuver.no;
  if (typeof value === 'number') return `${f.num(value, digitsFor(item, value))}${item.unit ? ` ${item.unit}` : ''}`;
  if (item.key === 'lane_status') return t.states.lane[value as LaneStatus] ?? value;
  if (item.key === 'frame_quality') return t.states.quality[value as QualityStatus] ?? value;
  return value;
}

function formatThreshold(f: Formatters, item: DecisionBasisItem): string | null {
  if (item.threshold === null || item.comparator === null) return null;
  const symbol = COMPARATOR_SYMBOL[item.comparator];
  return `${symbol} ${f.num(item.threshold, digitsFor(item, item.threshold))}${item.unit ? ` ${item.unit}` : ''}`;
}

function BasisRow({ item, emphasize }: { item: DecisionBasisItem; emphasize: boolean }) {
  const { t } = useI18n();
  const f = useFormat();
  const label = t.maneuver.keys[item.key] ?? item.key;
  const threshold = formatThreshold(f, item);
  const codes = item.key === 'safety_codes' || item.key === 'failure_code';
  return (
    <li
      className={`flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm ${emphasize ? '' : 'text-muted'}`}
      data-testid="basis-item"
      data-basis-key={item.key}
      data-triggered={item.triggered}
      data-frame-id={item.input_frame_id}
    >
      <span
        className={`rounded-full border px-1.5 text-[10px] leading-4 font-medium whitespace-nowrap ${TONE_BOX[BASIS_SOURCE_TONE[item.source]]}`}
        title={item.source}
        data-testid="basis-source"
      >
        {t.maneuver.sources[item.source]}
      </span>
      <span className={emphasize ? 'font-medium text-ink' : ''} title={item.key}>
        {label}
      </span>
      {codes && typeof item.value === 'string' ? (
        <span
          className="font-mono text-xs break-all text-ink"
          title={item.value
            .split(',')
            .map((code) => reasonText(t, code))
            .join(' ')}
        >
          {item.value.split(',').join(', ')}
        </span>
      ) : (
        <span className={`num ${emphasize ? 'font-semibold text-ink' : ''}`}>{formatValue(t, f, item)}</span>
      )}
      {threshold && <span className="num text-muted">{threshold}</span>}
      {emphasize && item.effect && (
        <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
          <ArrowRight className="size-4 text-muted" aria-label="→" />
          <span className="font-semibold text-ink" data-testid="basis-effect">
            {t.maneuver.effects[item.effect]}
          </span>
        </span>
      )}
    </li>
  );
}

/** The inputs evaluated for the shown frame, triggered ones first with an arrow to the effect they drive. */
export function DecisionBasisList({
  items,
  frameId,
  cameraFrameId,
}: {
  items: readonly DecisionBasisItem[];
  frameId: number;
  cameraFrameId: number | null;
}) {
  const { t } = useI18n();
  const triggered = items.filter((item) => item.triggered);
  const others = items.filter((item) => !item.triggered);
  const same = cameraFrameId !== null && cameraFrameId === frameId;
  return (
    <section aria-label={t.maneuver.basisTitle} data-testid="decision-basis" data-frame-id={frameId}>
      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
        <p className="label-caps flex items-center gap-1.5">
          <ListChecks className="size-4" aria-hidden />
          {t.maneuver.basisTitle}
        </p>
        <Pill tone={same ? 'success' : 'neutral'} dot testId="basis-frame">
          <span className="num">{same ? t.maneuver.sameFrame(frameId) : cameraFrameId === null ? t.maneuver.cameraFrame(frameId) : t.maneuver.otherFrame(frameId, cameraFrameId)}</span>
        </Pill>
      </div>
      <p className="mb-2 text-xs text-muted">{t.maneuver.basisHint}</p>
      {items.length === 0 ? (
        <p className="text-sm text-muted">{t.maneuver.basisEmpty}</p>
      ) : (
        <>
          <p className="mb-1 text-xs font-semibold text-ink">{t.maneuver.basisTriggered}</p>
          {triggered.length === 0 ? (
            <p className="text-sm text-muted" data-testid="basis-none-triggered">
              {t.maneuver.basisNoneTriggered}
            </p>
          ) : (
            <ul className="space-y-1.5" aria-label={t.maneuver.basisTriggered} data-testid="basis-triggered">
              {triggered.map((item, index) => (
                <BasisRow key={`${item.key}-${index}`} item={item} emphasize />
              ))}
            </ul>
          )}
          {others.length > 0 && (
            <details className="mt-2 rounded-control border border-line px-2.5 py-1.5" data-testid="basis-others">
              <summary className="cursor-pointer text-xs font-medium text-muted">{t.maneuver.basisOther(others.length)}</summary>
              <ul className="mt-1.5 space-y-1">
                {others.map((item, index) => (
                  <BasisRow key={`${item.key}-${index}`} item={item} emphasize={false} />
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </section>
  );
}

function ChangeRow({ change }: { change: ManeuverChange }) {
  const { t } = useI18n();
  const f = useFormat();
  const Icon = MANEUVER_ICON[change.primary];
  return (
    <li className="flex items-start gap-2.5 py-2" data-testid="maneuver-history-item" data-maneuver={change.primary}>
      <span className={`mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg border ${TONE_BOX[MANEUVER_TONE[change.primary]]}`} aria-hidden>
        <Icon className="size-4" strokeWidth={2.5} />
      </span>
      <div className="min-w-0 flex-1">
        <p className="flex flex-wrap items-baseline gap-x-2 text-sm">
          <span className="font-semibold">{t.maneuver.primary[change.primary]}</span>
          <span className="num text-xs text-muted">
            {f.num(change.since_sim_time_s, 2)} s · {t.maneuver.cameraFrame(change.frame_id)}
          </span>
        </p>
        {change.reason_codes.length > 0 && (
          <p className="truncate font-mono text-[11px] text-muted" title={change.reason_codes.map((code) => reasonText(t, code)).join(' ')}>
            {change.reason_codes.join(', ')}
          </p>
        )}
      </div>
    </li>
  );
}

/** Scrollable list of the most recent maneuver changes of the selected run (newest first). */
export function ManeuverHistory({ items }: { items: readonly ManeuverChange[] }) {
  const { t } = useI18n();
  return (
    <Card className="card-pad" testId="maneuver-history">
      <CardHeader title={t.maneuver.historyTitle} icon={<History className="size-5" />} subtitle={t.maneuver.historyHint} />
      {items.length === 0 ? (
        <p className="mt-3 text-sm text-muted">{t.maneuver.historyEmpty}</p>
      ) : (
        <ol className="mt-2 max-h-72 divide-y divide-line overflow-y-auto pr-1" aria-label={t.maneuver.historyTitle}>
          {[...items].reverse().map((change) => (
            <ChangeRow key={`${change.since_sim_time_s}-${change.primary}-${change.frame_id}`} change={change} />
          ))}
        </ol>
      )}
    </Card>
  );
}
