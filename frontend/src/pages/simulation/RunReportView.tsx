import { FileBarChart2, Route } from 'lucide-react';

import { api } from '../../api/client';
import type { ManeuverPrimary, Run } from '../../api/types';
import { LimitationsList, MetricTable, SourcePill } from '../../components/domain';
import { Card, CardHeader } from '../../components/ui/Card';
import { CopyText, KeyValueList } from '../../components/ui/Data';
import { ErrorNotice, LoadingBlock, Notice } from '../../components/ui/Feedback';
import { Pill } from '../../components/ui/Status';
import { useResource } from '../../hooks/useResource';
import { useI18n } from '../../i18n';
import { humanizeKey, outcomeText, shortHash, useFormat } from '../../lib/format';
import { MANEUVER_ICON, MANEUVER_TONE, TONE_BOX } from '../../lib/maneuver';
import { RUN_TONE } from '../../lib/tones';

const MANEUVER_KEYS = new Set(['maneuver_timeline', 'maneuver_counts', 'maneuver_timeline_truncated']);
const PRIMARIES: readonly ManeuverPrimary[] = ['STOP', 'BRAKE', 'SLOW_DOWN', 'ACCELERATE', 'CRUISE', 'TURN_LEFT', 'TURN_RIGHT', 'HOLD'];

interface SequenceRow {
  sim_time_s: number;
  frame_id: number;
  primary: ManeuverPrimary;
  speed_mps: number | null;
  reason_codes: string[];
}

const isPrimary = (value: unknown): value is ManeuverPrimary => PRIMARIES.includes(value as ManeuverPrimary);

/** details.maneuver_timeline of the run report (validated defensively: `details` is free-form JSON). */
function readSequence(details: Record<string, unknown>): { rows: SequenceRow[]; counts: [ManeuverPrimary, number][]; truncated: boolean } {
  const rows: SequenceRow[] = [];
  const raw = details.maneuver_timeline;
  if (Array.isArray(raw)) {
    for (const item of raw as unknown[]) {
      if (typeof item !== 'object' || item === null) continue;
      const entry = item as Record<string, unknown>;
      if (!isPrimary(entry.primary) || typeof entry.sim_time_s !== 'number' || typeof entry.frame_id !== 'number') continue;
      rows.push({
        sim_time_s: typeof entry.since_sim_time_s === 'number' ? entry.since_sim_time_s : entry.sim_time_s,
        frame_id: entry.frame_id,
        primary: entry.primary,
        speed_mps: typeof entry.speed_mps === 'number' ? entry.speed_mps : null,
        reason_codes: Array.isArray(entry.reason_codes) ? entry.reason_codes.filter((code): code is string => typeof code === 'string') : [],
      });
    }
  }
  const counts: [ManeuverPrimary, number][] = [];
  const rawCounts = details.maneuver_counts;
  if (typeof rawCounts === 'object' && rawCounts !== null) {
    for (const [key, value] of Object.entries(rawCounts as Record<string, unknown>)) {
      if (isPrimary(key) && typeof value === 'number') counts.push([key, value]);
    }
  }
  return { rows, counts, truncated: details.maneuver_timeline_truncated === true };
}

function ManeuverSequence({ details }: { details: Record<string, unknown> }) {
  const { t } = useI18n();
  const f = useFormat();
  const { rows, counts, truncated } = readSequence(details);
  return (
    <section data-testid="maneuver-sequence" aria-label={t.maneuver.sequenceTitle}>
      <p className="label-caps mb-1 flex items-center gap-1.5">
        <Route className="size-4" aria-hidden />
        {t.maneuver.sequenceTitle}
      </p>
      <p className="mb-2 text-xs text-muted">{t.maneuver.sequenceHint}</p>
      {rows.length === 0 ? (
        <p className="text-sm text-muted">{t.maneuver.sequenceEmpty}</p>
      ) : (
        <>
          {counts.length > 0 && (
            <ul className="mb-3 flex flex-wrap gap-1.5" aria-label={t.maneuver.ticks}>
              {counts.map(([primary, count]) => (
                <li key={primary}>
                  <Pill tone={MANEUVER_TONE[primary]}>
                    {t.maneuver.primary[primary]} · <span className="num">{count}</span> {t.maneuver.ticksUnit}
                  </Pill>
                </li>
              ))}
            </ul>
          )}
          <div className="scroll-x max-h-96 overflow-y-auto rounded-control border border-line">
            <table className="w-full text-left text-sm sm:min-w-[560px]">
              <thead className="sticky top-0 z-10 bg-surface">
                <tr className="border-b border-line text-xs text-muted">
                  <th scope="col" className="px-3 py-2 font-medium">{t.maneuver.colTime}</th>
                  <th scope="col" className="px-3 py-2 font-medium">{t.maneuver.colManeuver}</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">{t.maneuver.colFrame}</th>
                  <th scope="col" className="hidden px-3 py-2 text-right font-medium sm:table-cell">{t.maneuver.colSpeed}</th>
                  <th scope="col" className="hidden px-3 py-2 font-medium sm:table-cell">{t.maneuver.colReasons}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => {
                  const Icon = MANEUVER_ICON[row.primary];
                  return (
                    <tr key={index} className="border-b border-line/70 align-top last:border-0" data-testid="maneuver-sequence-row" data-maneuver={row.primary}>
                      <td className="num px-3 py-2 whitespace-nowrap">{f.num(row.sim_time_s, 2)} s</td>
                      <td className="px-3 py-2">
                        <span className="flex items-center gap-2">
                          <span className={`flex size-6 shrink-0 items-center justify-center rounded-md border ${TONE_BOX[MANEUVER_TONE[row.primary]]}`} aria-hidden>
                            <Icon className="size-3.5" strokeWidth={2.5} />
                          </span>
                          <span className="font-medium">{t.maneuver.primary[row.primary]}</span>
                        </span>
                        <span className="mt-1 block font-mono text-[11px] break-all text-muted sm:hidden">{row.reason_codes.join(', ')}</span>
                      </td>
                      <td className="num px-3 py-2 text-right text-muted">#{row.frame_id}</td>
                      <td className="num hidden px-3 py-2 text-right text-muted sm:table-cell">
                        {row.speed_mps === null ? '—' : `${f.num(row.speed_mps, 2)} m/s`}
                      </td>
                      <td className="hidden px-3 py-2 font-mono text-[11px] break-all text-muted sm:table-cell">{row.reason_codes.join(', ')}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {truncated && <p className="mt-2 text-xs text-muted">{t.maneuver.sequenceTruncated}</p>}
        </>
      )}
    </section>
  );
}

function DetailValue({ value }: { value: unknown }) {
  const f = useFormat();
  if (value === null || value === undefined) return <span className="text-muted">—</span>;
  if (typeof value === 'number') return <>{f.num(value, Number.isInteger(value) ? 0 : 3)}</>;
  if (typeof value === 'boolean' || typeof value === 'string') return <>{String(value)}</>;
  return (
    <pre className="max-h-48 overflow-auto rounded-lg bg-canvas p-2 text-left font-mono text-[11px] font-normal whitespace-pre-wrap">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

/** Measured run report (GET /runs/{id}/report), only for terminal runs. */
export function RunReportView({ run }: { run: Run }) {
  const { t } = useI18n();
  const f = useFormat();
  const report = useResource(`report:${run.id}:${run.state}`, (signal) => api.getRunReport(run.id, signal));

  return (
    <Card className="card-pad" testId="run-report">
      <CardHeader
        title={t.sim.reportTitle}
        icon={<FileBarChart2 className="size-5" />}
        actions={
          <div className="flex flex-wrap gap-2">
            <Pill tone={RUN_TONE[run.state]} dot testId="report-run-state">
              {t.states.run[run.state]}
            </Pill>
            {run.outcome && (
              <Pill tone={run.state === 'failed' ? 'danger' : run.state === 'arrived' ? 'success' : 'neutral'} testId="report-outcome">
                {outcomeText(t, run.outcome)}
              </Pill>
            )}
          </div>
        }
      />
      <div className="mt-4">
        {report.error && !report.data ? (
          report.error.code === 'report_unavailable' ? (
            <Notice tone="neutral">{t.errors.report_unavailable}</Notice>
          ) : (
            <ErrorNotice error={report.error} onRetry={report.reload} testId="report-error" />
          )
        ) : !report.data ? (
          <LoadingBlock label={t.sim.reportLoading} />
        ) : (
          <div className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              <KeyValueList
                items={[
                  { label: t.sim.reportStatus, value: report.data.status, testId: 'report-status' },
                  { label: t.sim.outcome, value: outcomeText(t, run.outcome) },
                  { label: t.common.source, value: <SourcePill source={report.data.perception_source} /> },
                  { label: t.sim.codeCommit, value: <CopyText value={report.data.code_commit} display={shortHash(report.data.code_commit, 10)} /> },
                  {
                    label: t.sim.modelSha,
                    value: report.data.model_sha256 ? (
                      <CopyText value={report.data.model_sha256} display={shortHash(report.data.model_sha256)} />
                    ) : (
                      '—'
                    ),
                  },
                  { label: t.common.created, value: f.dateTime(report.data.generated_utc) },
                ]}
              />
              <div className="space-y-2">
                <Notice tone={report.data.runtime_used_dynamic_ground_truth ? 'warning' : 'success'}>
                  {t.sim.runtimeTruth}: <code>runtime_used_dynamic_ground_truth = {String(report.data.runtime_used_dynamic_ground_truth)}</code>
                </Notice>
                <Notice tone="neutral">
                  {t.sim.truthUsed}: <code>evaluation_truth_used = {String(report.data.evaluation_truth_used)}</code>
                </Notice>
              </div>
            </div>
            <ManeuverSequence details={report.data.details} />
            <div>
              <p className="label-caps mb-2">{t.sim.metrics}</p>
              <MetricTable metrics={report.data.metrics} testId="report-metrics" />
            </div>
            {Object.keys(report.data.details).some((key) => !MANEUVER_KEYS.has(key)) && (
              <details className="rounded-control border border-line p-3">
                <summary className="cursor-pointer text-sm font-semibold">{t.sim.reportDetails}</summary>
                <KeyValueList
                  className="mt-2"
                  items={Object.entries(report.data.details)
                    .filter(([key]) => !MANEUVER_KEYS.has(key))
                    .map(([key, value]) => ({
                    label: humanizeKey(key),
                    value: <DetailValue value={value} />,
                  }))}
                />
              </details>
            )}
            <div>
              <p className="label-caps mb-2">{t.sim.limitations}</p>
              <LimitationsList items={report.data.limitations} testId="report-limitations" />
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
