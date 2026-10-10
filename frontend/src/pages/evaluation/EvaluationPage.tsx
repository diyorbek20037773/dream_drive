import { ClipboardCheck, FlaskConical, GitCompare, Microscope, PlayCircle } from 'lucide-react';
import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';

import type { ApiError } from '../../api/client';
import { api, toApiError } from '../../api/client';
import type { Evaluation } from '../../api/types';
import { LimitationsList, MetricTable, SessionPending } from '../../components/domain';
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, PageHeader } from '../../components/ui/Card';
import { CopyText, KeyValueList } from '../../components/ui/Data';
import { EmptyState, ErrorNotice, LoadingBlock, Notice } from '../../components/ui/Feedback';
import { Pill, ProgressBar } from '../../components/ui/Status';
import { useResync } from '../../hooks/useEvents';
import { useResource } from '../../hooks/useResource';
import { useInterval, useNow } from '../../hooks/useTimers';
import { useI18n } from '../../i18n';
import { evaluationTitle, metricLabel } from '../../lib/evaluation';
import { elapsedSeconds, shortHash, useFormat } from '../../lib/format';
import { EVALUATION_TONE } from '../../lib/tones';
import { useApp } from '../../state/app';

const ML_KINDS = new Set<Evaluation['kind']>(['model_benchmark', 'video_ml_accuracy']);
const MAX_COMPARE = 4;

function parseSeeds(text: string): number[] | null {
  const parts = text
    .split(/[\s,;]+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length < 1 || parts.length > 10) return null;
  const seeds = parts.map((part) => (/^\d{1,7}$/.test(part) ? Number(part) : Number.NaN));
  if (seeds.some((seed) => !Number.isInteger(seed) || seed < 0 || seed > 1_000_000)) return null;
  return [...new Set(seeds)];
}

const isRunning = (evaluation: Evaluation) => evaluation.state === 'queued' || evaluation.state === 'running';

function measuredCount(evaluation: Evaluation): string {
  const metrics = Object.values(evaluation.metrics);
  return `${metrics.filter((metric) => metric.status === 'measured').length} / ${metrics.length}`;
}

function EvaluationDetail({ evaluation }: { evaluation: Evaluation }) {
  const { t } = useI18n();
  const f = useFormat();
  const now = useNow(1000, isRunning(evaluation));
  return (
    <Card className="card-pad" testId="evaluation-detail">
      <CardHeader
        title={t.evaluation.detailTitle}
        icon={<Microscope className="size-5" />}
        subtitle={[t.evaluation.kinds[evaluation.kind], evaluationTitle(evaluation, t, f.int)].filter(Boolean).join(' — ')}
        actions={
          <Pill tone={EVALUATION_TONE[evaluation.state]} dot testId="evaluation-detail-state">
            {t.states.evaluation[evaluation.state]}
          </Pill>
        }
      />
      {isRunning(evaluation) && (
        <div className="mt-4" aria-live="polite">
          <ProgressBar value={null} label={t.evaluation.suiteRunning} />
          <p className="num mt-1.5 text-xs text-muted">
            {t.evaluation.suiteProgress} {t.common.elapsed}: {f.clock(elapsedSeconds(evaluation.created_utc, null, now))}
          </p>
        </div>
      )}
      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        <KeyValueList
          items={[
            { label: t.evaluation.colKind, value: <code className="font-mono text-xs">{evaluation.kind}</code> },
            { label: t.evaluation.colSource, value: <code className="font-mono text-xs">{evaluation.source}</code> },
            { label: t.evaluation.colSamples, value: f.int(evaluation.sample_count), testId: 'evaluation-samples' },
            ...(evaluation.dataset
              ? [
                  {
                    label: t.evaluation.colDataset,
                    value: (
                      <span data-testid="evaluation-dataset-detail">
                        {evaluation.dataset.url ? (
                          <a href={evaluation.dataset.url} target="_blank" rel="noreferrer noopener" className="text-accent-strong hover:underline">
                            {evaluationTitle(evaluation, t, f.int)}
                          </a>
                        ) : (
                          evaluationTitle(evaluation, t, f.int)
                        )}
                        <span className="block text-xs text-muted">
                          {[
                            evaluation.dataset.split ? `${t.evaluation.datasetSplit}: ${evaluation.dataset.split}` : null,
                            evaluation.dataset.licence ? `${t.evaluation.datasetLicence}: ${evaluation.dataset.licence}` : null,
                          ]
                            .filter(Boolean)
                            .join(' · ')}
                        </span>
                      </span>
                    ),
                  },
                ]
              : []),
            { label: t.evaluation.seedsLabel, value: evaluation.seeds?.join(', ') ?? '—' },
          ]}
        />
        <KeyValueList
          items={[
            { label: t.common.created, value: f.dateTime(evaluation.created_utc) },
            { label: t.common.finished, value: f.dateTime(evaluation.finished_utc) },
            {
              label: t.evaluation.colCommit,
              value: evaluation.code_commit ? <CopyText value={evaluation.code_commit} display={shortHash(evaluation.code_commit, 10)} /> : '—',
            },
            {
              label: t.evaluation.colModel,
              value: evaluation.model_sha256 ? <CopyText value={evaluation.model_sha256} display={shortHash(evaluation.model_sha256)} /> : '—',
            },
          ]}
        />
      </div>
      {evaluation.error_code && (
        <div className="mt-4">
          <Notice tone="warning" title={t.evaluation.errorCode}>
            <code>{evaluation.error_code}</code> {t.errors[evaluation.error_code as keyof typeof t.errors] ?? ''}
          </Notice>
        </div>
      )}
      <div className="mt-6">
        <MetricTable metrics={evaluation.metrics} testId="evaluation-metrics" />
      </div>
      <div className="mt-6">
        <p className="label-caps mb-2">{t.evaluation.limitations}</p>
        <LimitationsList items={evaluation.limitations} />
      </div>
    </Card>
  );
}

function Comparison({ items }: { items: Evaluation[] }) {
  const { t } = useI18n();
  const f = useFormat();
  const names = [...new Set(items.flatMap((item) => Object.keys(item.metrics)))];
  return (
    <Card className="card-pad" testId="evaluation-compare">
      <CardHeader title={t.evaluation.compareTitle} icon={<GitCompare className="size-5" />} />
      <div className="scroll-x mt-4">
        <table className="w-full min-w-[560px] text-left text-sm">
          <thead>
            <tr className="border-b border-line text-xs text-muted">
              <th scope="col" className="px-2 py-2 font-medium">{t.evaluation.metricName}</th>
              {items.map((item) => (
                <th key={item.id} scope="col" className="px-2 py-2 font-medium">
                  <code className="font-mono">{item.id.slice(0, 8)}</code>
                  <span className="block font-normal">
                    {shortHash(item.code_commit, 8)} · {item.source}
                  </span>
                  {item.dataset && <span className="block font-normal text-ink">{evaluationTitle(item, t, f.int)}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-line/70">
              <th scope="row" className="px-2 py-2 font-medium">{t.evaluation.colSamples}</th>
              {items.map((item) => (
                <td key={item.id} className="num px-2 py-2">{f.int(item.sample_count)}</td>
              ))}
            </tr>
            {names.map((name) => (
              <tr key={name} className="border-b border-line/70 last:border-0">
                <th scope="row" className="px-2 py-2 font-medium">{metricLabel(name, t)}</th>
                {items.map((item) => {
                  const metric = item.metrics[name];
                  return (
                    <td key={item.id} className="num px-2 py-2">
                      {!metric ? (
                        '—'
                      ) : metric.status === 'measured' && metric.value !== null ? (
                        <>
                          {f.num(metric.value, 3)} <span className="text-muted">{metric.unit}</span>
                        </>
                      ) : (
                        <span className="text-muted italic">{t.states.metric[metric.status]}</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export function EvaluationPage() {
  const { t } = useI18n();
  const f = useFormat();
  const navigate = useNavigate();
  const { evaluationId } = useParams();
  const { session } = useApp();
  const list = useResource(session ? 'evaluations' : null, (signal) => api.listEvaluations(signal));
  const detail = useResource(session && evaluationId ? `evaluation:${evaluationId}` : null, (signal) =>
    api.getEvaluation(evaluationId ?? '', signal),
  );
  const [seedsText, setSeedsText] = useState('1, 2, 3');
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<ApiError | null>(null);
  const [compare, setCompare] = useState<string[]>([]);

  const items = list.data?.items ?? [];
  const seeds = parseSeeds(seedsText);
  const anyRunning = items.some(isRunning) || (detail.data ? isRunning(detail.data) : false);
  useInterval(
    () => {
      list.reload();
      if (detail.data && isRunning(detail.data)) detail.reload();
    },
    anyRunning ? 3_000 : null,
  );
  useResync(() => {
    list.reload();
    detail.reload();
  });

  const runSuite = () => {
    if (!seeds) return;
    setSubmitting(true);
    setSubmitError(null);
    api.createEvaluation(seeds).then(
      (created) => {
        setSubmitting(false);
        list.setData((previous) => (previous ? { items: [created, ...previous.items.filter((item) => item.id !== created.id)] } : previous));
        void navigate(`/evaluation/${created.id}`);
      },
      (error: unknown) => {
        setSubmitting(false);
        setSubmitError(toApiError(error));
      },
    );
  };

  const toggleCompare = (id: string) =>
    setCompare((previous) =>
      previous.includes(id) ? previous.filter((item) => item !== id) : previous.length >= MAX_COMPARE ? previous : [...previous, id],
    );
  const compared = items.filter((item) => compare.includes(item.id));
  const mlItems = items.filter((item) => ML_KINDS.has(item.kind));
  const suiteRunning = items.some((item) => item.kind === 'synthetic_scenario_suite' && isRunning(item));

  return (
    <div data-testid="page-evaluation">
      <PageHeader title={t.evaluation.title} subtitle={t.evaluation.subtitle} />
      <div className="grid gap-6 lg:grid-cols-12">
        <div className="space-y-6 lg:col-span-8">
          <Card className="card-pad" testId="evaluation-list">
            <CardHeader title={t.evaluation.listTitle} icon={<ClipboardCheck className="size-5" />} subtitle={t.evaluation.compareHint} />
            <div className="mt-4">
              {list.error && !list.data ? (
                <ErrorNotice error={list.error} onRetry={list.reload} testId="evaluations-error" />
              ) : !list.data ? (
                session ? <LoadingBlock /> : <SessionPending />
              ) : items.length === 0 ? (
                <EmptyState title={t.evaluation.listEmpty} testId="evaluations-empty" />
              ) : (
                <div className="scroll-x rounded-control border border-line">
                  <table className="w-full min-w-[700px] text-left text-sm">
                    <thead>
                      <tr className="border-b border-line text-xs text-muted">
                        <th scope="col" className="w-10 px-3 py-2 font-medium">
                          <span className="sr-only">{t.evaluation.compare}</span>
                        </th>
                        <th scope="col" className="px-3 py-2 font-medium">{t.evaluation.colKind}</th>
                        <th scope="col" className="px-3 py-2 font-medium">{t.evaluation.colDataset}</th>
                        <th scope="col" className="px-3 py-2 font-medium">{t.evaluation.colState}</th>
                        <th scope="col" className="px-3 py-2 font-medium">{t.evaluation.colSource}</th>
                        <th scope="col" className="px-3 py-2 font-medium">{t.evaluation.colCreated}</th>
                        <th scope="col" className="px-3 py-2 font-medium">{t.evaluation.colCommit}</th>
                        <th scope="col" className="px-3 py-2 font-medium">{t.evaluation.colModel}</th>
                        <th scope="col" className="px-3 py-2 text-right font-medium">{t.evaluation.colSamples}</th>
                        <th scope="col" className="px-3 py-2 text-right font-medium">{t.evaluation.colMetrics}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((item) => (
                        <tr
                          key={item.id}
                          className={`border-b border-line/70 last:border-0 ${item.id === evaluationId ? 'bg-accent/5' : ''}`}
                          data-testid="evaluation-row"
                          data-evaluation-state={item.state}
                        >
                          <td className="px-3 py-2">
                            <input
                              type="checkbox"
                              checked={compare.includes(item.id)}
                              onChange={() => toggleCompare(item.id)}
                              disabled={!compare.includes(item.id) && compare.length >= MAX_COMPARE}
                              aria-label={t.evaluation.selectRow(item.id.slice(0, 8))}
                            />
                          </td>
                          <td className="px-3 py-2">
                            <Link to={`/evaluation/${item.id}`} className="font-medium text-accent-strong hover:underline" data-testid="evaluation-open">
                              {t.evaluation.kinds[item.kind]}
                            </Link>
                          </td>
                          <td className="min-w-[180px] px-3 py-2 text-xs" data-testid="evaluation-dataset">
                            {evaluationTitle(item, t, f.int) ?? '—'}
                          </td>
                          <td className="px-3 py-2">
                            <Pill tone={EVALUATION_TONE[item.state]} dot>
                              {t.states.evaluation[item.state]}
                            </Pill>
                          </td>
                          <td className="px-3 py-2">
                            <code className="font-mono text-xs">{item.source}</code>
                          </td>
                          <td className="num px-3 py-2 whitespace-nowrap text-muted">{f.dateTime(item.created_utc)}</td>
                          <td className="px-3 py-2">
                            <code className="font-mono text-xs">{shortHash(item.code_commit, 8)}</code>
                          </td>
                          <td className="px-3 py-2">
                            <code className="font-mono text-xs">{shortHash(item.model_sha256, 8)}</code>
                          </td>
                          <td className="num px-3 py-2 text-right">{f.int(item.sample_count)}</td>
                          <td className="num px-3 py-2 text-right">{measuredCount(item)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </Card>

          {compared.length >= 2 && <Comparison items={compared} />}

          {evaluationId &&
            (detail.data ? (
              <EvaluationDetail evaluation={detail.data} />
            ) : detail.error ? (
              <ErrorNotice error={detail.error} onRetry={detail.reload} testId="evaluation-error" />
            ) : (
              <LoadingBlock />
            ))}
        </div>

        <div className="space-y-6 lg:col-span-4">
          <Card className="card-pad" testId="suite-card">
            <CardHeader title={t.evaluation.suiteTitle} icon={<FlaskConical className="size-5" />} />
            <p className="mt-2 text-sm text-muted">{t.evaluation.suiteBody}</p>
            <label className="mt-4 block text-sm">
              <span className="mb-1 block font-medium">{t.evaluation.seeds}</span>
              <input
                type="text"
                inputMode="numeric"
                value={seedsText}
                onChange={(event) => setSeedsText(event.target.value)}
                className="h-10 w-full rounded-control border border-line bg-surface px-3 text-sm"
                aria-invalid={seeds === null}
                aria-describedby="suite-seeds-hint"
                data-testid="suite-seeds"
              />
              {seeds === null && (
                <span id="suite-seeds-hint" className="mt-1 block text-xs text-danger">
                  {t.evaluation.seedsInvalid}
                </span>
              )}
            </label>
            <Button
              variant="primary"
              className="mt-4"
              icon={<PlayCircle className="size-4" aria-hidden />}
              onClick={runSuite}
              busy={submitting}
              disabled={seeds === null || !session || suiteRunning}
              title={suiteRunning ? t.errors.evaluation_busy : !session ? t.errors.service_offline : undefined}
              data-testid="suite-run"
            >
              {suiteRunning ? t.evaluation.suiteRunning : t.evaluation.runSuite}
            </Button>
            {suiteRunning && <p className="mt-2 text-xs text-muted">{t.errors.evaluation_busy}</p>}
            {submitError && (
              <div className="mt-3">
                <ErrorNotice error={submitError} compact testId="suite-error" />
              </div>
            )}
          </Card>

          <Card className="card-pad" testId="ml-accuracy">
            <CardHeader title={t.evaluation.mlTitle} icon={<Microscope className="size-5" />} />
            <p className="mt-2 text-sm text-muted">{t.evaluation.mlBody}</p>
            <div className="mt-4 space-y-4">
              {!list.data ? null : mlItems.length === 0 ? (
                <Notice tone="neutral" testId="ml-none">
                  {t.evaluation.mlNone}
                </Notice>
              ) : (
                mlItems.map((item) => (
                  <div key={item.id} className="rounded-control border border-line p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <Link to={`/evaluation/${item.id}`} className="text-sm font-semibold hover:text-accent">
                        {t.evaluation.kinds[item.kind]}
                      </Link>
                      <Pill tone={EVALUATION_TONE[item.state]} dot>
                        {t.states.evaluation[item.state]}
                      </Pill>
                    </div>
                    {item.dataset && (
                      <p className="mt-1 text-sm font-medium" data-testid="ml-dataset" data-dataset-id={item.dataset.id}>
                        {evaluationTitle(item, t, f.int)}
                      </p>
                    )}
                    <p className="num mt-1 text-xs text-muted">
                      {item.source} · {t.evaluation.colSamples}: {f.int(item.sample_count)}
                    </p>
                    <ul className="mt-2 space-y-1.5">
                      {Object.entries(item.metrics).map(([name, metric]) => (
                        <li key={name} className="text-sm" data-testid="ml-metric" data-status={metric.status}>
                          <span className="font-medium">{metricLabel(name, t)}:</span>{' '}
                          {metric.status === 'measured' && metric.value !== null ? (
                            <span className="num">
                              {f.num(metric.value, 3)} {metric.unit} (n={metric.sample_count})
                            </span>
                          ) : (
                            <span className="text-muted">
                              {t.states.metric[metric.status]} — {metric.note}
                            </span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
