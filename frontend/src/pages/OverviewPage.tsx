import { ArrowRight, Film, History, Route, ShieldAlert } from 'lucide-react';
import { Link } from 'react-router';

import { api } from '../api/client';
import { ProvenanceLegend, SessionPending, SourcePill } from '../components/domain';
import { Card, CardHeader, PageHeader } from '../components/ui/Card';
import { EmptyState, ErrorNotice, LoadingBlock } from '../components/ui/Feedback';
import { Pill } from '../components/ui/Status';
import { useEvents, useResync } from '../hooks/useEvents';
import { useResource } from '../hooks/useResource';
import { useI18n } from '../i18n';
import { outcomeText, shortHash, unavailableText, useFormat } from '../lib/format';
import { JOB_TONE, RUN_TONE } from '../lib/tones';
import { useApp } from '../state/app';

function CapabilityCards() {
  const { t } = useI18n();
  const { capabilities, capabilitiesError, capabilitiesLoading, refreshCapabilities } = useApp();
  if (!capabilities) {
    if (capabilitiesError) return <ErrorNotice error={capabilitiesError} onRetry={refreshCapabilities} testId="capabilities-error" />;
    return capabilitiesLoading ? <LoadingBlock /> : null;
  }
  const video = capabilities.video_analysis;
  const queue = capabilities.queue;
  return (
    <div className="grid gap-4 md:grid-cols-3" data-testid="capability-summary">
      <div className="rounded-control border border-line p-4">
        <p className="label-caps">{t.overview.videoCap}</p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Pill tone={video.available ? 'success' : 'warning'} dot testId="overview-video-cap">
            {video.available ? t.system.modelReady : t.common.featureAbsent}
          </Pill>
          {video.model && <SourcePill source="learned_model" extra={`${video.model.model_id} · ${(video.device ?? '').toUpperCase()}`} />}
        </div>
        {!video.available && <p className="mt-2 text-sm text-muted">{unavailableText(t, video.reason)}</p>}
        {video.model?.sha256 && (
          <p className="mt-2 text-xs text-muted">
            SHA256 <code className="font-mono">{shortHash(video.model.sha256)}</code>
          </p>
        )}
      </div>
      <div className="rounded-control border border-line p-4">
        <p className="label-caps">{t.overview.simCap}</p>
        <ul className="mt-2 space-y-2">
          {capabilities.simulation.simulators.map((sim) => (
            <li key={sim.id} className="text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{sim.name}</span>
                <Pill tone={sim.available ? 'success' : 'neutral'} dot>
                  {sim.available ? t.system.modelReady : t.common.featureAbsent}
                </Pill>
              </div>
              {!sim.available && <p className="text-muted">{unavailableText(t, sim.reason)}</p>}
              {sim.available && (
                <p className="mt-1 flex flex-wrap gap-1">
                  {sim.perception_modes
                    .filter((mode) => mode.available)
                    .map((mode) => (
                      <SourcePill key={mode.id} source={mode.id} />
                    ))}
                </p>
              )}
            </li>
          ))}
        </ul>
      </div>
      <div className="rounded-control border border-line p-4">
        <p className="label-caps">{t.overview.queueCap}</p>
        {queue.video_queued === null || queue.video_running === null ? (
          <p className="mt-2 text-sm text-muted">
            {t.video.queueUnknown} {queue.reason ?? ''}
          </p>
        ) : (
          <p className="num mt-2 text-sm font-medium">
            {t.overview.queueLine(queue.video_queued, queue.video_running, queue.capacity_total)}
          </p>
        )}
        {queue.sim_active !== null && (
          <p className="num mt-1 text-sm text-muted">{t.overview.simActiveLine(queue.sim_active, queue.sim_capacity)}</p>
        )}
      </div>
    </div>
  );
}

function RecentJobs() {
  const { t } = useI18n();
  const f = useFormat();
  const { session } = useApp();
  const jobs = useResource(session ? 'recent-jobs' : null, (signal) => api.listJobs(5, null, signal));
  useEvents({ job_state: jobs.reload });
  useResync(jobs.reload);
  return (
    <Card className="card-pad" testId="recent-jobs">
      <CardHeader
        title={t.overview.recentJobs}
        icon={<Film className="size-5" />}
        actions={
          <Link to="/video" className="text-sm font-medium text-accent hover:underline">
            {t.overview.viewAll}
          </Link>
        }
      />
      <div className="mt-4">
        {jobs.error && !jobs.data ? (
          <ErrorNotice error={jobs.error} onRetry={jobs.reload} compact />
        ) : !jobs.data ? (
          session ? <LoadingBlock /> : <SessionPending />
        ) : jobs.data.items.length === 0 ? (
          <EmptyState title={t.overview.noJobs} />
        ) : (
          <ul className="divide-y divide-line">
            {jobs.data.items.map((job) => (
              <li key={job.id}>
                <Link to={`/video/jobs/${job.id}`} className="flex items-center justify-between gap-3 py-2.5 text-sm hover:text-accent">
                  <span className="min-w-0 truncate">
                    <code className="font-mono text-xs text-muted">{job.id.slice(0, 8)}</code>{' '}
                    <span className="text-muted">· {f.dateTime(job.created_utc)}</span>
                  </span>
                  <Pill tone={JOB_TONE[job.state]} dot>
                    {t.states.job[job.state]}
                  </Pill>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

function RecentRuns() {
  const { t } = useI18n();
  const f = useFormat();
  const { session } = useApp();
  const runs = useResource(session ? 'recent-runs' : null, (signal) => api.listRuns(5, signal));
  useEvents({ run_state: runs.reload });
  useResync(runs.reload);
  return (
    <Card className="card-pad" testId="recent-runs">
      <CardHeader
        title={t.overview.recentRuns}
        icon={<History className="size-5" />}
        actions={
          <Link to="/simulation" className="text-sm font-medium text-accent hover:underline">
            {t.overview.viewAll}
          </Link>
        }
      />
      <div className="mt-4">
        {runs.error && !runs.data ? (
          <ErrorNotice error={runs.error} onRetry={runs.reload} compact />
        ) : !runs.data ? (
          session ? <LoadingBlock /> : <SessionPending />
        ) : runs.data.items.length === 0 ? (
          <EmptyState title={t.overview.noRuns} />
        ) : (
          <ul className="divide-y divide-line">
            {runs.data.items.map((run) => (
              <li key={run.id}>
                <Link
                  to={`/simulation/runs/${run.id}`}
                  className="flex items-center justify-between gap-3 py-2.5 text-sm hover:text-accent"
                >
                  <span className="min-w-0 truncate">
                    {t.sim.scenarios[run.scenario] ?? run.scenario}
                    <span className="text-muted"> · {f.dateTime(run.created_utc)}</span>
                    {run.outcome && <span className="text-muted"> · {outcomeText(t, run.outcome)}</span>}
                  </span>
                  <Pill tone={RUN_TONE[run.state]} dot>
                    {t.states.run[run.state]}
                  </Pill>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

export function OverviewPage() {
  const { t } = useI18n();
  return (
    <div data-testid="page-overview">
      <PageHeader title={t.overview.title} subtitle={t.overview.intro} />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="card-pad flex flex-col">
          <CardHeader title={t.overview.videoTitle} icon={<Film className="size-5 text-accent" />} />
          <p className="mt-2 flex-1 text-sm text-ink/85">{t.overview.videoBody}</p>
          <Link
            to="/video"
            className="mt-4 inline-flex h-10 w-fit items-center gap-2 rounded-control bg-accent px-4 text-sm font-medium text-white hover:bg-accent-strong"
            data-testid="overview-video-cta"
          >
            {t.overview.videoCta}
            <ArrowRight className="size-4" aria-hidden />
          </Link>
        </Card>
        <Card className="card-pad flex flex-col">
          <CardHeader title={t.overview.simTitle} icon={<Route className="size-5 text-cyan" />} />
          <p className="mt-2 flex-1 text-sm text-ink/85">{t.overview.simBody}</p>
          <Link
            to="/simulation"
            className="mt-4 inline-flex h-10 w-fit items-center gap-2 rounded-control border border-line bg-surface px-4 text-sm font-medium hover:bg-canvas"
            data-testid="overview-sim-cta"
          >
            {t.overview.simCta}
            <ArrowRight className="size-4" aria-hidden />
          </Link>
        </Card>
      </div>

      <Card className="card-pad mt-6">
        <CardHeader title={t.overview.capabilities} />
        <div className="mt-4">
          <CapabilityCards />
        </div>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <RecentJobs />
        <RecentRuns />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card className="card-pad lg:col-span-2">
          <CardHeader title={t.provenance.title} />
          <div className="mt-4">
            <ProvenanceLegend />
          </div>
        </Card>
        <Card className="card-pad border-warning/30">
          <CardHeader title={t.overview.safetyTitle} icon={<ShieldAlert className="size-5 text-warning" />} />
          <p className="mt-2 text-sm text-ink/85" data-testid="safety-note">
            {t.overview.safetyBody}
          </p>
        </Card>
      </div>
    </div>
  );
}
