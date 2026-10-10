import { Ban, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Link } from 'react-router';

import type { ApiError } from '../../api/client';
import type { Job } from '../../api/types';
import { isTerminalJob } from '../../api/types';
import { Button } from '../../components/ui/Button';
import { EmptyState, ErrorNotice, LoadingBlock } from '../../components/ui/Feedback';
import { Pill, ProgressBar } from '../../components/ui/Status';
import { useNow } from '../../hooks/useTimers';
import { useI18n } from '../../i18n';
import { elapsedSeconds, stageText, useFormat } from '../../lib/format';
import { JOB_TONE } from '../../lib/tones';

export function JobList({
  jobs,
  selectedId,
  loading,
  error,
  onReload,
  onCancel,
  onDelete,
  busy,
  hasMore,
  onLoadMore,
  loadingMore,
}: {
  jobs: Job[] | undefined;
  selectedId: string | undefined;
  loading: boolean;
  error: ApiError | null;
  onReload: () => void;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
  busy: Record<string, 'cancel' | 'delete' | undefined>;
  hasMore: boolean;
  onLoadMore: () => void;
  loadingMore: boolean;
}) {
  const { t } = useI18n();
  const f = useFormat();
  const [confirming, setConfirming] = useState<string | null>(null);
  const anyActive = jobs?.some((job) => !isTerminalJob(job.state)) ?? false;
  const now = useNow(1000, anyActive);

  if (!jobs) {
    if (error) return <ErrorNotice error={error} onRetry={onReload} compact testId="jobs-error" />;
    return loading ? <LoadingBlock /> : null;
  }
  if (jobs.length === 0) return <EmptyState title={t.video.jobsEmpty} testId="jobs-empty" />;

  return (
    <div>
      {error && (
        <div className="mb-3">
          <ErrorNotice error={error} onRetry={onReload} compact testId="jobs-error" />
        </div>
      )}
      <ul className="divide-y divide-line" data-testid="job-list">
        {jobs.map((job) => {
          const active = !isTerminalJob(job.state);
          const elapsed = elapsedSeconds(job.started_utc ?? job.created_utc, job.finished_utc, now);
          const selected = job.id === selectedId;
          const cancellable = job.state === 'queued' || job.state === 'validating' || job.state === 'running';
          return (
            <li
              key={job.id}
              className={`py-3 ${selected ? '-mx-2 rounded-control bg-accent/5 px-2' : ''}`}
              data-testid="job-item"
              data-job-id={job.id}
              data-job-state={job.state}
            >
              <div className="flex items-center justify-between gap-2">
                <Link
                  to={`/video/jobs/${job.id}`}
                  className="min-w-0 truncate text-sm font-medium hover:text-accent"
                  aria-current={selected ? 'true' : undefined}
                  data-testid="job-open"
                >
                  <code className="font-mono text-xs">{job.id.slice(0, 8)}</code>
                  <span className="ml-2 text-muted">{f.time(job.created_utc)}</span>
                </Link>
                <Pill tone={JOB_TONE[job.state]} dot testId="job-state">
                  {t.states.job[job.state]}
                </Pill>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
                <span className="num">{f.bytes(job.input.bytes)}</span>
                <span className="uppercase">{job.input.ext}</span>
                {elapsed !== null && (
                  <span className="num">
                    {t.common.elapsed}: {f.clock(elapsed)}
                  </span>
                )}
                {job.error_code && <code className="text-danger">{job.error_code}</code>}
              </div>
              {active && (
                <div className="mt-2">
                  <ProgressBar
                    value={job.state === 'queued' ? null : job.progress}
                    label={`${t.states.job[job.state]} ${job.id.slice(0, 8)}`}
                    tone={job.state === 'cancelling' ? 'warning' : 'info'}
                  />
                  <p className="num mt-1 text-xs text-muted">
                    {job.stage && `${t.video.stage}: ${stageText(t, job.stage)} · `}
                    {job.frames_total ? t.video.framesOf(job.frames_done, job.frames_total) : t.video.framesDone(job.frames_done)}
                  </p>
                </div>
              )}
              <div className="mt-2 flex flex-wrap gap-2">
                {cancellable && (
                  <Button
                    size="sm"
                    icon={<Ban className="size-3.5" aria-hidden />}
                    onClick={() => onCancel(job.id)}
                    busy={busy[job.id] === 'cancel'}
                    data-testid="job-cancel"
                  >
                    {t.video.cancel}
                  </Button>
                )}
                {confirming === job.id ? (
                  <>
                    <Button
                      size="sm"
                      variant="danger"
                      onClick={() => {
                        setConfirming(null);
                        onDelete(job.id);
                      }}
                      data-testid="job-delete-confirm"
                    >
                      {t.video.confirmDelete}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setConfirming(null)}>
                      {t.common.cancel}
                    </Button>
                  </>
                ) : (
                  <Button
                    size="sm"
                    variant="ghost"
                    icon={<Trash2 className="size-3.5" aria-hidden />}
                    onClick={() => setConfirming(job.id)}
                    busy={busy[job.id] === 'delete'}
                    data-testid="job-delete"
                  >
                    {t.video.delete}
                  </Button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      {hasMore && (
        <Button size="sm" className="mt-3" onClick={onLoadMore} busy={loadingMore} data-testid="jobs-load-more">
          {t.common.loadMore}
        </Button>
      )}
    </div>
  );
}
