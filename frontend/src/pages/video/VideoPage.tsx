import { ListVideo, Plus, UploadCloud } from 'lucide-react';
import type { ReactNode } from 'react';
import { useState } from 'react';
import { useNavigate, useParams } from 'react-router';

import type { ApiError } from '../../api/client';
import { api, toApiError } from '../../api/client';
import type { Job } from '../../api/types';
import { isTerminalJob } from '../../api/types';
import { SessionPending } from '../../components/domain';
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, PageHeader } from '../../components/ui/Card';
import { ErrorNotice, LoadingBlock, Notice } from '../../components/ui/Feedback';
import { useEvents, useEventStatus, useResync } from '../../hooks/useEvents';
import { useResource } from '../../hooks/useResource';
import { useInterval } from '../../hooks/useTimers';
import { useI18n } from '../../i18n';
import { unavailableText } from '../../lib/format';
import { useApp } from '../../state/app';
import { JobList } from './JobList';
import { JobStatusPanel } from './JobStatusPanel';
import { ResultWorkspace } from './ResultWorkspace';
import { UploadPanel } from './UploadPanel';

const PAGE_SIZE = 20;

function mergeJob(list: Job[], job: Job): Job[] {
  const index = list.findIndex((item) => item.id === job.id);
  if (index === -1) return [job, ...list];
  const next = [...list];
  next[index] = job;
  return next;
}

export function VideoPage() {
  const { t } = useI18n();
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { session, capabilities } = useApp();
  const streamStatus = useEventStatus();

  const jobs = useResource(session ? 'jobs' : null, (signal) => api.listJobs(PAGE_SIZE, null, signal));
  // Pages after the first one ("show more"); the first page is re-fetched on events, older pages are appended once.
  const [older, setOlder] = useState<{ items: Job[]; cursor: string | null; loaded: boolean; loading: boolean }>({
    items: [],
    cursor: null,
    loaded: false,
    loading: false,
  });
  const selected = useResource(session && jobId ? `job:${jobId}` : null, (signal) => api.getJob(jobId ?? '', signal));
  const job = selected.data;
  const results = useResource(job?.state === 'succeeded' ? `results:${job.id}` : null, (signal) =>
    api.getJobResults(job?.id ?? '', signal),
  );
  const [busy, setBusy] = useState<Record<string, 'cancel' | 'delete' | undefined>>({});
  const [actionError, setActionError] = useState<ApiError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const firstPage = jobs.data?.items;
  const nextCursor = older.loaded ? older.cursor : (jobs.data?.next_cursor ?? null);
  const allJobs = firstPage ? [...firstPage, ...older.items.filter((item) => !firstPage.some((j) => j.id === item.id))] : undefined;

  const patchJob = (id: string, patch: Partial<Job>) => {
    jobs.setData((previous) =>
      previous ? { ...previous, items: previous.items.map((item) => (item.id === id ? { ...item, ...patch } : item)) } : previous,
    );
    if (id === jobId) selected.setData((previous) => (previous ? { ...previous, ...patch } : previous));
  };

  useEvents({
    job_state: (payload) => {
      jobs.reload();
      if (payload.job_id === jobId) selected.reload();
    },
    job_progress: (payload) =>
      patchJob(payload.job_id, {
        progress: payload.progress,
        stage: payload.stage,
        frames_done: payload.frames_done,
        frames_total: payload.frames_total,
      }),
  });
  useResync(() => {
    jobs.reload();
    selected.reload();
    results.reload();
  });

  // Polling fallback: fast while the stream is down, slow safety net while it is up.
  const anyActive = (allJobs?.some((item) => !isTerminalJob(item.state)) ?? false) || (job ? !isTerminalJob(job.state) : false);
  useInterval(
    () => {
      jobs.reload();
      if (job && !isTerminalJob(job.state)) selected.reload();
    },
    anyActive ? (streamStatus === 'online' ? 15_000 : 3_000) : null,
  );

  const onUploaded = (newJob: Job) => {
    jobs.setData((previous) => (previous ? { ...previous, items: mergeJob(previous.items, newJob) } : previous));
    void navigate(`/video/jobs/${newJob.id}`);
  };

  const cancelJob = (id: string) => {
    setBusy((previous) => ({ ...previous, [id]: 'cancel' }));
    setActionError(null);
    api.cancelJob(id).then(
      (updated) => {
        patchJob(id, updated);
        setBusy((previous) => ({ ...previous, [id]: undefined }));
      },
      (error: unknown) => {
        setActionError(toApiError(error));
        setBusy((previous) => ({ ...previous, [id]: undefined }));
      },
    );
  };

  const deleteJob = (id: string) => {
    setBusy((previous) => ({ ...previous, [id]: 'delete' }));
    setActionError(null);
    api.deleteJob(id).then(
      (result) => {
        setBusy((previous) => ({ ...previous, [id]: undefined }));
        if (result.status === 'deleted') {
          jobs.setData((previous) => (previous ? { ...previous, items: previous.items.filter((item) => item.id !== id) } : previous));
          setOlder((previous) => ({ ...previous, items: previous.items.filter((item) => item.id !== id) }));
          setNotice(null);
          if (id === jobId) void navigate('/video');
        } else {
          setNotice(t.video.deletionScheduled);
          jobs.reload();
        }
      },
      (error: unknown) => {
        setActionError(toApiError(error));
        setBusy((previous) => ({ ...previous, [id]: undefined }));
      },
    );
  };

  const loadMore = () => {
    if (!nextCursor) return;
    setOlder((previous) => ({ ...previous, loading: true }));
    api.listJobs(PAGE_SIZE, nextCursor).then(
      (page) =>
        setOlder((previous) => ({
          items: [...previous.items, ...page.items],
          cursor: page.next_cursor,
          loaded: true,
          loading: false,
        })),
      (error: unknown) => {
        setActionError(toApiError(error));
        setOlder((previous) => ({ ...previous, loading: false }));
      },
    );
  };

  const videoCap = capabilities?.video_analysis;
  const jobsCard = (
    <Card className="card-pad" testId="jobs-card">
      <CardHeader
        title={t.video.jobsTitle}
        icon={<ListVideo className="size-5" />}
        actions={
          jobId ? (
            <Button size="sm" icon={<Plus className="size-4" aria-hidden />} onClick={() => void navigate('/video')} data-testid="new-upload">
              {t.video.newUpload}
            </Button>
          ) : undefined
        }
      />
      <div className="mt-4 space-y-3">
        {actionError && <ErrorNotice error={actionError} compact testId="job-action-error" />}
        {notice && <Notice tone="info">{notice}</Notice>}
        {!session ? (
          <SessionPending />
        ) : (
        <JobList
          jobs={allJobs}
          selectedId={jobId}
          loading={jobs.loading}
          error={jobs.error}
          onReload={jobs.reload}
          onCancel={cancelJob}
          onDelete={deleteJob}
          busy={busy}
          hasMore={Boolean(nextCursor)}
          onLoadMore={loadMore}
          loadingMore={older.loading}
        />
        )}
      </div>
    </Card>
  );

  let main: ReactNode;
  if (!jobId) {
    main = (
      <Card className="card-pad" testId="upload-card">
        <CardHeader title={t.video.newUpload} icon={<UploadCloud className="size-5" />} />
        <div className="mt-4 space-y-4">
          {videoCap && !videoCap.available && (
            <Notice tone="warning" title={t.video.unavailableTitle} testId="video-unavailable">
              {unavailableText(t, videoCap.reason)} {t.errors.model_not_ready}
            </Notice>
          )}
          <UploadPanel onUploaded={onUploaded} />
        </div>
      </Card>
    );
  } else if (!job) {
    main = selected.error ? (
      <ErrorNotice error={selected.error} onRetry={selected.reload} testId="job-error" />
    ) : (
      <LoadingBlock />
    );
  } else if (job.state !== 'succeeded') {
    main = (
      <JobStatusPanel job={job} onCancel={() => cancelJob(job.id)} cancelBusy={busy[job.id] === 'cancel'} onNewUpload={() => void navigate('/video')} />
    );
  }

  return (
    <div data-testid="page-video">
      <PageHeader title={t.video.title} subtitle={t.video.subtitle} />
      {job && job.state === 'succeeded' && jobId ? (
        results.data ? (
          <ResultWorkspace key={job.id} job={job} results={results.data} side={jobsCard} />
        ) : (
          <div className="grid gap-6 lg:grid-cols-12">
            <div className="lg:col-span-8">
              {results.error ? (
                <ErrorNotice error={results.error} onRetry={results.reload} testId="results-error" />
              ) : (
                <LoadingBlock label={t.video.resultsLoading} />
              )}
            </div>
            <div className="lg:col-span-4">{jobsCard}</div>
          </div>
        )
      ) : (
        <div className="grid gap-6 lg:grid-cols-12">
          <div className="space-y-6 lg:col-span-8">{main}</div>
          <div className="space-y-6 lg:col-span-4">
            {jobId && (
              <Card className="card-pad">
                <CardHeader title={t.video.newUpload} icon={<UploadCloud className="size-5" />} />
                <div className="mt-3">
                  <UploadPanel onUploaded={onUploaded} compact />
                </div>
              </Card>
            )}
            {jobsCard}
          </div>
        </div>
      )}
    </div>
  );
}
