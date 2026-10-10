import { Ban, CircleAlert, Clock3, Hourglass } from 'lucide-react';

import type { Job } from '../../api/types';
import { isTerminalJob } from '../../api/types';
import { Button } from '../../components/ui/Button';
import { Pill, ProgressBar, Spinner } from '../../components/ui/Status';
import { useNow } from '../../hooks/useTimers';
import { useI18n } from '../../i18n';
import { elapsedSeconds, stageText, useFormat } from '../../lib/format';
import { JOB_TONE } from '../../lib/tones';
import { useApp } from '../../state/app';

/** Dark media panel shown in place of the video while a job is queued/running or after it failed/was cancelled. */
export function JobStatusPanel({
  job,
  onCancel,
  cancelBusy,
  onNewUpload,
}: {
  job: Job;
  onCancel: () => void;
  cancelBusy: boolean;
  onNewUpload: () => void;
}) {
  const { t } = useI18n();
  const f = useFormat();
  const { capabilities } = useApp();
  const active = !isTerminalJob(job.state);
  const now = useNow(1000, active);
  const elapsed = elapsedSeconds(job.started_utc ?? job.created_utc, job.finished_utc, now);
  const queue = capabilities?.queue;
  const cancellable = job.state === 'queued' || job.state === 'validating' || job.state === 'running';

  return (
    <div
      data-on-dark
      className="flex w-full flex-col justify-center rounded-card bg-night p-6 text-white sm:aspect-video sm:p-10"
      data-testid="job-status-panel"
      data-job-state={job.state}
    >
      <div className="mx-auto w-full max-w-xl" aria-live="polite">
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone={JOB_TONE[job.state]} dot onDark testId="job-status-state">
            {t.states.job[job.state]}
          </Pill>
          <code className="font-mono text-xs text-white/60">{job.id}</code>
        </div>

        {active && (
          <>
            <h2 className="mt-4 flex items-center gap-2 text-xl font-semibold text-white">
              {job.state === 'queued' ? <Hourglass className="size-5" aria-hidden /> : <Spinner className="size-5" />}
              {job.state === 'queued' ? t.video.waitingQueue : job.state === 'cancelling' ? t.video.cancellingBody : t.video.progressTitle}
            </h2>
            <ProgressBar
              className="mt-4 bg-white/15"
              value={job.state === 'queued' ? null : job.progress}
              label={t.video.progressTitle}
              tone={job.state === 'cancelling' ? 'warning' : 'info'}
              testId="job-progress"
            />
            <dl className="num mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              <div>
                <dt className="text-white/60">{t.common.progress}</dt>
                <dd className="font-semibold" data-testid="job-progress-pct">
                  {job.state === 'queued' ? '—' : f.pct(job.progress, 0)}
                </dd>
              </div>
              <div>
                <dt className="text-white/60">{t.video.stage}</dt>
                <dd className="font-semibold">{stageText(t, job.stage ?? (job.state === 'queued' ? 'queued' : null))}</dd>
              </div>
              <div>
                <dt className="text-white/60">{t.video.colFrames}</dt>
                <dd className="font-semibold">
                  {job.frames_total ? t.video.framesOf(job.frames_done, job.frames_total) : t.video.framesDone(job.frames_done)}
                </dd>
              </div>
              <div>
                <dt className="flex items-center gap-1 text-white/60">
                  <Clock3 className="size-3.5" aria-hidden />
                  {t.common.elapsed}
                </dt>
                <dd className="font-semibold" data-testid="job-elapsed">
                  {f.clock(elapsed)}
                </dd>
              </div>
            </dl>
            {queue && (
              <p className="mt-3 text-xs text-white/60">
                {queue.video_queued !== null && queue.video_running !== null
                  ? t.video.queueLine(queue.video_queued, queue.video_running)
                  : t.video.queueUnknown}
              </p>
            )}
            {cancellable && (
              <Button
                variant="dark"
                className="mt-5"
                icon={<Ban className="size-4" aria-hidden />}
                onClick={onCancel}
                busy={cancelBusy}
                data-testid="job-status-cancel"
              >
                {t.video.cancel}
              </Button>
            )}
          </>
        )}

        {(job.state === 'failed' || job.state === 'interrupted' || job.state === 'cancelled') && (
          <>
            <h2 className="mt-4 flex items-center gap-2 text-xl font-semibold text-white">
              <CircleAlert className={`size-5 ${job.state === 'failed' ? 'text-red-300' : 'text-amber-300'}`} aria-hidden />
              {job.state === 'failed' ? t.video.failedTitle : t.states.job[job.state]}
            </h2>
            <p className="mt-2 text-sm text-white/85" data-testid="job-failure-text">
              {job.state === 'failed'
                ? (t.errors[job.error_code as keyof typeof t.errors] ?? t.errors.internal_error)
                : job.state === 'interrupted'
                  ? t.video.interruptedBody
                  : t.video.cancelledBody}
            </p>
            {job.error_code && (
              <p className="mt-2 text-xs text-white/60">
                <code>{job.error_code}</code>
                {job.error_message && (
                  <>
                    {' '}
                    · {t.common.serverMessage}: {job.error_message}
                  </>
                )}
              </p>
            )}
            <Button variant="dark" className="mt-5" onClick={onNewUpload} data-testid="job-new-upload">
              {t.video.newUpload}
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
