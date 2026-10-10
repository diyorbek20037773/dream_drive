import { CheckCircle2, FileVideo, UploadCloud, X } from 'lucide-react';
import type { DragEvent } from 'react';
import { useEffect, useRef, useState } from 'react';

import type { ApiError, UploadHandle } from '../../api/client';
import { toApiError, uploadVideo } from '../../api/client';
import type { Job, Limits } from '../../api/types';
import { Button } from '../../components/ui/Button';
import { ErrorNotice, Notice } from '../../components/ui/Feedback';
import { ProgressBar } from '../../components/ui/Status';
import { useNow } from '../../hooks/useTimers';
import { useI18n } from '../../i18n';
import { unavailableText, useFormat } from '../../lib/format';
import { useApp } from '../../state/app';

type UploadState =
  | { status: 'idle' }
  | { status: 'uploading'; file: File; loaded: number; total: number; startedAt: number }
  | { status: 'error'; file: File | null; error: ApiError | null; message: string | null }
  | { status: 'cancelled'; file: File }
  | { status: 'done'; job: Job };

/** Codes for which re-sending the same file can succeed. */
const RETRIABLE = new Set([
  'network_error',
  'service_offline',
  'rate_limited',
  'queue_full',
  'session_queue_full',
  'model_not_ready',
  'internal_error',
  'csrf_failed',
  'session_required',
  'inference_unavailable',
  'service_unavailable',
]);

function fileExtension(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot >= 0 ? name.slice(dot + 1).toLowerCase() : '';
}

function LimitsList({ limits }: { limits: Limits | null }) {
  const { t } = useI18n();
  const f = useFormat();
  if (!limits) return <p className="text-sm text-muted">{t.video.limitsUnknown}</p>;
  const exts = limits.accepted_extensions.map((ext) => ext.toUpperCase()).join(', ');
  const items = [
    t.video.limitFormats(exts),
    t.video.limitSize(f.mb(limits.max_upload_bytes)),
    t.video.limitDuration(limits.max_video_duration_s),
    t.video.limitResolution(limits.max_video_width_px, limits.max_video_height_px),
    t.video.limitFps(limits.analysis_fps),
    t.video.limitQueue(limits.max_queued_per_session),
    t.video.limitRetention(limits.artifact_retention_hours),
  ];
  return (
    <ul className="grid gap-x-6 gap-y-1 text-sm text-ink/85 sm:grid-cols-2" data-testid="upload-limits">
      {items.map((item) => (
        <li key={item} className="flex items-start gap-2">
          <span aria-hidden className="mt-2 size-1 shrink-0 rounded-full bg-muted" />
          {item}
        </li>
      ))}
    </ul>
  );
}

export function UploadPanel({ onUploaded, compact = false }: { onUploaded: (job: Job) => void; compact?: boolean }) {
  const { t } = useI18n();
  const f = useFormat();
  const { limits, capabilities, capabilitiesError, session } = useApp();
  const [state, setState] = useState<UploadState>({ status: 'idle' });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const handleRef = useRef<UploadHandle | null>(null);
  const now = useNow(1000, state.status === 'uploading');

  useEffect(() => () => handleRef.current?.abort(), []);

  const video = capabilities?.video_analysis;
  const disabledReason = !session
    ? t.errors.service_offline
    : !capabilities
      ? capabilitiesError
        ? t.errors.service_offline
        : t.common.loading
      : video && !video.available
        ? `${t.video.unavailableHint} ${unavailableText(t, video.reason)}`
        : null;
  const busy = state.status === 'uploading';

  const start = (file: File) => {
    if (limits) {
      const ext = fileExtension(file.name);
      const accepted = limits.accepted_extensions.map((value) => value.toLowerCase().replace(/^\./, ''));
      if (!accepted.includes(ext)) {
        setState({ status: 'error', file, error: null, message: t.video.clientBadExt(accepted.map((e) => e.toUpperCase()).join(', ')) });
        return;
      }
      if (file.size > limits.max_upload_bytes) {
        setState({ status: 'error', file, error: null, message: t.video.clientTooLarge(f.mb(limits.max_upload_bytes)) });
        return;
      }
    }
    if (file.size === 0) {
      setState({ status: 'error', file, error: null, message: t.errors.upload_empty });
      return;
    }
    const handle = uploadVideo(file, (loaded, total) =>
      setState((previous) => (previous.status === 'uploading' && previous.file === file ? { ...previous, loaded, total } : previous)),
    );
    handleRef.current = handle;
    setState({ status: 'uploading', file, loaded: 0, total: file.size, startedAt: Date.now() });
    handle.promise.then(
      (job) => {
        handleRef.current = null;
        setState({ status: 'done', job });
        onUploaded(job);
      },
      (error: unknown) => {
        handleRef.current = null;
        const apiError = toApiError(error);
        if (apiError.code === 'aborted') setState({ status: 'cancelled', file });
        else setState({ status: 'error', file, error: apiError, message: null });
      },
    );
  };

  const onFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file && !busy && !disabledReason) start(file);
    if (inputRef.current) inputRef.current.value = '';
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    onFiles(event.dataTransfer.files);
  };

  const pct = state.status === 'uploading' && state.total > 0 ? Math.round((state.loaded / state.total) * 100) : 0;
  const retryFile = state.status === 'cancelled' ? state.file : state.status === 'error' && state.error && RETRIABLE.has(state.error.code) ? state.file : null;

  return (
    <div data-testid="upload-panel">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!busy && !disabledReason) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        data-testid="upload-dropzone"
        aria-disabled={Boolean(disabledReason) || busy}
        className={`flex flex-col items-center justify-center rounded-card border-2 border-dashed text-center transition-colors ${
          compact ? 'gap-2 px-4 py-6' : 'gap-3 px-6 py-12 sm:py-16'
        } ${dragging ? 'border-accent bg-accent/5' : 'border-line bg-canvas/60'} ${disabledReason ? 'opacity-70' : ''}`}
      >
        <UploadCloud className={`${compact ? 'size-7' : 'size-10'} text-accent`} aria-hidden />
        <p className={`${compact ? 'text-sm' : 'text-base'} font-semibold`}>{dragging ? t.video.dropActive : t.video.dropTitle}</p>
        <p className="text-sm text-muted">{t.video.dropOr}</p>
        <Button
          variant="primary"
          icon={<FileVideo className="size-4" aria-hidden />}
          onClick={() => inputRef.current?.click()}
          disabled={Boolean(disabledReason) || busy}
          title={disabledReason ?? undefined}
          data-testid="upload-choose"
        >
          {t.video.choose}
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,video/mp4,video/quicktime"
          className="sr-only"
          tabIndex={-1}
          aria-hidden
          disabled={Boolean(disabledReason) || busy}
          onChange={(event) => onFiles(event.target.files)}
          data-testid="upload-input"
        />
        {disabledReason && (
          <p className="max-w-md text-sm text-warning" data-testid="upload-disabled-reason">
            {disabledReason}
          </p>
        )}
      </div>

      {!compact && (
        <div className="mt-4">
          <p className="label-caps mb-2">{t.video.limitsTitle}</p>
          <LimitsList limits={limits} />
        </div>
      )}

      <div className="mt-4 space-y-3" aria-live="polite">
        {state.status === 'uploading' && (
          <div className="rounded-control border border-line bg-surface p-3" data-testid="upload-progress">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="min-w-0 truncate font-medium">{state.file.name}</span>
              <Button size="sm" variant="ghost" icon={<X className="size-4" aria-hidden />} onClick={() => handleRef.current?.abort()} data-testid="upload-cancel">
                {t.video.uploadCancel}
              </Button>
            </div>
            <ProgressBar className="mt-2" value={state.total > 0 ? state.loaded / state.total : null} label={t.video.uploading} />
            <p className="num mt-1.5 text-xs text-muted">
              {t.video.uploadProgress(pct)} · {f.bytes(state.loaded)} / {f.bytes(state.total)} · {t.common.elapsed}{' '}
              {f.clock((now - state.startedAt) / 1000)}
            </p>
          </div>
        )}
        {state.status === 'done' && (
          <Notice tone="success" icon={<CheckCircle2 className="size-4" />} testId="upload-done">
            {t.video.uploadDone}
          </Notice>
        )}
        {state.status === 'cancelled' && (
          <Notice tone="neutral" testId="upload-cancelled">
            {t.video.uploadCancelled}
          </Notice>
        )}
        {state.status === 'error' && state.error && (
          <ErrorNotice error={state.error} title={t.video.uploadFailed} testId="upload-error" />
        )}
        {state.status === 'error' && !state.error && state.message && (
          <Notice tone="warning" title={t.video.uploadFailed} testId="upload-error">
            {state.message}
          </Notice>
        )}
        {retryFile && (
          <Button size="sm" onClick={() => start(retryFile)} disabled={Boolean(disabledReason)} data-testid="upload-retry">
            {t.video.uploadRetry}
          </Button>
        )}
      </div>
    </div>
  );
}
