import { AlertTriangle, CircleSlash, Info, PlugZap, RotateCw } from 'lucide-react';
import type { ReactNode } from 'react';

import type { ApiError } from '../../api/client';
import { useI18n } from '../../i18n';
import { errorText } from '../../lib/format';
import { Button } from './Button';
import { Spinner } from './Status';

/** Localized ApiError with request id, retry-after hint and an optional retry button. */
export function ErrorNotice({
  error,
  onRetry,
  title,
  compact = false,
  testId = 'error-notice',
  children,
}: {
  error: ApiError;
  onRetry?: () => void;
  title?: string;
  compact?: boolean;
  testId?: string;
  children?: ReactNode;
}) {
  const { t } = useI18n();
  const offline = error.isOffline;
  const notImplemented = error.code === 'not_implemented';
  const Icon = offline ? PlugZap : notImplemented ? CircleSlash : AlertTriangle;
  const tone = notImplemented ? 'border-line bg-canvas text-ink' : 'border-danger/25 bg-danger/5 text-ink';
  const heading =
    title ?? (offline ? t.common.serviceOffline : notImplemented ? t.common.notImplemented : t.common.errorTitle);
  return (
    <div role="alert" data-testid={testId} data-error-code={error.code} className={`rounded-control border ${tone} ${compact ? 'p-3' : 'p-4'}`}>
      <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
        <Icon className={`mt-0.5 size-5 shrink-0 ${notImplemented ? 'text-muted' : 'text-danger'}`} aria-hidden />
        <div className="min-w-[12rem] flex-1">
          <p className="font-semibold">{heading}</p>
          <p className="mt-0.5 text-sm break-words">{errorText(t, error)}</p>
          {error.retryAfterS !== null && <p className="mt-1 text-sm text-muted">{t.common.retryAfter(error.retryAfterS)}</p>}
          {children}
          <p className="mt-1 text-xs break-words text-muted">
            <code>{error.code}</code>
            {error.status > 0 && <> · HTTP {error.status}</>}
            {error.requestId && (
              <>
                {' '}
                · {t.common.requestId}: <code>{error.requestId}</code>
              </>
            )}
          </p>
        </div>
        {onRetry && (
          <Button size="sm" onClick={onRetry} icon={<RotateCw className="size-4" aria-hidden />} data-testid={`${testId}-retry`}>
            {t.common.retry}
          </Button>
        )}
      </div>
    </div>
  );
}

export function Notice({
  tone = 'info',
  title,
  children,
  icon,
  testId,
}: {
  tone?: 'info' | 'warning' | 'neutral' | 'success';
  title?: ReactNode;
  children?: ReactNode;
  icon?: ReactNode;
  testId?: string;
}) {
  const styles = {
    info: 'border-accent/20 bg-accent/5',
    warning: 'border-warning/30 bg-warning/5',
    neutral: 'border-line bg-canvas',
    success: 'border-success/25 bg-success/5',
  }[tone];
  const iconColor = { info: 'text-accent', warning: 'text-warning', neutral: 'text-muted', success: 'text-success' }[tone];
  return (
    <div className={`rounded-control border p-3 text-sm ${styles}`} data-testid={testId}>
      <div className="flex items-start gap-2.5">
        <span className={`mt-0.5 shrink-0 ${iconColor}`} aria-hidden>
          {icon ?? <Info className="size-4" />}
        </span>
        <div className="min-w-0">
          {title && <p className="font-semibold text-ink">{title}</p>}
          {children && <div className={title ? 'mt-0.5 text-ink/90' : 'text-ink/90'}>{children}</div>}
        </div>
      </div>
    </div>
  );
}

export function EmptyState({ icon, title, children, testId }: { icon?: ReactNode; title: ReactNode; children?: ReactNode; testId?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-control border border-dashed border-line px-4 py-8 text-center" data-testid={testId}>
      {icon && <span className="text-muted" aria-hidden>{icon}</span>}
      <p className="font-medium text-ink">{title}</p>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}

export function LoadingBlock({ label, className = '' }: { label?: string; className?: string }) {
  const { t } = useI18n();
  return (
    <div className={`flex items-center gap-2 py-6 text-sm text-muted ${className}`} role="status" aria-live="polite">
      <Spinner className="size-4" />
      {label ?? t.common.loading}
    </div>
  );
}
