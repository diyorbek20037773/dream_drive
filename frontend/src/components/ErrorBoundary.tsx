import { AlertTriangle, RotateCw } from 'lucide-react';
import type { ErrorInfo, ReactNode } from 'react';
import { Component } from 'react';

import { useI18n } from '../i18n';

function Fallback({ onReset, message }: { onReset: () => void; message: string }) {
  const { t } = useI18n();
  return (
    <div role="alert" className="rounded-card border border-danger/25 bg-danger/5 p-6" data-testid="render-error">
      <p className="flex items-center gap-2 font-semibold">
        <AlertTriangle className="size-5 text-danger" aria-hidden />
        {t.common.renderErrorTitle}
      </p>
      <p className="mt-1 text-sm text-ink/85">{t.common.renderErrorBody}</p>
      <p className="mt-1 font-mono text-xs break-words text-muted">{message}</p>
      <button
        type="button"
        onClick={onReset}
        className="mt-4 inline-flex h-10 items-center gap-2 rounded-control border border-line bg-surface px-4 text-sm font-medium hover:bg-canvas"
      >
        <RotateCw className="size-4" aria-hidden />
        {t.common.retry}
      </button>
    </div>
  );
}

/**
 * Keeps the shell (navigation, language switch, offline banner) usable when one page fails to render, e.g. on an
 * unexpected API payload. `resetKey` (the route) clears the error on navigation.
 */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey: string }, { error: Error | null; key: string }> {
  override state: { error: Error | null; key: string } = { error: null, key: this.props.resetKey };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  static getDerivedStateFromProps(props: { resetKey: string }, state: { error: Error | null; key: string }) {
    return props.resetKey !== state.key ? { error: null, key: props.resetKey } : null;
  }

  override componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('page render failed', error, info.componentStack);
  }

  override render() {
    if (this.state.error) {
      return <Fallback message={this.state.error.message} onReset={() => this.setState({ error: null })} />;
    }
    return this.props.children;
  }
}
