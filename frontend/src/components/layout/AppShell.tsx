import {
  ClipboardCheck,
  Film,
  GitCommitHorizontal,
  LayoutDashboard,
  Menu,
  Radio,
  Route,
  ServerCog,
  ShieldCheck,
  X,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { useEffect, useRef, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router';

import { eventHub, useEventStatus } from '../../hooks/useEvents';
import { useNow } from '../../hooks/useTimers';
import { useI18n } from '../../i18n';
import { shortHash } from '../../lib/format';
import type { Tone } from '../../lib/tones';
import { useApp } from '../../state/app';
import { ErrorBoundary } from '../ErrorBoundary';
import { Button } from '../ui/Button';
import { ErrorNotice } from '../ui/Feedback';
import { Pill } from '../ui/Status';
import { Segmented } from '../ui/Data';

function Logo({ className = 'size-8' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="8" fill="#12233B" />
      <path d="M9 25 L14.5 7 h3 L23 25" fill="none" stroke="#FFFFFF" strokeWidth="2.4" strokeLinejoin="round" />
      <path d="M16 11 v3 M16 17 v3" stroke="#0891B2" strokeWidth="2.4" strokeLinecap="round" />
    </svg>
  );
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  const { t } = useI18n();
  const items: { to: string; label: string; icon: ReactNode; testId: string; end?: boolean }[] = [
    { to: '/', label: t.nav.overview, icon: <LayoutDashboard className="size-[18px]" />, testId: 'nav-overview', end: true },
    { to: '/video', label: t.nav.video, icon: <Film className="size-[18px]" />, testId: 'nav-video' },
    { to: '/simulation', label: t.nav.simulation, icon: <Route className="size-[18px]" />, testId: 'nav-simulation' },
    { to: '/evaluation', label: t.nav.evaluation, icon: <ClipboardCheck className="size-[18px]" />, testId: 'nav-evaluation' },
    { to: '/system', label: t.nav.system, icon: <ServerCog className="size-[18px]" />, testId: 'nav-system' },
  ];
  return (
    <ul className="space-y-1">
      {items.map((item) => (
        <li key={item.to}>
          <NavLink
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            data-testid={item.testId}
            className={({ isActive }) =>
              `flex h-10 items-center gap-3 rounded-control px-3 text-sm font-medium transition-colors ${
                isActive ? 'bg-accent/8 text-accent-strong' : 'text-ink/80 hover:bg-canvas hover:text-ink'
              }`
            }
          >
            <span aria-hidden>{item.icon}</span>
            {item.label}
          </NavLink>
        </li>
      ))}
    </ul>
  );
}

function SafetyFootnote() {
  const { t } = useI18n();
  return (
    <div className="rounded-control border border-line bg-canvas p-3 text-xs text-muted">
      <p className="flex items-center gap-1.5 font-semibold text-ink">
        <ShieldCheck className="size-4 text-success" aria-hidden />
        {t.app.prototype}
      </p>
      <p className="mt-1">{t.overview.safetyBody}</p>
    </div>
  );
}

function ConnectionPill() {
  const { t } = useI18n();
  const { sessionError, session } = useApp();
  const streamStatus = useEventStatus();
  // Without a session the stream cannot start; report that as offline rather than "idle".
  const status = streamStatus === 'idle' && sessionError && !session ? 'offline' : streamStatus;
  const tone: Tone = status === 'online' ? 'success' : status === 'offline' ? 'danger' : 'warning';
  return (
    <Pill tone={tone} dot testId="sse-status" title={`${t.conn.label}: ${t.conn[status]}`}>
      <Radio className="mr-0.5 inline size-3" aria-hidden />
      {t.conn[status]}
    </Pill>
  );
}

function CapabilityPills() {
  const { t } = useI18n();
  const { capabilities, capabilitiesError } = useApp();
  if (!capabilities) {
    return capabilitiesError ? (
      <Pill tone="danger" dot testId="cap-pill-offline">
        {t.top.capsOffline}
      </Pill>
    ) : (
      <Pill tone="neutral">{t.top.capsLoading}</Pill>
    );
  }
  const video = capabilities.video_analysis;
  const simReady = capabilities.simulation.simulators.some((sim) => sim.available);
  return (
    <>
      <Pill tone={video.available ? 'success' : 'warning'} dot testId="cap-pill-video" title={video.reason ?? undefined}>
        {video.available ? t.top.videoReady : t.top.videoUnavailable}
      </Pill>
      <Pill tone={simReady ? 'success' : 'warning'} dot testId="cap-pill-sim">
        {simReady ? t.top.simReady : t.top.simUnavailable}
      </Pill>
    </>
  );
}

function BuildPill() {
  const { t } = useI18n();
  const { capabilities } = useApp();
  const build = capabilities?.build;
  if (!build) return null;
  const label = build.commit ? shortHash(build.commit, 8) : t.top.noCommit;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-line bg-surface px-2 py-0.5 text-xs text-muted"
      title={`${t.top.build}: ${build.commit ?? t.top.noCommit} · ${build.version}`}
      data-testid="build-commit"
    >
      <GitCommitHorizontal className="size-3.5" aria-hidden />
      <span className="font-mono">{label}</span>
      {build.dirty && <span className="text-warning">· {t.top.dirty}</span>}
    </span>
  );
}

function LanguageSwitch() {
  const { lang, setLang, t } = useI18n();
  return (
    <Segmented
      label={t.top.language}
      size="sm"
      value={lang}
      onChange={setLang}
      options={[
        { value: 'uz', label: 'UZ', testId: 'lang-uz' },
        { value: 'en', label: 'EN', testId: 'lang-en' },
      ]}
    />
  );
}

function PrototypeLabel() {
  const { t } = useI18n();
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full bg-ink px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap text-white"
      data-testid="prototype-label"
    >
      {t.app.prototype}
    </span>
  );
}

function OfflineBanner() {
  const { t } = useI18n();
  const { sessionError, sessionRetryAt, retrySession, session } = useApp();
  const now = useNow(1000, sessionRetryAt !== null);
  if (!sessionError || session) return null;
  const seconds = sessionRetryAt ? Math.max(0, Math.ceil((sessionRetryAt - now) / 1000)) : null;
  return (
    <div className="px-4 pt-4 sm:px-6" data-testid="offline-banner">
      <ErrorNotice error={sessionError} onRetry={retrySession} title={t.offline.title} testId="session-error">
        <p className="mt-1 text-sm text-muted">
          {t.offline.body}
          {seconds !== null && <> {t.offline.retryIn(seconds)}</>}
        </p>
      </ErrorNotice>
    </div>
  );
}

function StreamBanner() {
  const { t } = useI18n();
  const status = useEventStatus();
  const { session } = useApp();
  if (!session || (status !== 'offline' && status !== 'reconnecting')) return null;
  return (
    <div className="px-4 pt-4 sm:px-6">
      <div
        className="flex flex-wrap items-center justify-between gap-2 rounded-control border border-warning/30 bg-warning/5 px-3 py-2 text-sm"
        role="status"
        data-testid="stream-banner"
      >
        <span>{t.conn.pollingNote}</span>
        <Button size="sm" onClick={() => eventHub.reconnectNow()}>
          {t.conn.reconnect}
        </Button>
      </div>
    </div>
  );
}

export function AppShell() {
  const { t } = useI18n();
  const location = useLocation();
  // The drawer belongs to the path it was opened on, so any navigation (links, back/forward) closes it.
  const [menuPath, setMenuPath] = useState<string | null>(null);
  const menuOpen = menuPath === location.pathname;
  const setMenuOpen = (open: boolean) => setMenuPath(open ? location.pathname : null);
  const drawerRef = useRef<HTMLDivElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);

  // Move focus into the drawer when it opens; Escape closes it and returns focus to the menu button.
  useEffect(() => {
    if (!menuOpen) return;
    drawerRef.current?.querySelector<HTMLElement>('a,button')?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMenuPath(null);
        menuButtonRef.current?.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [menuOpen]);

  return (
    <div className="min-h-screen bg-canvas lg:flex">
      <a
        href="#main"
        className="sr-only z-50 rounded-control bg-surface px-4 py-2 text-sm font-medium focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
      >
        {t.app.skipToContent}
      </a>

      <aside className="hidden w-[232px] shrink-0 border-r border-line bg-surface lg:block">
        <div className="sticky top-0 flex h-screen flex-col">
          <div className="flex h-16 items-center gap-2.5 px-5">
            <Logo />
            <span className="text-[15px] font-semibold tracking-tight">{t.app.name}</span>
          </div>
          <nav aria-label={t.nav.label} className="flex-1 overflow-y-auto px-3 py-2">
            <NavItems />
          </nav>
          <div className="p-3">
            <SafetyFootnote />
          </div>
        </div>
      </aside>

      {menuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label={t.nav.closeMenu}
            className="absolute inset-0 bg-ink/40"
            onClick={() => setMenuOpen(false)}
          />
          <div
            ref={drawerRef}
            role="dialog"
            aria-modal="true"
            aria-label={t.nav.label}
            className="absolute inset-y-0 left-0 flex w-[280px] max-w-[85vw] flex-col bg-surface shadow-pop"
          >
            <div className="flex h-14 items-center justify-between px-4">
              <span className="flex items-center gap-2 font-semibold">
                <Logo className="size-7" />
                {t.app.name}
              </span>
              <button
                type="button"
                onClick={() => setMenuOpen(false)}
                aria-label={t.nav.closeMenu}
                className="inline-flex size-10 items-center justify-center rounded-control hover:bg-canvas"
                data-testid="nav-close"
              >
                <X className="size-5" aria-hidden />
              </button>
            </div>
            <nav aria-label={t.nav.label} className="flex-1 overflow-y-auto px-3 py-2">
              <NavItems onNavigate={() => setMenuOpen(false)} />
            </nav>
            <div className="space-y-3 p-3">
              <div className="flex flex-wrap gap-2">
                <CapabilityPills />
              </div>
              <SafetyFootnote />
            </div>
          </div>
        </div>
      )}

      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-30 border-b border-line bg-surface/95 backdrop-blur supports-[backdrop-filter]:bg-surface/85">
          <div className="flex min-h-14 flex-wrap items-center gap-x-3 gap-y-2 px-4 py-2 sm:px-6">
            <button
              ref={menuButtonRef}
              type="button"
              onClick={() => setMenuOpen(true)}
              aria-label={t.nav.openMenu}
              aria-expanded={menuOpen}
              className="-ml-2 inline-flex size-10 items-center justify-center rounded-control hover:bg-canvas lg:hidden"
              data-testid="nav-open"
            >
              <Menu className="size-5" aria-hidden />
            </button>
            <span className="flex items-center gap-2 text-sm font-semibold lg:hidden">
              <Logo className="size-6" />
              {t.app.name}
            </span>
            <span className="hidden text-sm font-semibold lg:inline">{t.app.name}</span>
            <span className="hidden sm:inline-flex">
              <BuildPill />
            </span>
            <div className="hidden items-center gap-2 lg:flex">
              <CapabilityPills />
            </div>
            <span className="hidden sm:inline-flex">
              <ConnectionPill />
            </span>
            <div className="ml-auto flex items-center gap-2">
              <span className="hidden sm:inline-flex">
                <PrototypeLabel />
              </span>
              <LanguageSwitch />
            </div>
            {/* Phones: second row keeps the prototype label and live status visible. */}
            <div className="flex w-full flex-wrap items-center gap-2 sm:hidden">
              <PrototypeLabel />
              <ConnectionPill />
              <BuildPill />
            </div>
          </div>
        </header>
        <OfflineBanner />
        <StreamBanner />
        <main id="main" tabIndex={-1} className="mx-auto w-full max-w-[1600px] px-4 py-6 outline-none sm:px-6 lg:py-8">
          <ErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
