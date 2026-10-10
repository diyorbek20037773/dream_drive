import { Activity, Boxes, Cpu, GitCommitHorizontal, Gauge, HeartPulse, ListChecks, Radio, Server, ShieldQuestion, UserRound } from 'lucide-react';
import type { ReactNode } from 'react';

import { api } from '../api/client';
import type { ModelEntry, ModelInfo } from '../api/types';
import { LimitationsList, SourcePill } from '../components/domain';
import { Card, CardHeader, PageHeader } from '../components/ui/Card';
import { CopyText, KeyValueList } from '../components/ui/Data';
import { ErrorNotice, LoadingBlock, Notice } from '../components/ui/Feedback';
import { Pill } from '../components/ui/Status';
import { eventHub, useEventStatus } from '../hooks/useEvents';
import { useResource } from '../hooks/useResource';
import { useInterval } from '../hooks/useTimers';
import { useI18n } from '../i18n';
import { shortHash, unavailableText, useFormat } from '../lib/format';
import { useApp } from '../state/app';
import { Button } from '../components/ui/Button';

/** Ensemble entries (docs/IMPLEMENTATION_SPEC.md section 2.1), read defensively until `models` is in the OpenAPI. */
function modelEntries(model: ModelInfo | null | undefined): ModelEntry[] {
  const raw = (model as { models?: unknown } | null | undefined)?.models;
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (entry): entry is ModelEntry =>
      typeof entry === 'object' && entry !== null && typeof (entry as { model_id?: unknown }).model_id === 'string',
  );
}

function Section({ title, icon, children, testId }: { title: string; icon: ReactNode; children: ReactNode; testId?: string }) {
  return (
    <Card className="card-pad" testId={testId}>
      <CardHeader title={title} icon={icon} />
      <div className="mt-3">{children}</div>
    </Card>
  );
}

export function SystemPage() {
  const { t } = useI18n();
  const f = useFormat();
  const { capabilities, capabilitiesError, capabilitiesLoading, refreshCapabilities, session } = useApp();
  const health = useResource('health', (signal) => api.healthReady(signal));
  useInterval(health.reload, 15_000);
  const streamStatus = useEventStatus();

  const offline = capabilitiesError && !capabilities ? capabilitiesError : null;
  const video = capabilities?.video_analysis;
  const model = video?.model;
  const models = modelEntries(model);
  const limits = capabilities?.limits ?? session?.limits;

  const limitValue = (key: string, value: unknown): string => {
    if (Array.isArray(value)) return value.map((item) => String(item).toUpperCase()).join(', ');
    if (typeof value !== 'number') return String(value);
    if (key === 'max_upload_bytes') return f.bytes(value);
    if (key.endsWith('_s')) return `${f.num(value, 0)} ${t.units.s}`;
    if (key.endsWith('_px')) return `${f.int(value)} ${t.units.px}`;
    if (key === 'analysis_fps') return `${f.num(value, 1)} ${t.units.fps}`;
    if (key.endsWith('_hours')) return `${f.int(value)} ${t.units.hours}`;
    return f.int(value);
  };

  return (
    <div data-testid="page-system">
      <PageHeader title={t.system.title} subtitle={t.system.subtitle} />

      <div className="mb-6 grid gap-3 md:grid-cols-2">
        <Notice tone="neutral" icon={<ShieldQuestion className="size-4" />} title={t.common.featureAbsent}>
          {t.system.legendAbsent}
        </Notice>
        <Notice tone="neutral" icon={<Activity className="size-4" />} title={t.common.serviceOffline}>
          {t.system.legendOffline}
        </Notice>
      </div>

      {offline && (
        <div className="mb-6">
          <ErrorNotice error={offline} onRetry={refreshCapabilities} testId="system-offline" />
        </div>
      )}
      {!capabilities && !offline && capabilitiesLoading && <LoadingBlock />}

      <div className="grid items-start gap-6 lg:grid-cols-2 xl:grid-cols-3">
        <Section title={t.system.buildTitle} icon={<GitCommitHorizontal className="size-5" />} testId="system-build">
          {capabilities ? (
            <KeyValueList
              items={[
                {
                  label: t.system.commit,
                  value: capabilities.build.commit ? (
                    <CopyText value={capabilities.build.commit} display={shortHash(capabilities.build.commit, 12)} testId="system-commit" />
                  ) : (
                    t.system.noCommit
                  ),
                },
                { label: t.system.version, value: capabilities.build.version },
                { label: t.system.tree, value: capabilities.build.dirty ? t.system.dirty : t.system.clean },
              ]}
            />
          ) : (
            <p className="text-sm text-muted">{offline ? t.common.serviceOffline : t.common.loading}</p>
          )}
        </Section>

        <Section title={t.system.apiTitle} icon={<HeartPulse className="size-5" />} testId="system-health">
          {health.data ? (
            <>
              <Pill tone={health.data.status === 'ok' ? 'success' : 'danger'} dot testId="system-health-status">
                {health.data.status === 'ok' ? t.system.apiReady : t.system.apiUnavailable}
              </Pill>
              <KeyValueList
                className="mt-2"
                items={[
                  ...Object.entries(health.data.checks).map(([name, check]) => ({
                    label: <code className="font-mono text-xs">{name}</code>,
                    value: check.ok ? (
                      <span className="text-success">{t.common.yes}</span>
                    ) : (
                      <span className="text-danger">{check.reason ?? t.common.no}</span>
                    ),
                  })),
                  { label: t.system.checkedAt, value: f.dateTime(health.data.checked_utc) },
                ]}
              />
            </>
          ) : health.error ? (
            <ErrorNotice error={health.error} onRetry={health.reload} compact testId="system-health-error" />
          ) : (
            <LoadingBlock />
          )}
        </Section>

        <Section title={t.system.modelTitle} icon={<Cpu className="size-5" />} testId="system-model">
          {!video ? (
            <p className="text-sm text-muted">{offline ? t.common.serviceOffline : t.common.loading}</p>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Pill tone={video.available ? 'success' : 'warning'} dot testId="system-model-ready">
                  {video.available ? t.system.modelReady : t.system.modelNotReady}
                </Pill>
                {video.available && <SourcePill source="learned_model" />}
              </div>
              {!video.available && (
                <p className="mt-2 text-sm text-muted" data-testid="system-model-reason">
                  {t.common.featureAbsent}: {unavailableText(t, video.reason)}
                  {video.reason && <code className="ml-1 font-mono text-xs">({video.reason})</code>}
                </p>
              )}
              {model ? (
                <KeyValueList
                  className="mt-2"
                  items={[
                    { label: t.system.modelId, value: model.model_id },
                    { label: t.system.device, value: (video.device ?? model.device ?? '—').toUpperCase() },
                    { label: t.system.precision, value: video.precision ?? model.precision ?? '—' },
                    {
                      label: t.system.sha256,
                      value: model.sha256 ? <CopyText value={model.sha256} display={shortHash(model.sha256, 16)} testId="system-model-sha" /> : '—',
                    },
                    { label: t.system.size, value: f.bytes(model.size_bytes ?? null) },
                    { label: t.system.revision, value: model.repository_revision ? shortHash(model.repository_revision, 12) : '—' },
                    { label: t.system.torch, value: model.torch_version ?? '—' },
                    { label: t.system.inputPolicy, value: model.input_policy ?? '—' },
                  ]}
                />
              ) : (
                <p className="mt-2 text-sm text-muted">{t.system.noModelInfo}</p>
              )}
              {models.length > 0 && (
                <ul className="mt-3 space-y-3" data-testid="system-models">
                  {models.map((entry) => (
                    <li key={entry.model_id} className="rounded-control border border-line p-3 text-sm">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-semibold">{entry.model_id}</span>
                        <Pill tone={entry.ready === false ? 'warning' : entry.ready ? 'success' : 'neutral'} dot>
                          {entry.ready === false ? t.system.modelNotReady : entry.ready ? t.system.modelReady : t.common.unknown}
                        </Pill>
                        {(entry.device ?? entry.precision) && (
                          <span className="text-xs text-muted">
                            {(entry.device ?? '').toUpperCase()} {entry.precision ?? ''}
                          </span>
                        )}
                      </div>
                      {entry.reason && <p className="mt-1 text-xs text-muted">{unavailableText(t, entry.reason)}</p>}
                      {entry.sha256 && (
                        <div className="mt-1 text-xs">
                          {t.system.sha256}: <CopyText value={entry.sha256} display={shortHash(entry.sha256, 16)} />
                        </div>
                      )}
                      {entry.class_map && Object.keys(entry.class_map).length > 0 && (
                        <ul className="mt-2 flex flex-wrap gap-1">
                          {Object.entries(entry.class_map).map(([id, name]) => (
                            <li key={id}>
                              <Pill tone="info">
                                {id}: {name}
                              </Pill>
                            </li>
                          ))}
                        </ul>
                      )}
                    </li>
                  ))}
                </ul>
              )}
              <div className="mt-3 space-y-2">
                <p className="label-caps">{t.system.classMap}</p>
                {model?.class_map && Object.keys(model.class_map).length > 0 ? (
                  <ul className="flex flex-wrap gap-1.5" data-testid="system-class-map">
                    {Object.entries(model.class_map).map(([id, name]) => (
                      <li key={id}>
                        <Pill tone="info">
                          {id}: {name}
                        </Pill>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted">—</p>
                )}
                <p className="label-caps pt-1">{t.system.supported}</p>
                <p className="text-sm">{video.supported_classes.length > 0 ? video.supported_classes.join(', ') : '—'}</p>
                <p className="label-caps pt-1">{t.system.unsupported}</p>
                <ul className="flex flex-wrap gap-1.5">
                  {video.unsupported_examples.map((name) => (
                    <li key={name}>
                      <Pill tone="warning">{name}</Pill>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </Section>

        <Section title={t.system.simsTitle} icon={<Boxes className="size-5" />} testId="system-simulators">
          {!capabilities ? (
            <p className="text-sm text-muted">{offline ? t.common.serviceOffline : t.common.loading}</p>
          ) : (
            <ul className="space-y-4">
              {capabilities.simulation.simulators.map((sim) => (
                <li key={sim.id} data-testid={`system-sim-${sim.id}`}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold">{sim.name}</span>
                    <code className="font-mono text-xs text-muted">{sim.id}</code>
                    <Pill tone={sim.available ? 'success' : 'neutral'} dot>
                      {sim.available ? t.system.modelReady : t.common.featureAbsent}
                    </Pill>
                  </div>
                  {!sim.available && <p className="mt-1 text-sm text-muted">{unavailableText(t, sim.reason)}</p>}
                  <p className="label-caps mt-2">{t.system.modes}</p>
                  <ul className="mt-1 space-y-1 text-sm">
                    {sim.perception_modes.map((mode) => (
                      <li key={mode.id} className="flex flex-wrap items-center gap-2">
                        <SourcePill source={mode.id} />
                        {mode.available ? (
                          <span className="text-success">{t.system.modelReady}</span>
                        ) : (
                          <span className="text-muted">{unavailableText(t, mode.reason)}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                  <p className="label-caps mt-2">{t.system.scenariosLabel}</p>
                  <p className="mt-1 text-sm">
                    {sim.scenarios.map((scenario) => (
                      <span key={scenario.id} className={`mr-2 inline-block ${scenario.available ? '' : 'text-muted line-through'}`} title={scenario.reason ?? undefined}>
                        {scenario.id}
                      </span>
                    ))}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Section>

        <Section title={t.system.queueTitle} icon={<Gauge className="size-5" />} testId="system-queue">
          {!capabilities ? (
            <p className="text-sm text-muted">{offline ? t.common.serviceOffline : t.common.loading}</p>
          ) : (
            <>
              <KeyValueList
                items={[
                  {
                    label: t.system.videoQueued,
                    value: capabilities.queue.video_queued === null ? t.system.queueUnreadable : f.int(capabilities.queue.video_queued),
                  },
                  {
                    label: t.system.videoRunning,
                    value: capabilities.queue.video_running === null ? t.system.queueUnreadable : f.int(capabilities.queue.video_running),
                  },
                  { label: t.system.capacity, value: f.int(capabilities.queue.capacity_total) },
                  { label: t.system.perSession, value: f.int(capabilities.queue.per_session) },
                  {
                    label: t.system.simActive,
                    value:
                      capabilities.queue.sim_active === null
                        ? t.system.queueUnreadable
                        : `${f.int(capabilities.queue.sim_active)} / ${f.int(capabilities.queue.sim_capacity)}`,
                  },
                ]}
              />
              {capabilities.queue.reason && <p className="mt-2 text-xs text-muted">{capabilities.queue.reason}</p>}
            </>
          )}
        </Section>

        <Section title={t.system.limitsTitle} icon={<ListChecks className="size-5" />} testId="system-limits">
          {limits ? (
            <KeyValueList
              items={Object.entries(limits).map(([key, value]) => ({
                label: t.system.limits[key] ?? key,
                value: limitValue(key, value),
              }))}
            />
          ) : (
            <p className="text-sm text-muted">{offline ? t.common.serviceOffline : t.common.loading}</p>
          )}
        </Section>

        <Section title={t.system.sessionTitle} icon={<UserRound className="size-5" />} testId="system-session">
          {session ? (
            <KeyValueList
              items={[
                { label: t.system.role, value: t.system.roles[session.role] ?? session.role },
                { label: t.system.sessionSuffix, value: <code className="font-mono">…{session.session_id_suffix}</code> },
                { label: t.system.expires, value: f.dateTime(session.expires_utc) },
              ]}
            />
          ) : (
            <p className="text-sm text-muted">{t.common.serviceOffline}</p>
          )}
        </Section>

        <Section title={t.system.eventsTitle} icon={<Radio className="size-5" />} testId="system-events">
          <div className="flex flex-wrap items-center gap-3">
            <Pill tone={streamStatus === 'online' ? 'success' : streamStatus === 'offline' ? 'danger' : 'warning'} dot>
              {t.conn[streamStatus]}
            </Pill>
            {streamStatus !== 'online' && session && (
              <Button size="sm" onClick={() => eventHub.reconnectNow()}>
                {t.conn.reconnect}
              </Button>
            )}
          </div>
        </Section>

        <Section title={t.system.monitoringTitle} icon={<Server className="size-5" />} testId="system-monitoring">
          <p className="text-sm">
            {capabilities ? (capabilities.monitoring.metrics ? t.system.monitoringOn : t.system.monitoringOff) : t.common.serviceOffline}
          </p>
        </Section>
      </div>

      <Card className="card-pad mt-6" testId="system-limitations">
        <CardHeader title={t.system.limitationsTitle} />
        <div className="mt-3">
          <LimitationsList items={t.system.limitationsList} />
        </div>
      </Card>
    </div>
  );
}
