import { Construction, History, Map as MapIcon, Pause, Play, Plus, Settings2, Square, Video } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';

import type { ApiError } from '../../api/client';
import { api, toApiError } from '../../api/client';
import type { ObstacleInjectResult, Run, ScenarioId, SimPerceptionSource, Simulator, SimulatorId } from '../../api/types';
import { isActiveRun } from '../../api/types';
import { SessionPending, SourcePill } from '../../components/domain';
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, PageHeader } from '../../components/ui/Card';
import { KeyValueList, Segmented } from '../../components/ui/Data';
import { EmptyState, ErrorNotice, LoadingBlock, Notice } from '../../components/ui/Feedback';
import { Pill } from '../../components/ui/Status';
import { useEvents, useEventStatus, useResync } from '../../hooks/useEvents';
import { useResource } from '../../hooks/useResource';
import { useInterval } from '../../hooks/useTimers';
import { useI18n } from '../../i18n';
import { outcomeText, unavailableText, useFormat } from '../../lib/format';
import { connectedNodes, reachableFrom } from '../../lib/graph';
import type { ManeuverChange } from '../../lib/maneuver';
import { appendChange } from '../../lib/maneuver';
import { RUN_TONE } from '../../lib/tones';
import { useApp } from '../../state/app';
import { DecisionPanel } from './DecisionPanel';
import { LiveCamera } from './LiveCamera';
import { ManeuverHistory, ManeuverIndicator } from './Maneuver';
import type { MapPickMode } from './RoadMap';
import { MapLegend, RoadMap } from './RoadMap';
import { RunReportView } from './RunReportView';
import { useLiveRun } from './useLiveRun';

const SEED_MAX = 1_000_000;

function parseSeed(text: string): number | null {
  if (!/^\d{1,7}$/.test(text.trim())) return null;
  const value = Number(text.trim());
  return value >= 0 && value <= SEED_MAX ? value : null;
}

const selectClass =
  'h-10 w-full rounded-control border border-line bg-surface px-3 text-sm text-ink disabled:bg-canvas disabled:text-muted';

function ObstaclePanel({ run, maxPerRun }: { run: Run; maxPerRun: number | null }) {
  const { t } = useI18n();
  const f = useFormat();
  const [distance, setDistance] = useState(20);
  const [lateral, setLateral] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ObstacleInjectResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const canInject = run.state === 'running' || run.state === 'paused';
  const inject = () => {
    setBusy(true);
    setError(null);
    api.injectObstacle(run.id, { distance_ahead_m: distance, lateral_offset_m: lateral }).then(
      (value) => {
        setResult(value);
        setBusy(false);
      },
      (reason: unknown) => {
        setError(toApiError(reason));
        setBusy(false);
      },
    );
  };
  return (
    <Card className="card-pad" testId="obstacle-panel">
      <CardHeader title={t.sim.obstacleTitle} icon={<Construction className="size-5" />} />
      <p className="mt-2 text-xs text-muted">{t.sim.obstacleNote}</p>
      <div className="mt-4 space-y-4">
        <div className="text-sm">
          <div className="flex justify-between">
            <label htmlFor="obstacle-distance">{t.sim.obstacleDistance}</label>
            <span className="num font-medium">
              {f.num(distance, 0)} {t.units.m}
            </span>
          </div>
          <input
            id="obstacle-distance"
            type="range"
            min={8}
            max={40}
            step={1}
            value={distance}
            onChange={(event) => setDistance(Number(event.target.value))}
            className="mt-1 w-full"
            disabled={!canInject}
            data-testid="obstacle-distance"
          />
        </div>
        <div className="text-sm">
          <div className="flex justify-between">
            <label htmlFor="obstacle-lateral">{t.sim.obstacleLateral}</label>
            <span className="num font-medium">
              {f.num(lateral, 1)} {t.units.m}
            </span>
          </div>
          <input
            id="obstacle-lateral"
            type="range"
            min={-1.5}
            max={1.5}
            step={0.1}
            value={lateral}
            onChange={(event) => setLateral(Number(event.target.value))}
            className="mt-1 w-full"
            disabled={!canInject}
            data-testid="obstacle-lateral"
          />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={inject} busy={busy} disabled={!canInject} data-testid="obstacle-inject">
            {t.sim.obstacleInject}
          </Button>
          {result ? (
            <span className="num text-sm text-muted" data-testid="obstacle-count">
              {t.sim.obstacleInjected(result.injected_total, result.max_per_run)}
            </span>
          ) : (
            maxPerRun !== null && <span className="num text-sm text-muted">{t.sim.obstacleInjected(0, maxPerRun)}</span>
          )}
        </div>
        {!canInject && <p className="text-xs text-muted">{t.sim.obstacleOnlyActive}</p>}
        {error && <ErrorNotice error={error} compact testId="obstacle-error" />}
      </div>
    </Card>
  );
}

export function SimulationPage() {
  const { t, lang } = useI18n();
  const f = useFormat();
  const navigate = useNavigate();
  const { runId } = useParams();
  const { session, capabilities, limits, refreshCapabilities } = useApp();
  const streamStatus = useEventStatus();

  const simulators = useResource(session ? 'simulators' : null, (signal) => api.listSimulators(signal));
  const runs = useResource(session ? 'runs' : null, (signal) => api.listRuns(10, signal));
  const live = useLiveRun(runId ?? null);
  const run = live.run;
  const [frameTick, setFrameTick] = useState(0);
  // Frame id of the image the camera shows; the decision panel and the strip below the camera describe that frame.
  const [cameraFrameId, setCameraFrameId] = useState<number | null>(null);
  const onShownFrame = useCallback((frameId: number | null) => setCameraFrameId(frameId), []);
  // Maneuver changes of the selected run received as SSE decision events.
  const [sseChanges, setSseChanges] = useState<{ runId: string | null; items: readonly ManeuverChange[] }>({ runId: null, items: [] });
  const pushChange = useCallback((forRun: string, change: ManeuverChange) => {
    setSseChanges((previous) => {
      const base = previous.runId === forRun ? previous.items : [];
      const items = appendChange(base, change, 60);
      return items === base && previous.runId === forRun ? previous : { runId: forRun, items };
    });
  }, []);

  // Setup choices (null = use the default derived from /simulators).
  const [simChoice, setSimChoice] = useState<SimulatorId | null>(null);
  const [modeChoice, setModeChoice] = useState<SimPerceptionSource | null>(null);
  const [scenarioChoice, setScenarioChoice] = useState<ScenarioId | null>(null);
  const [seedText, setSeedText] = useState('42');
  const [startChoice, setStartChoice] = useState<string | null>(null);
  const [destination, setDestination] = useState<string | null>(null);
  const [mapMode, setMapMode] = useState<MapPickMode>('destination');

  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<ApiError | null>(null);
  const [controlBusy, setControlBusy] = useState<'pause' | 'resume' | 'stop' | null>(null);
  const [controlError, setControlError] = useState<ApiError | null>(null);
  const [stopRequestedFor, setStopRequestedFor] = useState<string | null>(null);

  const simList = simulators.data?.items ?? [];
  const simulator: Simulator | undefined =
    simList.find((sim) => sim.id === simChoice) ?? simList.find((sim) => sim.available) ?? simList[0];
  const modes = simulator?.perception_modes ?? [];
  const mode = modes.find((item) => item.id === modeChoice) ?? modes.find((item) => item.available) ?? modes[0];
  const scenarios = simulator?.scenarios ?? [];
  const scenario =
    scenarios.find((item) => item.id === scenarioChoice) ?? scenarios.find((item) => item.id === 'normal') ?? scenarios[0];
  const graph = simulator?.road_graph ?? null;
  const start = startChoice ?? simulator?.default_start ?? null;
  const seed = parseSeed(seedText);

  const startCandidates = useMemo(() => (graph ? connectedNodes(graph) : new Set<string>()), [graph]);
  const reachable = useMemo(() => (graph ? reachableFrom(graph, start) : new Set<string>()), [graph, start]);
  const destinationCandidates = useMemo(() => {
    const set = new Set(reachable);
    if (start) set.delete(start);
    return set;
  }, [reachable, start]);

  const runIsActive = run ? isActiveRun(run.state) : false;
  const otherActive = runs.data?.items.find((item) => isActiveRun(item.state) && item.id !== runId);
  const queue = capabilities?.queue;
  const instanceBusy = queue?.sim_active !== null && queue?.sim_active !== undefined && queue.sim_active >= queue.sim_capacity;

  // Re-attach to an active run after a reload (once per page visit).
  const attached = useRef(false);
  useEffect(() => {
    if (attached.current || runId || !runs.data) return;
    attached.current = true;
    const active = runs.data.items.find((item) => isActiveRun(item.state));
    if (active) void navigate(`/simulation/runs/${active.id}`, { replace: true });
  }, [runs.data, runId, navigate]);

  useEvents({
    run_state: (payload) => {
      runs.reload();
      refreshCapabilities();
      if (payload.run_id === runId) {
        live.refresh();
        setFrameTick((value) => value + 1);
      }
    },
    frame_ready: (payload) => {
      if (payload.run_id !== runId) return;
      live.refresh();
      setFrameTick((value) => value + 1);
    },
    decision: (payload) => {
      if (payload.run_id !== runId || !payload.maneuver) return;
      pushChange(payload.run_id, {
        primary: payload.maneuver.primary,
        longitudinal: payload.maneuver.longitudinal,
        lateral: payload.maneuver.lateral,
        since_sim_time_s: payload.maneuver.since_sim_time_s,
        frame_id: payload.frame_id,
        reason_codes: payload.reason_codes,
      });
    },
  });
  // History = SSE changes merged with the fetched live snapshots (which keep it filling when the stream is down),
  // ordered by simulated time; consecutive identical maneuvers collapse; the newest 30 are shown.
  const sseItems = sseChanges.runId === runId ? sseChanges.items : null;
  const historyItems = useMemo(() => {
    const fromLive: ManeuverChange[] = live.lives.flatMap((item) =>
      item.maneuver
        ? [
            {
              primary: item.maneuver.primary,
              longitudinal: item.maneuver.longitudinal,
              lateral: item.maneuver.lateral,
              since_sim_time_s: item.maneuver.since_sim_time_s,
              frame_id: item.frame_id,
              reason_codes: item.decision.reason_codes,
            },
          ]
        : [],
    );
    const all = [...(sseItems ?? []), ...fromLive].sort((a, b) => a.since_sim_time_s - b.since_sim_time_s || a.frame_id - b.frame_id);
    let items: readonly ManeuverChange[] = [];
    for (const change of all) items = appendChange(items, change);
    return items;
  }, [sseItems, live.lives]);
  // Snapshot of the frame on screen (falls back to the newest snapshot when that frame's state was not fetched).
  const shownLive =
    (cameraFrameId !== null && run?.live?.frame_id !== cameraFrameId ? live.lives.find((item) => item.frame_id === cameraFrameId) : undefined) ??
    run?.live ??
    null;
  useResync(() => {
    simulators.reload();
    runs.reload();
    live.refresh();
    setFrameTick((value) => value + 1);
  });
  useInterval(
    () => {
      live.refresh();
      setFrameTick((value) => value + 1);
    },
    runIsActive ? (streamStatus === 'online' ? 5_000 : 1_000) : null,
  );
  useInterval(runs.reload, runIsActive || otherActive ? 10_000 : null);

  const disabledReason: string | null = !session
    ? t.errors.service_offline
    : !simulator
      ? simulators.error
        ? t.errors[simulators.error.code]
        : t.common.loading
    : !simulator.available
      ? `${t.sim.simUnavailable} ${unavailableText(t, simulator.reason)}`
      : !mode?.available
        ? t.sim.modeUnavailable
        : !scenario?.available
          ? t.sim.scenarioUnavailable
          : runIsActive || otherActive
            ? t.sim.runActive
            : instanceBusy
              ? t.errors.simulation_busy
              : !start
                ? t.sim.needStart
                : !destination
                  ? t.sim.needDestination
                  : destination === start
                    ? t.sim.sameNode
                    : !reachable.has(destination)
                      ? t.sim.unreachable
                      : seed === null
                        ? t.sim.seedInvalid
                        : null;

  const startRun = () => {
    if (disabledReason || !simulator || !mode || !scenario || !start || !destination || seed === null) return;
    setStarting(true);
    setStartError(null);
    api
      .createRun({
        simulator: simulator.id,
        perception_source: mode.id,
        scenario: scenario.id,
        seed,
        start_node: start,
        destination_node: destination,
      })
      .then(
        (created) => {
          setStarting(false);
          runs.setData((previous) => (previous ? { ...previous, items: [created, ...previous.items] } : previous));
          void navigate(`/simulation/runs/${created.id}`);
        },
        (error: unknown) => {
          setStarting(false);
          setStartError(toApiError(error));
        },
      );
  };

  const control = (action: 'pause' | 'resume' | 'stop') => {
    if (!run) return;
    if (action === 'stop') setStopRequestedFor(run.id);
    setControlBusy(action);
    setControlError(null);
    api.controlRun(run.id, action).then(
      (updated) => {
        live.accept(updated);
        setControlBusy(null);
        live.refresh();
      },
      (error: unknown) => {
        setControlError(toApiError(error));
        setControlBusy(null);
        if (action === 'stop') setStopRequestedFor(null);
      },
    );
  };

  const stopping = run !== null && (run.state === 'stopping' || (stopRequestedFor === run.id && isActiveRun(run.state)));
  const nodeLabel = (id: string | null | undefined) => {
    const node = graph?.nodes.find((item) => item.id === id);
    return node ? (lang === 'uz' ? node.label_uz : node.label_en) : (id ?? '—');
  };

  const pickNode = (nodeId: string) => {
    if (mapMode === 'start') {
      setStartChoice(nodeId);
      if (destination === nodeId) setDestination(null);
      setMapMode('destination');
    } else {
      setDestination(nodeId);
    }
  };

  const showRunOnMap = run !== null;
  const setupEditable = !runId;

  // ---------------------------------------------------------------------------------------------------------------
  const cameraCard = (
    <Card className="overflow-hidden" testId="sim-camera-card">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 sm:px-5">
        <div className="flex items-center gap-2">
          <Video className="size-5 text-muted" aria-hidden />
          <h2 className="text-base font-semibold">{t.sim.cameraTitle}</h2>
        </div>
        {run && (
          <div className="flex flex-wrap items-center gap-2" aria-live="polite">
            <Pill tone={RUN_TONE[run.state]} dot testId="run-state">
              {t.states.run[run.state]}
            </Pill>
            {run.outcome && (
              <Pill tone={run.state === 'failed' ? 'danger' : run.state === 'arrived' ? 'success' : 'neutral'} testId="run-outcome">
                {outcomeText(t, run.outcome)}
              </Pill>
            )}
          </div>
        )}
      </div>
      <LiveCamera
        key={runId ?? 'none'}
        runId={runId ?? null}
        frameTick={frameTick}
        lives={live.lives}
        active={runIsActive}
        hasRun={Boolean(run)}
        onShownFrame={onShownFrame}
      />
      {shownLive?.maneuver && (
        <div className="px-4 pt-3 sm:px-5">
          <ManeuverIndicator maneuver={shownLive.maneuver} frameId={shownLive.frame_id} compact testId="camera-maneuver" />
        </div>
      )}
      <div
        className="flex flex-wrap items-center gap-2 px-4 py-3 sm:px-5"
        role="toolbar"
        aria-label={t.sim.controlsLabel}
        data-testid="sim-toolbar"
        data-run-state={run?.state ?? 'none'}
        data-run-outcome={run?.outcome ?? ''}
      >
        {runIsActive && run ? (
          <>
            {run.state === 'paused' ? (
              <Button
                icon={<Play className="size-4" aria-hidden />}
                onClick={() => control('resume')}
                busy={controlBusy === 'resume'}
                disabled={stopping}
                data-testid="sim-resume"
              >
                {t.sim.resume}
              </Button>
            ) : (
              <Button
                icon={<Pause className="size-4" aria-hidden />}
                onClick={() => control('pause')}
                busy={controlBusy === 'pause'}
                disabled={run.state !== 'running' || stopping}
                title={run.state !== 'running' ? t.states.run[run.state] : undefined}
                data-testid="sim-pause"
              >
                {t.sim.pause}
              </Button>
            )}
            {/* Stop is never disabled while a run is active; repeated stops are idempotent. */}
            <Button
              variant="danger"
              icon={<Square className="size-4" aria-hidden />}
              onClick={() => control('stop')}
              busy={stopping || controlBusy === 'stop'}
              data-testid="sim-stop"
              data-stopping={stopping}
            >
              {stopping ? t.sim.stopping : t.sim.stop}
            </Button>
          </>
        ) : runId ? (
          <Button
            variant="primary"
            icon={<Plus className="size-4" aria-hidden />}
            onClick={() => void navigate('/simulation')}
            data-testid="sim-new"
          >
            {t.sim.newRun}
          </Button>
        ) : (
          <>
            <Button
              variant="primary"
              icon={<Play className="size-4" aria-hidden />}
              onClick={startRun}
              busy={starting}
              disabled={Boolean(disabledReason)}
              aria-describedby="sim-start-reason"
              data-testid="sim-start"
            >
              {starting ? t.sim.starting : t.sim.startRun}
            </Button>
            {disabledReason && (
              <p id="sim-start-reason" className="text-sm text-muted" data-testid="sim-start-reason">
                {disabledReason}
              </p>
            )}
          </>
        )}
      </div>
      {(controlError || startError) && (
        <div className="space-y-2 px-4 pb-4 sm:px-5">
          {controlError && <ErrorNotice error={controlError} compact testId="sim-control-error" />}
          {startError && <ErrorNotice error={startError} compact testId="sim-start-error" />}
        </div>
      )}
    </Card>
  );

  const mapCard = (
    <Card className="card-pad" testId="sim-map-card">
      <CardHeader
        title={t.sim.mapTitle}
        icon={<MapIcon className="size-5" />}
        subtitle={t.sim.frameNote}
        actions={
          !showRunOnMap && graph ? (
            <Segmented
              label={t.sim.mapModeLabel}
              size="sm"
              value={mapMode}
              onChange={setMapMode}
              options={[
                { value: 'start', label: t.sim.pickStart, testId: 'map-mode-start' },
                { value: 'destination', label: t.sim.pickDestination, testId: 'map-mode-destination' },
              ]}
            />
          ) : undefined
        }
      />
      <div className="mt-4">
        {graph ? (
          <>
            <RoadMap
              graph={graph}
              startId={showRunOnMap ? run.start_node : start}
              destinationId={showRunOnMap ? run.destination_node : destination}
              startCandidates={startCandidates}
              destinationCandidates={showRunOnMap ? reachableFrom(graph, run.start_node) : destinationCandidates}
              mode={mapMode}
              onPick={pickNode}
              route={showRunOnMap ? run.route.polyline : null}
              ego={run?.live?.ego ?? null}
              interactive={!showRunOnMap && setupEditable}
            />
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <MapLegend />
              <p className="num text-xs text-muted" data-testid="route-info">
                {showRunOnMap ? t.sim.routeLength(f.num(run.route.length_m, 1)) : t.sim.routePending}
              </p>
            </div>
          </>
        ) : simulators.error ? null : simulators.loading || !session ? (
          session ? <LoadingBlock /> : <SessionPending />
        ) : (
          <EmptyState title={t.sim.noGraph} />
        )}
      </div>
    </Card>
  );

  const setupCard = (
    <Card className="card-pad" testId="sim-setup">
      <CardHeader title={t.sim.setupTitle} icon={<Settings2 className="size-5" />} />
      <div className="mt-4 space-y-4">
        {simulators.error && !simulators.data ? (
          <ErrorNotice error={simulators.error} onRetry={simulators.reload} compact testId="simulators-error" />
        ) : !simulators.data ? (
          session ? <LoadingBlock /> : <SessionPending />
        ) : (
          <>
            <label className="block text-sm">
              <span className="mb-1 block font-medium">{t.sim.simulator}</span>
              <select
                className={selectClass}
                value={simulator?.id ?? ''}
                onChange={(event) => setSimChoice(event.target.value as SimulatorId)}
                disabled={!setupEditable}
                data-testid="sim-simulator"
              >
                {simList.map((sim) => (
                  <option key={sim.id} value={sim.id} disabled={!sim.available}>
                    {sim.name} {sim.available ? '' : t.sim.unavailableSuffix}
                  </option>
                ))}
              </select>
              {simList
                .filter((sim) => !sim.available)
                .map((sim) => (
                  <span key={sim.id} className="mt-1 block text-xs text-muted">
                    {sim.name}: {unavailableText(t, sim.reason)}
                  </span>
                ))}
            </label>
            <label className="block text-sm">
              <span className="mb-1 block font-medium">{t.sim.perception}</span>
              <select
                className={selectClass}
                value={mode?.id ?? ''}
                onChange={(event) => setModeChoice(event.target.value as SimPerceptionSource)}
                disabled={!setupEditable}
                data-testid="sim-perception"
              >
                {modes.map((item) => (
                  <option key={item.id} value={item.id} disabled={!item.available}>
                    {t.sources.simModes[item.id]} {item.available ? '' : t.sim.unavailableSuffix}
                  </option>
                ))}
              </select>
              {modes
                .filter((item) => !item.available)
                .map((item) => (
                  <span key={item.id} className="mt-1 block text-xs text-muted" data-testid={`mode-reason-${item.id}`}>
                    <code className="font-mono">{item.id}</code>: {unavailableText(t, item.reason)}
                  </span>
                ))}
            </label>
            <label className="block text-sm">
              <span className="mb-1 block font-medium">{t.sim.scenario}</span>
              <select
                className={selectClass}
                value={scenario?.id ?? ''}
                onChange={(event) => setScenarioChoice(event.target.value as ScenarioId)}
                disabled={!setupEditable}
                data-testid="sim-scenario"
              >
                {scenarios.map((item) => (
                  <option key={item.id} value={item.id} disabled={!item.available} title={item.reason ?? undefined}>
                    {t.sim.scenarios[item.id] ?? item.id} {item.available ? '' : t.sim.unavailableSuffix}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              <span className="mb-1 block font-medium">{t.sim.seed}</span>
              <input
                type="number"
                inputMode="numeric"
                min={0}
                max={SEED_MAX}
                step={1}
                className={selectClass}
                value={seedText}
                onChange={(event) => setSeedText(event.target.value)}
                disabled={!setupEditable}
                aria-invalid={seed === null}
                aria-describedby="sim-seed-hint"
                data-testid="sim-seed"
              />
              <span id="sim-seed-hint" className={`mt-1 block text-xs ${seed === null ? 'text-danger' : 'text-muted'}`}>
                {seed === null ? t.sim.seedInvalid : t.sim.seedHint}
              </span>
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="block text-sm">
                <span className="mb-1 block font-medium">{t.sim.start}</span>
                <select
                  className={selectClass}
                  value={start ?? ''}
                  onChange={(event) => {
                    setStartChoice(event.target.value || null);
                    if (destination === event.target.value) setDestination(null);
                  }}
                  disabled={!setupEditable || !graph}
                  data-testid="sim-start-node"
                >
                  <option value="">{t.sim.choose}</option>
                  {graph?.nodes.map((node) => (
                    <option key={node.id} value={node.id} disabled={!startCandidates.has(node.id)}>
                      {lang === 'uz' ? node.label_uz : node.label_en}
                      {!startCandidates.has(node.id) ? ` ${t.sim.unavailableSuffix}` : ''}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-sm">
                <span className="mb-1 block font-medium">{t.sim.destination}</span>
                <select
                  className={selectClass}
                  value={destination ?? ''}
                  onChange={(event) => setDestination(event.target.value || null)}
                  disabled={!setupEditable || !graph}
                  data-testid="sim-destination-node"
                >
                  <option value="">{t.sim.choose}</option>
                  {graph?.nodes.map((node) => (
                    <option key={node.id} value={node.id} disabled={!destinationCandidates.has(node.id)}>
                      {lang === 'uz' ? node.label_uz : node.label_en}
                      {!destinationCandidates.has(node.id) && node.id !== start ? ` — ${t.sim.nodeUnreachable}` : ''}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </>
        )}
      </div>
    </Card>
  );

  const runInfoCard = run && (
    <Card className="card-pad" testId="sim-run-info">
      <CardHeader title={t.sim.runLabel(t.sim.scenarios[run.scenario] ?? run.scenario, run.seed)} />
      <KeyValueList
        className="mt-3"
        items={[
          { label: t.sim.simulator, value: run.simulator },
          { label: t.sim.perception, value: <SourcePill source={run.perception_source} /> },
          { label: t.sim.start, value: nodeLabel(run.start_node) },
          { label: t.sim.destination, value: nodeLabel(run.destination_node) },
          { label: t.common.created, value: f.dateTime(run.created_utc) },
          { label: t.common.finished, value: f.dateTime(run.finished_utc) },
        ]}
      />
      {run.error_code && (
        <div className="mt-3">
          <Notice tone="warning" title={t.sim.runError}>
            <code>{run.error_code}</code> {t.errors[run.error_code as keyof typeof t.errors] ?? ''}
            {run.error_message && <span className="block text-xs text-muted">{run.error_message}</span>}
          </Notice>
        </div>
      )}
    </Card>
  );

  const runsCard = (
    <Card className="card-pad" testId="sim-runs">
      <CardHeader title={t.sim.runsTitle} icon={<History className="size-5" />} />
      <div className="mt-3">
        {runs.error && !runs.data ? (
          <ErrorNotice error={runs.error} onRetry={runs.reload} compact />
        ) : !runs.data ? (
          session ? <LoadingBlock /> : <SessionPending />
        ) : runs.data.items.length === 0 ? (
          <EmptyState title={t.sim.runsEmpty} />
        ) : (
          <ul className="divide-y divide-line">
            {runs.data.items.map((item) => (
              <li key={item.id}>
                <Link
                  to={`/simulation/runs/${item.id}`}
                  aria-current={item.id === runId ? 'true' : undefined}
                  className={`flex items-center justify-between gap-2 py-2 text-sm hover:text-accent ${item.id === runId ? 'font-semibold' : ''}`}
                  data-testid="sim-run-item"
                >
                  <span className="min-w-0 truncate">
                    {t.sim.runLabel(t.sim.scenarios[item.scenario] ?? item.scenario, item.seed)}
                    <span className="font-normal text-muted"> · {f.time(item.created_utc)}</span>
                  </span>
                  <Pill tone={RUN_TONE[item.state]} dot>
                    {t.states.run[item.state]}
                  </Pill>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );

  return (
    <div data-testid="page-simulation">
      <PageHeader title={t.sim.title} subtitle={t.sim.subtitle} />
      {runId && live.error && !run && (
        <div className="mb-6">
          <ErrorNotice error={live.error} onRetry={live.refresh} testId="run-error" />
        </div>
      )}
      <div className="grid gap-6 lg:grid-cols-12">
        <div className="space-y-6 lg:col-span-8">
          {cameraCard}
          {mapCard}
        </div>
        <div className="space-y-6 lg:col-span-4">
          {run && <DecisionPanel run={run} live={shownLive} cameraFrameId={cameraFrameId} />}
          {run && <ManeuverHistory items={historyItems} />}
          {run && runIsActive && <ObstaclePanel run={run} maxPerRun={limits?.max_obstacles_per_run ?? null} />}
          {!runId && setupCard}
          {runInfoCard}
          {runsCard}
        </div>
      </div>
      {run && !runIsActive && (
        <div className="mt-6">
          <RunReportView run={run} />
        </div>
      )}
    </div>
  );
}
