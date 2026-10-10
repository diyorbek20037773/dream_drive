import { BrainCircuit, ShieldAlert } from 'lucide-react';

import type { QualityStatus, Run, RunLive } from '../../api/types';
import { ReasonList, SourcePill } from '../../components/domain';
import { Card, CardHeader } from '../../components/ui/Card';
import { Pill, ProgressBar } from '../../components/ui/Status';
import { useI18n } from '../../i18n';
import { useFormat } from '../../lib/format';
import type { Tone } from '../../lib/tones';
import { DECISION_TONE } from '../../lib/tones';
import { DecisionBasisList, ManeuverIndicator, ManeuverSignals } from './Maneuver';

const QUALITY_TONE: Record<QualityStatus, Tone> = {
  valid: 'success',
  degraded: 'warning',
  missing: 'danger',
  invalid: 'danger',
  stale: 'danger',
};

/** Signed bar for steering in [-1, 1]; +1 is full left (docs/IMPLEMENTATION_SPEC.md section 4). */
function SteerBar({ value }: { value: number }) {
  const { t } = useI18n();
  const clamped = Math.max(-1, Math.min(1, value));
  const pct = Math.abs(clamped) * 50;
  return (
    <div>
      <div className="flex justify-between text-[11px] text-muted">
        <span>← {t.sim.left}</span>
        <span>{t.sim.right} →</span>
      </div>
      <div
        className="relative mt-1 h-2.5 w-full rounded-full bg-line"
        role="meter"
        aria-label={t.sim.steer}
        aria-valuemin={-1}
        aria-valuemax={1}
        aria-valuenow={clamped}
        data-testid="cmd-steer"
      >
        <span className="absolute inset-y-[-3px] left-1/2 w-px bg-muted" aria-hidden />
        <span
          className="absolute inset-y-0 rounded-full bg-accent"
          style={clamped >= 0 ? { right: '50%', width: `${pct}%` } : { left: '50%', width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function UnitBar({ value, label, tone, testId }: { value: number; label: string; tone: 'info' | 'danger'; testId: string }) {
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div
      className="h-2.5 w-full overflow-hidden rounded-full bg-line"
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={clamped}
      data-testid={testId}
    >
      <div className={`h-full rounded-full ${tone === 'danger' ? 'bg-danger' : 'bg-success'}`} style={{ width: `${clamped * 100}%` }} />
    </div>
  );
}

/**
 * Right-hand decision panel. `live` is the Run.live snapshot of the frame the camera currently shows (falls back to
 * the latest snapshot), so the maneuver, decision basis, command and telemetry all describe that same frame.
 */
export function DecisionPanel({ run, live: shownLive, cameraFrameId }: { run: Run; live?: RunLive | null; cameraFrameId?: number | null }) {
  const { t } = useI18n();
  const f = useFormat();
  const live = shownLive ?? run.live;
  return (
    <Card className="card-pad" testId="decision-panel">
      <CardHeader title={t.sim.decisionTitle} icon={<BrainCircuit className="size-5" />} />
      {!live ? (
        <p className="mt-4 text-sm text-muted">{t.sim.noLive}</p>
      ) : (
        <div className="mt-4 space-y-5" data-frame-id={live.frame_id}>
          {live.maneuver && (
            <div className="space-y-1.5">
              <ManeuverIndicator maneuver={live.maneuver} frameId={live.frame_id} testId="maneuver-indicator" />
              <ManeuverSignals maneuver={live.maneuver} />
              <p className="text-xs text-muted">{t.maneuver.describes}</p>
            </div>
          )}

          <DecisionBasisList items={live.decision_basis} frameId={live.frame_id} cameraFrameId={cameraFrameId ?? null} />

          <div aria-live="polite">
            <p className="label-caps mb-1.5">{t.sim.fsmState}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone={DECISION_TONE[live.decision.state]} dot testId="fsm-state">
                {t.states.decision[live.decision.state]}
              </Pill>
              <code className="font-mono text-xs text-muted">{live.decision.state}</code>
            </div>
            <div className="mt-2">
              <ReasonList codes={live.decision.reason_codes} testId="fsm-reasons" />
            </div>
          </div>

          <div>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <p className="label-caps">{t.sim.command}</p>
              {live.command && (
                <span className="num text-xs text-muted">{t.sim.inputFrame(live.command.input_frame_id)}</span>
              )}
            </div>
            {!live.command ? (
              <p className="text-sm text-muted">{t.sim.noCommand}</p>
            ) : (
              <div className="space-y-3 text-sm">
                <div>
                  <div className="flex justify-between">
                    <span>{t.sim.steer}</span>
                    <span className="num font-medium">{f.num(live.command.steer, 2)}</span>
                  </div>
                  <SteerBar value={live.command.steer} />
                </div>
                <div>
                  <div className="flex justify-between">
                    <span>{t.sim.throttle}</span>
                    <span className="num font-medium">{f.num(live.command.throttle, 2)}</span>
                  </div>
                  <UnitBar value={live.command.throttle} label={t.sim.throttle} tone="info" testId="cmd-throttle" />
                </div>
                <div>
                  <div className="flex justify-between">
                    <span>{t.sim.brake}</span>
                    <span className="num font-medium">{f.num(live.command.brake, 2)}</span>
                  </div>
                  <UnitBar value={live.command.brake} label={t.sim.brake} tone="danger" testId="cmd-brake" />
                </div>
                {live.command.safety_override && (
                  <p className="flex items-center gap-1.5 text-sm font-medium text-danger" data-testid="cmd-safety-override">
                    <ShieldAlert className="size-4" aria-hidden />
                    {t.sim.safetyOverride}
                  </p>
                )}
                <p className="text-xs text-muted">
                  {t.common.source}: <code className="font-mono">{live.command.source}</code> — {t.sources.controller}
                </p>
              </div>
            )}
          </div>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
            <div>
              <dt className="text-muted">{t.sim.speed}</dt>
              <dd className="num font-semibold" data-testid="ego-speed">
                {f.num(live.ego.speed_mps, 2)} {t.units.mps}
                <span className="ml-1 font-normal text-muted">({f.num(live.ego.speed_mps * 3.6, 1)} {t.units.kmh})</span>
              </dd>
            </div>
            <div>
              <dt className="text-muted">{t.sim.simTime}</dt>
              <dd className="num font-semibold" data-testid="sim-time">
                {f.num(live.sim_time_s, 2)} {t.units.s}
              </dd>
            </div>
            <div>
              <dt className="text-muted">{t.sim.wallTime}</dt>
              <dd className="num font-semibold">
                {f.num(live.wall_time_s, 2)} {t.units.s}
              </dd>
            </div>
            <div>
              <dt className="text-muted">{t.sim.rtf}</dt>
              <dd className="num font-semibold">{f.num(live.realtime_factor, 2)}</dd>
            </div>
          </dl>

          <div>
            <div className="mb-1 flex justify-between text-sm">
              <span className="text-muted">{t.sim.routeProgress}</span>
              <span className="num font-medium">{f.pct(live.route_progress, 0)}</span>
            </div>
            <ProgressBar value={live.route_progress} label={t.sim.routeProgress} tone="success" />
          </div>

          <div>
            <p className="label-caps mb-2">{t.sim.acquisition}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone={QUALITY_TONE[live.perception.quality_status]} dot testId="acq-quality">
                {t.sim.quality}: {t.states.quality[live.perception.quality_status]}
              </Pill>
              <SourcePill source={live.perception.source} />
            </div>
            <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <div>
                <dt className="text-muted">{t.sim.acquisitionAge}</dt>
                <dd className="num font-medium">
                  {live.acquisition_age_s === null ? t.common.notMeasured : `${f.num(live.acquisition_age_s, 3)} ${t.units.s}`}
                </dd>
              </div>
              <div>
                <dt className="text-muted">{t.sim.lane}</dt>
                <dd className="font-medium">{t.states.lane[live.perception.lane_status]}</dd>
              </div>
              <div>
                <dt className="text-muted">{t.sim.laneOffset}</dt>
                <dd className="num font-medium">{f.num(live.perception.image_lane_offset_normalized, 2)}</dd>
              </div>
              <div>
                <dt className="text-muted">{t.sim.obstacleCue}</dt>
                <dd className="num font-medium">{f.num(live.perception.image_obstacle_cue, 2)}</dd>
              </div>
            </dl>
            <p className="mt-2 text-xs text-muted">
              {t.sources.egoTelemetry}: <code className="font-mono">{live.ego.source}</code> · x {f.num(live.ego.x_m, 1)} m · y{' '}
              {f.num(live.ego.y_m, 1)} m · yaw {f.num(live.ego.yaw_rad, 2)} rad
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}
