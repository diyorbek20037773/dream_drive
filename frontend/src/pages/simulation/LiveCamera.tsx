import { Camera } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import type { ApiError } from '../../api/client';
import { fetchRunFrame, toApiError } from '../../api/client';
import type { RunLive } from '../../api/types';
import { useI18n } from '../../i18n';
import { errorText, useFormat } from '../../lib/format';

interface ShownFrame {
  url: string;
  frameId: number | null;
  simTimeS: number | null;
  source: string | null;
}

/**
 * Latest actual simulator camera JPEG for a run. A new fetch is triggered whenever `frameTick` changes (frame_ready
 * events or polling); concurrent triggers are coalesced. The overlay is drawn only from the Run.live snapshot whose
 * frame_id equals the X-Frame-Id of the displayed image, so geometry never mixes two different frames.
 */
export function LiveCamera({
  runId,
  frameTick,
  lives,
  active,
  hasRun,
  onShownFrame,
}: {
  runId: string | null;
  frameTick: number;
  lives: RunLive[];
  active: boolean;
  hasRun: boolean;
  /** Reports the X-Frame-Id of the displayed image so side panels can show the same frame's decision. */
  onShownFrame?: (frameId: number | null) => void;
}) {
  const { t } = useI18n();
  const f = useFormat();
  const [shown, setShown] = useState<ShownFrame | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const inFlight = useRef(false);
  const pending = useRef(false);
  const latestTick = useRef(frameTick);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const shownFrameId = shown?.frameId ?? null;
  useEffect(() => {
    onShownFrame?.(shownFrameId);
  }, [shownFrameId, onShownFrame]);

  // Revoke the previous object URL whenever the shown frame changes (and on unmount).
  useEffect(() => {
    const url = shown?.url;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [shown?.url]);

  useEffect(() => {
    latestTick.current = frameTick;
    if (!runId) return;
    const load = () => {
      if (inFlight.current) {
        pending.current = true;
        return;
      }
      inFlight.current = true;
      fetchRunFrame(runId, latestTick.current)
        .then((frame) => {
          if (!mounted.current) return;
          setError(null);
          if (!frame) return;
          setShown({
            url: URL.createObjectURL(frame.blob),
            frameId: frame.frameId,
            simTimeS: frame.simTimeS,
            source: frame.perceptionSource,
          });
        })
        .catch((reason: unknown) => {
          if (mounted.current) setError(toApiError(reason));
        })
        .finally(() => {
          inFlight.current = false;
          if (pending.current && mounted.current) {
            pending.current = false;
            load();
          }
        });
    };
    load();
  }, [runId, frameTick]);

  const live = shown?.frameId !== null && shown?.frameId !== undefined ? lives.find((item) => item.frame_id === shown.frameId) : undefined;
  const perception = live?.perception;
  const width = perception?.frame_width ?? 640;
  const height = perception?.frame_height ?? 360;
  const source = shown?.source ?? perception?.source ?? null;

  return (
    <div className="relative w-full overflow-hidden bg-night" style={{ aspectRatio: `${width} / ${height}` }} data-on-dark data-testid="sim-camera">
      {shown ? (
        <img
          src={shown.url}
          alt={t.sim.cameraAlt(shown.frameId ?? 0)}
          className="absolute inset-0 h-full w-full object-contain"
          data-testid="sim-camera-img"
          data-frame-id={shown.frameId ?? ''}
        />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-white/75">
          <Camera className="size-8 text-white/40" aria-hidden />
          {hasRun ? (active ? t.sim.cameraWaiting : t.sim.cameraEnded) : t.sim.cameraIdle}
        </div>
      )}

      {shown && showOverlay && perception && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMidYMid meet"
          aria-hidden
          data-testid="sim-camera-overlay"
          data-frame-id={live?.frame_id}
        >
          {perception.lane_points.length > 1 && (
            <polyline
              points={perception.lane_points.map((point) => `${point[0] ?? 0},${point[1] ?? 0}`).join(' ')}
              fill="none"
              stroke="#22d3ee"
              strokeWidth={2}
              strokeDasharray="6 4"
            />
          )}
          {perception.lane_points.map((point, index) => (
            <circle key={index} cx={point[0] ?? 0} cy={point[1] ?? 0} r={2.5} fill="#22d3ee" />
          ))}
          {perception.obstacle_boxes.map((box, index) => {
            const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
            return (
              <g key={index}>
                <rect x={x1} y={y1} width={Math.max(0, x2 - x1)} height={Math.max(0, y2 - y1)} fill="rgba(249,115,22,0.15)" stroke="#f97316" strokeWidth={2} />
                <text x={x1 + 3} y={Math.max(12, y1 - 4)} fill="#fed7aa" fontSize={12} fontWeight={600}>
                  obstacle
                </text>
              </g>
            );
          })}
        </svg>
      )}

      <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap gap-2">
        {source && (
          <span className="rounded-full bg-black/65 px-2.5 py-1 font-mono text-xs font-medium text-cyan-200 backdrop-blur" data-testid="sim-source-pill">
            {source}
            {t.sources.perception[source as keyof typeof t.sources.perception] && (
              <span className="font-sans text-white/80"> · {t.sources.perception[source as keyof typeof t.sources.perception]}</span>
            )}
          </span>
        )}
      </div>

      {shown && (
        <div className="absolute inset-x-3 bottom-3 flex flex-wrap items-end justify-between gap-2">
          <span className="num rounded-full bg-black/65 px-2.5 py-1 text-xs text-white/90" data-testid="sim-frame-label">
            {shown.frameId !== null ? t.sim.frameLabel(shown.frameId, f.num(shown.simTimeS, 2)) : '—'}
          </span>
          <span className="flex items-center gap-2">
            {showOverlay && !perception && (
              <span className="rounded-full bg-black/65 px-2.5 py-1 text-xs text-white/70">{t.sim.overlayPending}</span>
            )}
            <label className="flex items-center gap-1.5 rounded-full bg-black/65 px-2.5 py-1 text-xs text-white">
              <input type="checkbox" checked={showOverlay} onChange={(event) => setShowOverlay(event.target.checked)} data-testid="sim-overlay-toggle" />
              {t.sim.overlayToggle}
            </label>
          </span>
        </div>
      )}

      {error && (
        <div className="absolute inset-x-3 top-12 rounded-control bg-black/75 p-2.5 text-xs text-white" role="alert" data-testid="sim-frame-error">
          {t.sim.frameError}: {errorText(t, error)}
        </div>
      )}
    </div>
  );
}
