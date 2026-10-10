import { Clapperboard, ListTree, ScanSearch } from 'lucide-react';
import type { ReactNode } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';

import { api, safeRelativeUrl } from '../../api/client';
import type { DecisionState, Job, JobResults, TimelineFile, TimelineFrame } from '../../api/types';
import { ReasonList, SourcePill } from '../../components/domain';
import { Card, CardHeader } from '../../components/ui/Card';
import { Segmented } from '../../components/ui/Data';
import { ErrorNotice, LoadingBlock, Notice } from '../../components/ui/Feedback';
import { Pill } from '../../components/ui/Status';
import { useResource } from '../../hooks/useResource';
import { useI18n } from '../../i18n';
import { useFormat } from '../../lib/format';
import { RECOMMENDATION_ICON, RECOMMENDATION_TONE, recommendationFor, TONE_BOX } from '../../lib/maneuver';
import type { TimelineEvent, VideoView } from '../../lib/timeline';
import { buildEvents, eventForIndex, firstIndexAtOrAfter, indexForTime, timeForIndex } from '../../lib/timeline';
import { DECISION_FILL, DECISION_TONE } from '../../lib/tones';
import { ResultDetails } from './ResultDetails';

const MODEL_DISPLAY: Record<string, string> = {
  yolopv2: 'YOLOPv2',
  rfdetr_nano: 'RF-DETR-N',
};

/** Fallback when an older job stored no reporting threshold (DETECTION_REPORT_MIN_SCORE default). */
const DEFAULT_REPORT_MIN_SCORE = 0.4;

/** Display name of the perception identity: the timeline's member list when present (the identity every analyzed
 * frame carries), else the job's model id split into its "+"-joined members. Never hard-coded. */
function modelDisplayName(results: JobResults, timeline: TimelineFile | null | undefined): string {
  const members = timeline?.model?.models;
  if (members && members.length > 0) return members.map((m) => m.label ?? MODEL_DISPLAY[m.model_id] ?? m.model_id).join(' + ');
  const id = timeline?.model?.model_id ?? results.model?.model_id;
  if (!id) return '—';
  return id
    .split('+')
    .map((part) => MODEL_DISPLAY[part] ?? part)
    .join(' + ');
}

function overlayLabel(results: JobResults, timeline: TimelineFile | null | undefined): string {
  const device = results.model?.device ?? timeline?.model?.device;
  return `${modelDisplayName(results, timeline)} · ${results.perception_source} · ${device ? device.toUpperCase() : '—'}`;
}

/** Detection boxes of the nearest analyzed frame drawn over the original video (same pixel geometry). */
function BoxesOverlay({ frame, minScore, showLow }: { frame: TimelineFrame; minScore: number; showLow: boolean }) {
  const { width, height } = frame.frame;
  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden
      data-testid="video-boxes-overlay"
    >
      {frame.perception.detections.map((det, index) => {
        const low = det.score < minScore;
        if (low && !showLow) return null;
        const [x1, y1, x2, y2] = det.xyxy_pixels;
        const stroke = Math.max(2, width / 400);
        return (
          <g key={index} data-low-confidence={low ? 'true' : undefined} opacity={low ? 0.7 : 1}>
            <rect
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill="none"
              stroke={low ? '#a3a3a3' : '#f97316'}
              strokeWidth={stroke}
              strokeDasharray={low ? `${stroke * 3} ${stroke * 2}` : undefined}
            />
            <text x={x1 + stroke} y={Math.max(y1 - stroke * 2, stroke * 8)} fill="#fff7ed" fontSize={Math.max(12, width / 70)} fontWeight={600} paintOrder="stroke" stroke="#7c2d12" strokeWidth={stroke}>
              {det.class_name}
              {det.track_id !== null ? ` #${det.track_id}` : ''} {det.score.toFixed(2)}
              {det.light_state ? ` · ${det.light_state} (HSV)` : ''}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function Scrubber({
  timeline,
  events,
  playhead,
  onSeek,
}: {
  timeline: TimelineFile;
  events: TimelineEvent[];
  playhead: number;
  onSeek: (index: number) => void;
}) {
  const { t } = useI18n();
  const f = useFormat();
  const count = timeline.frames.length;
  const barRef = useRef<HTMLDivElement>(null);
  const current = timeline.frames[playhead];
  const valueText = current
    ? `${t.video.frameAt(playhead, f.num(current.pts_s, 2))} · ${t.states.decision[current.decision.state]}`
    : '';
  return (
    <div className="space-y-2 px-4 pt-3 pb-4 sm:px-5" data-testid="timeline-scrubber">
      {/* Decision segments; pointer users click anywhere, keyboard users use the slider below. */}
      <div
        ref={barRef}
        className="relative flex h-3 w-full cursor-pointer overflow-hidden rounded-full bg-line"
        aria-hidden
        onClick={(event) => {
          const rect = barRef.current?.getBoundingClientRect();
          if (!rect || count === 0) return;
          const ratio = (event.clientX - rect.left) / rect.width;
          onSeek(Math.min(count - 1, Math.max(0, Math.floor(ratio * count))));
        }}
      >
        {events.map((event) => (
          <div
            key={event.id}
            className="h-full"
            style={{
              width: `${((event.endIndex - event.startIndex + 1) / count) * 100}%`,
              backgroundColor: DECISION_FILL[event.state],
            }}
            title={`${f.num(event.startPts, 2)} s · ${t.states.decision[event.state]}`}
          />
        ))}
        <div
          className="pointer-events-none absolute inset-y-[-2px] w-0.5 rounded bg-ink"
          style={{ left: `${((playhead + 0.5) / Math.max(1, count)) * 100}%` }}
        />
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(0, count - 1)}
        step={1}
        value={playhead}
        onChange={(event) => onSeek(Number(event.target.value))}
        aria-label={t.video.frameSlider}
        aria-valuetext={valueText}
        className="w-full"
        data-testid="frame-slider"
      />
      <p className="num text-xs text-muted" data-testid="playhead-label" data-frame-index={playhead}>
        {valueText}
      </p>
    </div>
  );
}

/**
 * Analysis-only recommendation derived from the analytical decision state (observe -> continue, slow -> slow down,
 * controlled_stop -> stop). The uploaded video is an observation recording: nothing is controlled.
 */
function Recommendation({ state, reasons, compact = false }: { state: DecisionState; reasons?: readonly string[]; compact?: boolean }) {
  const { t } = useI18n();
  const rec = recommendationFor(state);
  if (!rec) return <span className="text-sm text-muted">—</span>;
  const Icon = RECOMMENDATION_ICON[rec];
  if (compact) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-semibold whitespace-nowrap ${TONE_BOX[RECOMMENDATION_TONE[rec]]}`}
        data-testid="event-recommendation"
        data-recommendation={rec}
      >
        <Icon className="size-3.5" strokeWidth={2.5} aria-hidden />
        {t.maneuver.recommendation[rec]}
      </span>
    );
  }
  return (
    <div className={`rounded-card border p-3 ${TONE_BOX[RECOMMENDATION_TONE[rec]]}`} data-testid="current-recommendation" data-recommendation={rec} role="status" aria-live="polite" aria-atomic="true">
      <div className="flex items-center gap-3">
        <span className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-surface/80" aria-hidden>
          <Icon className="size-7" strokeWidth={2.5} />
        </span>
        <div className="min-w-0">
          <p className="text-[11px] font-semibold tracking-wide text-muted uppercase">{t.maneuver.recommended}</p>
          <p className="text-xl leading-tight font-semibold" data-testid="current-recommendation-label">
            {t.maneuver.recommendation[rec]}
          </p>
        </div>
      </div>
      {reasons && reasons.length > 0 && (
        <div className="mt-3 text-ink">
          <p className="mb-1 text-xs font-semibold text-muted">{t.maneuver.recommendationBasis}</p>
          <ReasonList codes={reasons} dense testId="current-decision-reasons" />
        </div>
      )}
      <p className="mt-2 text-xs text-muted" data-testid="no-vehicle-controlled">
        {t.maneuver.noVehicleControlled}
      </p>
    </div>
  );
}

function CurrentDecision({
  frame,
  index,
  minScore,
  showLow,
  onShowLow,
}: {
  frame: TimelineFrame | undefined;
  index: number;
  minScore: number;
  showLow: boolean;
  onShowLow: (value: boolean) => void;
}) {
  const { t } = useI18n();
  const f = useFormat();
  if (!frame) return null;
  const counts = new Map<string, { label: string; count: number; max: number; low: boolean }>();
  let hiddenLow = 0;
  for (const det of frame.perception.detections) {
    const low = det.score < minScore;
    if (low && !showLow) {
      hiddenLow += 1;
      continue;
    }
    // Heuristic traffic-light colour is shown separately from the detector class, never merged into it.
    const label = det.light_state ? `${det.class_name} · ${det.light_state} (${t.video.heuristic})` : det.class_name;
    const key = `${low ? 'low' : 'ok'}:${label}`;
    const entry = counts.get(key) ?? { label, count: 0, max: 0, low };
    entry.count += 1;
    entry.max = Math.max(entry.max, det.score);
    counts.set(key, entry);
  }
  const inferenceMs = frame.timing_ms?.inference ?? frame.timing_ms?.total ?? null;
  return (
    <Card className="card-pad" testId="current-decision">
      <CardHeader title={t.video.currentDecision} icon={<ScanSearch className="size-5" />} subtitle={t.video.frameAt(index, f.num(frame.pts_s, 2))} />
      <div className="mt-4 space-y-4">
        <Recommendation state={frame.decision.state} reasons={frame.decision.reason_codes} />
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone={DECISION_TONE[frame.decision.state]} dot testId="current-decision-state">
            {t.states.decision[frame.decision.state]}
          </Pill>
          <SourcePill source={frame.perception.source} extra={frame.perception.model_id ?? undefined} />
        </div>
        <dl className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <dt className="text-muted">{t.video.lane}</dt>
            <dd className="font-medium">{t.states.lane[frame.perception.lane_status]}</dd>
          </div>
          <div>
            <dt className="text-muted">{t.video.quality}</dt>
            <dd className="font-medium">{t.states.quality[frame.frame.quality_status]}</dd>
          </div>
          <div>
            <dt className="text-muted">{t.video.obstacleCue}</dt>
            <dd className="num font-medium">{f.num(frame.perception.image_obstacle_cue, 2)}</dd>
          </div>
          <div>
            <dt className="text-muted">{t.video.inference}</dt>
            <dd className="num font-medium">{inferenceMs === null ? t.common.notMeasured : `${f.num(inferenceMs, 0)} ${t.units.ms}`}</dd>
          </div>
        </dl>
        <div>
          <p className="label-caps mb-1.5">{t.video.detections}</p>
          {counts.size === 0 ? (
            <p className="text-sm text-muted" data-testid="no-detections">
              {t.video.noDetections}
            </p>
          ) : (
            <ul className="flex flex-wrap gap-1.5" data-testid="detection-chips">
              {[...counts.entries()].map(([key, entry]) => (
                <li key={key} data-testid="detection-chip" data-low-confidence={entry.low ? 'true' : 'false'}>
                  <Pill tone={entry.low ? 'warning' : 'neutral'}>
                    {entry.label} × {entry.count} · max {entry.max.toFixed(2)}
                    {entry.low ? ` · ${t.video.lowConfidence}` : ''}
                  </Pill>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-1.5 text-xs text-muted" data-testid="report-threshold-note">
            {t.video.reportThreshold(minScore.toFixed(2))}
            {!showLow && hiddenLow > 0 ? ` ${t.video.hiddenLowConfidence(hiddenLow)}.` : ''}
          </p>
          <label className="mt-1.5 inline-flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={showLow} onChange={(e) => onShowLow(e.target.checked)} data-testid="toggle-low-confidence" />
            {t.video.lowConfidenceToggle}
          </label>
        </div>
        <p className="text-xs text-muted">{t.video.analyticalOnly}</p>
      </div>
    </Card>
  );
}

function EventsTable({
  timeline,
  events,
  activeEvent,
  playing,
  onSeek,
}: {
  timeline: TimelineFile;
  events: TimelineEvent[];
  activeEvent: number;
  playing: boolean;
  onSeek: (index: number) => void;
}) {
  const { t } = useI18n();
  const f = useFormat();
  const containerRef = useRef<HTMLDivElement>(null);

  // Keep the highlighted row visible during playback without scrolling the page itself.
  useEffect(() => {
    if (!playing || activeEvent < 0) return;
    const container = containerRef.current;
    const row = container?.querySelector<HTMLElement>(`[data-event-id="${activeEvent}"]`);
    if (!container || !row) return;
    const top = row.offsetTop;
    if (top < container.scrollTop || top + row.offsetHeight > container.scrollTop + container.clientHeight) {
      container.scrollTop = Math.max(0, top - container.clientHeight / 3);
    }
  }, [activeEvent, playing]);

  return (
    <Card className="card-pad" testId="timeline-events">
      <CardHeader title={t.video.timelineTitle} icon={<ListTree className="size-5" />} subtitle={t.video.timelineHint} />
      <p className="mt-2 text-xs text-muted" data-testid="timeline-no-vehicle">
        {t.maneuver.noVehicleControlled}
      </p>
      <div ref={containerRef} className="scroll-x relative mt-4 max-h-[420px] overflow-y-auto rounded-control border border-line">
        <table className="w-full text-left text-sm sm:min-w-[640px]">
          <thead className="sticky top-0 z-10 bg-surface">
            <tr className="border-b border-line text-xs text-muted">
              <th scope="col" className="px-3 py-2 font-medium">{t.video.colTime}</th>
              <th scope="col" className="px-3 py-2 font-medium">
                {t.video.colState}
                <span className="sm:hidden"> · {t.video.colReasons}</span>
              </th>
              <th scope="col" className="px-3 py-2 font-medium">{t.maneuver.recommendedShort}</th>
              <th scope="col" className="hidden px-3 py-2 font-medium sm:table-cell">{t.video.colReasons}</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">{t.video.colFrames}</th>
            </tr>
          </thead>
          <tbody>
            {events.map((event) => {
              const active = event.id === activeEvent;
              return (
                <tr
                  key={event.id}
                  data-event-id={event.id}
                  data-testid="timeline-event"
                  aria-current={active ? 'true' : undefined}
                  className={`border-b border-line/70 align-top last:border-0 ${active ? 'bg-accent/8' : 'hover:bg-canvas'}`}
                >
                  <td className="px-3 py-2">
                    <button
                      type="button"
                      className="num font-mono text-[13px] font-medium whitespace-nowrap text-accent-strong hover:underline"
                      onClick={() => onSeek(event.startIndex)}
                      data-testid="timeline-event-seek"
                      data-pts={event.startPts}
                      data-frame-index={event.startIndex}
                    >
                      {f.num(event.startPts, 2)}
                      {event.endPts > event.startPts && `–${f.num(event.endPts, 2)}`} s
                    </button>
                  </td>
                  <td className="px-3 py-2">
                    <Pill tone={DECISION_TONE[event.state]} dot>
                      {t.states.decision[event.state]}
                    </Pill>
                    {/* Phones: reasons stacked under the state instead of a separate, off-screen column. */}
                    <div className="mt-1.5 sm:hidden">
                      <ReasonList codes={event.reasons} dense />
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <Recommendation state={event.state} compact />
                  </td>
                  <td className="hidden px-3 py-2 sm:table-cell">
                    <ReasonList codes={event.reasons} dense />
                  </td>
                  <td className="num px-3 py-2 text-right text-muted">{event.endIndex - event.startIndex + 1}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-muted">
        {timeline.frames.length} {t.units.frames} · {timeline.analysis_fps} {t.units.fps}
      </p>
    </Card>
  );
}

export function ResultWorkspace({ job, results, side }: { job: Job; results: JobResults; side: ReactNode }) {
  const { t } = useI18n();
  const timelineUrl = safeRelativeUrl(results.timeline_url);
  const timelineRes = useResource(timelineUrl ? `timeline:${job.id}` : null, (signal) => api.getTimeline(timelineUrl ?? '', signal));
  const timeline = timelineRes.data;
  const events = useMemo(() => (timeline ? buildEvents(timeline.frames) : []), [timeline]);

  const artifact = (name: string) => results.artifacts.find((item) => item.name === name);
  const annotatedUrl = safeRelativeUrl(artifact('annotated.mp4')?.url);
  const originalUrl = safeRelativeUrl(artifact('input')?.url);
  const [view, setView] = useState<VideoView>(annotatedUrl ? 'annotated' : 'original');
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [videoError, setVideoError] = useState<VideoView | null>(null);
  const [showBoxes, setShowBoxes] = useState(false);
  const [showLow, setShowLow] = useState(false);
  const minScore = results.summary.detection_report_min_score ?? DEFAULT_REPORT_MIN_SCORE;
  const videoRef = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);

  const src = view === 'annotated' ? annotatedUrl : originalUrl;
  const activeEvent = eventForIndex(events, playhead);
  const currentFrame = timeline?.frames[playhead];

  const seekTo = (index: number) => {
    setPlayhead(index);
    const video = videoRef.current;
    if (!video || !timeline) return;
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) video.currentTime = timeForIndex(timeline, index, view);
    else pendingSeek.current = index;
  };
  const syncFromVideo = () => {
    const video = videoRef.current;
    if (video && timeline) setPlayhead(indexForTime(timeline, video.currentTime, view));
  };
  const switchView = (next: VideoView) => {
    if (next === view) return;
    pendingSeek.current = playhead;
    setPlaying(false);
    setView(next);
  };

  const viewOptions = [
    { value: 'annotated' as const, label: t.video.annotated, disabled: !annotatedUrl, testId: 'view-annotated' },
    { value: 'original' as const, label: t.video.original, disabled: !originalUrl, testId: 'view-original' },
  ];

  return (
    // Phones: video → current decision → timeline → results → job list. Desktop: 8/4 split, then full-width rows.
    // The right column uses `display: contents` below lg so its cards can be re-ordered individually.
    <div className="flex flex-col gap-6 lg:grid lg:grid-cols-12 lg:items-start">
        <div className="order-1 lg:order-none lg:col-span-8">
          <Card className="overflow-hidden" testId="result-player">
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-5">
              <div className="flex items-center gap-2">
                <Clapperboard className="size-5 text-muted" aria-hidden />
                <h2 className="text-base font-semibold">{t.video.title}</h2>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Segmented label={t.video.viewLabel} value={view} options={viewOptions} onChange={switchView} size="sm" />
                {view === 'original' && timeline && (
                  <label className="inline-flex items-center gap-2 text-xs text-muted">
                    <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} data-testid="toggle-boxes" />
                    {t.video.boxesToggle}
                  </label>
                )}
              </div>
            </div>
            <div className="relative aspect-video w-full bg-black" data-on-dark>
              {src ? (
                // The analysis outputs carry no audio track (audio is removed), so there is nothing to caption.
                // eslint-disable-next-line jsx-a11y/media-has-caption
                <video
                  key={view}
                  ref={videoRef}
                  src={src}
                  controls
                  preload="metadata"
                  playsInline
                  className="absolute inset-0 h-full w-full object-contain"
                  data-testid="video-player"
                  data-view={view}
                  onLoadedMetadata={() => {
                    setVideoError(null);
                    if (pendingSeek.current !== null && timeline && videoRef.current) {
                      videoRef.current.currentTime = timeForIndex(timeline, pendingSeek.current, view);
                      pendingSeek.current = null;
                    }
                  }}
                  onTimeUpdate={syncFromVideo}
                  onSeeked={syncFromVideo}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onEnded={() => setPlaying(false)}
                  onError={() => setVideoError(view)}
                />
              ) : (
                <p className="absolute inset-0 flex items-center justify-center p-6 text-sm text-white/80">{t.video.artifactMissing}</p>
              )}
              {view === 'original' && showBoxes && currentFrame && <BoxesOverlay frame={currentFrame} minScore={minScore} showLow={showLow} />}
              {/* Top-right: the annotated frames carry their own HUD text burned into the top-left corner. */}
              <div className="pointer-events-none absolute top-3 right-3 flex max-w-[70%] flex-wrap justify-end gap-2">
                <span
                  className="rounded-full bg-black/65 px-2.5 py-1 font-mono text-xs font-medium text-white backdrop-blur"
                  data-testid="video-source-label"
                >
                  {view === 'annotated' ? overlayLabel(results, timeline) : t.video.originalLabel}
                </span>
              </div>
              {videoError === view && (
                <div className="absolute inset-x-3 bottom-14 rounded-control bg-black/75 p-3 text-sm text-white" role="alert" data-testid="video-error">
                  {t.video.videoError} {view === 'original' && t.video.originalCodecNote}
                </div>
              )}
            </div>
            {timeline && timeline.frames.length > 0 && (
              <Scrubber timeline={timeline} events={events} playhead={playhead} onSeek={seekTo} />
            )}
            {timelineRes.initialLoading && <LoadingBlock label={t.video.timelineLoading} className="px-5" />}
            {timelineRes.error && !timeline && (
              <div className="p-4">
                <ErrorNotice error={timelineRes.error} onRetry={timelineRes.reload} compact testId="timeline-error" />
              </div>
            )}
          </Card>
        </div>
        <div className="contents lg:block lg:col-span-4 lg:space-y-6">
          <div className="order-2">
            {timeline && timeline.frames.length > 0 ? (
              <CurrentDecision frame={currentFrame} index={playhead} minScore={minScore} showLow={showLow} onShowLow={setShowLow} />
            ) : timeline ? (
              <Notice tone="neutral">{t.video.timelineEmpty}</Notice>
            ) : null}
          </div>
          <div className="order-5">{side}</div>
        </div>

      {timeline && timeline.frames.length > 0 && (
        <div className="order-3 lg:order-none lg:col-span-12">
          <EventsTable timeline={timeline} events={events} activeEvent={activeEvent} playing={playing} onSeek={seekTo} />
        </div>
      )}

      <div className="order-4 lg:order-none lg:col-span-12">
        <ResultDetails
          job={job}
          results={results}
          onSeekTrack={(firstT) => {
            if (timeline) seekTo(firstIndexAtOrAfter(timeline.frames, firstT));
          }}
          canSeek={Boolean(timeline)}
          minScore={minScore}
          showLow={showLow}
        />
      </div>
    </div>
  );
}
