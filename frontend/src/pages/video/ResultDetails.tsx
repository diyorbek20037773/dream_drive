import { Boxes, Download, ExternalLink, FileJson, FileText, FileVideo, Gauge, Image as ImageIcon, Info, Tags } from 'lucide-react';
import type { ReactNode } from 'react';

import { safeRelativeUrl } from '../../api/client';
import type { Job, JobArtifact, JobResults } from '../../api/types';
import { LimitationsList, ReasonList, SourcePill } from '../../components/domain';
import { Card, CardHeader } from '../../components/ui/Card';
import { CopyText, KeyValueList, MetricTile } from '../../components/ui/Data';
import { Notice } from '../../components/ui/Feedback';
import { Pill } from '../../components/ui/Status';
import { useI18n } from '../../i18n';
import { shortHash, useFormat } from '../../lib/format';
import { DECISION_TONE } from '../../lib/tones';
import { useApp } from '../../state/app';

const ARTIFACT_ICONS: Record<string, ReactNode> = {
  'annotated.mp4': <FileVideo className="size-4" />,
  input: <FileVideo className="size-4" />,
  'report.json': <FileJson className="size-4" />,
  'timeline.json': <FileJson className="size-4" />,
  'report.html': <FileText className="size-4" />,
  'poster.jpg': <ImageIcon className="size-4" />,
};
const ARTIFACT_ORDER = ['annotated.mp4', 'report.json', 'report.html', 'timeline.json', 'input', 'poster.jpg'];

function SummaryMetrics({ results }: { results: JobResults }) {
  const { t } = useI18n();
  const f = useFormat();
  const s = results.summary;
  // With zero analyzed frames there is nothing measured; never show a zero as if it were a measurement.
  const measured = s.analyzed_frames > 0;
  const ms = (value: number | null | undefined) => (measured && value !== null && value !== undefined ? f.num(value, 0) : null);
  return (
    <Card className="card-pad" testId="result-metrics">
      <CardHeader title={t.video.metricsTitle} icon={<Gauge className="size-5" />} />
      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <MetricTile label={t.video.analyzedFrames} value={f.int(s.analyzed_frames)} testId="metric-analyzed-frames" />
        <MetricTile label={t.video.decodedFrames} value={f.int(s.decoded_frames)} testId="metric-decoded-frames" />
        <MetricTile label={t.video.coverage} value={measured ? f.pct(s.coverage_ratio) : null} testId="metric-coverage" />
        <MetricTile label={t.video.duration} value={f.num(s.duration_s, 2)} unit={t.units.s} testId="metric-duration" />
        <MetricTile
          label={t.video.effectiveFps}
          value={measured ? f.num(s.effective_analyzed_fps, 2) : null}
          unit={t.units.fps}
          testId="metric-effective-fps"
        />
        <MetricTile label={t.video.p50} value={ms(s.timing_ms.inference_p50)} unit={t.units.ms} testId="metric-p50" />
        <MetricTile label={t.video.p95} value={ms(s.timing_ms.inference_p95)} unit={t.units.ms} testId="metric-p95" />
        <MetricTile label={t.video.decode} value={ms(s.timing_ms.decode)} unit={t.units.ms} testId="metric-decode" />
        <MetricTile label={t.video.postprocess} value={ms(s.timing_ms.postprocess)} unit={t.units.ms} testId="metric-postprocess" />
        <MetricTile label={t.video.encode} value={ms(s.timing_ms.encode)} unit={t.units.ms} testId="metric-encode" />
        <MetricTile
          label={t.video.totalWall}
          value={measured ? f.num(s.timing_ms.total_wall / 1000, 1) : null}
          unit={t.units.s}
          testId="metric-total-wall"
        />
        <MetricTile label={t.video.audioRemoved} value={s.audio === 'removed' ? t.common.yes : t.common.no} />
      </div>
    </Card>
  );
}

function TracksTable({
  results,
  onSeekTrack,
  canSeek,
  minScore,
}: {
  results: JobResults;
  onSeekTrack: (t: number) => void;
  canSeek: boolean;
  minScore: number;
}) {
  const { t } = useI18n();
  const f = useFormat();
  const tracks = results.summary.tracks;
  const dropped = results.summary.low_score_tracks_dropped ?? 0;
  return (
    <Card className="card-pad" testId="tracks">
      <CardHeader title={t.video.tracksTitle} icon={<Boxes className="size-5" />} subtitle={t.video.scoreNote} />
      {dropped > 0 && (
        <p className="mt-2 text-xs text-muted" data-testid="tracks-low-score-note">
          {t.video.lowScoreTracks(dropped, minScore.toFixed(2))}
        </p>
      )}
      {tracks.length === 0 ? (
        <p className="mt-4 text-sm text-muted">{t.video.tracksEmpty}</p>
      ) : (
        <div className="scroll-x mt-4 max-h-[360px] overflow-y-auto rounded-control border border-line">
          <table className="w-full min-w-[520px] text-left text-sm">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-line text-xs text-muted">
                <th scope="col" className="px-3 py-2 font-medium">{t.video.colTrack}</th>
                <th scope="col" className="px-3 py-2 font-medium">{t.video.colClass}</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">{t.video.colFirst}</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">{t.video.colLast}</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">{t.video.colFrames}</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">{t.video.colMaxScore}</th>
              </tr>
            </thead>
            <tbody>
              {tracks.map((track) => (
                <tr key={track.track_id} className="border-b border-line/70 last:border-0" data-testid="track-row">
                  <td className="px-3 py-2">
                    <button
                      type="button"
                      className="font-mono text-[13px] font-medium text-accent-strong hover:underline disabled:text-muted disabled:no-underline"
                      onClick={() => onSeekTrack(track.first_t)}
                      disabled={!canSeek}
                      aria-label={t.video.seekTrack(track.track_id)}
                    >
                      #{track.track_id}
                    </button>
                  </td>
                  <td className="px-3 py-2">{track.class_name}</td>
                  <td className="num px-3 py-2 text-right">{f.num(track.first_t, 2)}</td>
                  <td className="num px-3 py-2 text-right">{f.num(track.last_t, 2)}</td>
                  <td className="num px-3 py-2 text-right">{f.int(track.frames)}</td>
                  <td className="num px-3 py-2 text-right">{f.num(track.max_score, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function ClassesCard({ results, minScore, showLow }: { results: JobResults; minScore: number; showLow: boolean }) {
  const { t } = useI18n();
  const f = useFormat();
  const { capabilities } = useApp();
  const video = capabilities?.video_analysis;
  const counts = results.summary.class_counts;
  const lowCounts = results.summary.class_counts_low_score ?? {};
  const supported = new Set([
    ...(video?.supported_classes ?? []),
    ...Object.keys(counts),
    ...(showLow ? Object.keys(lowCounts) : []),
  ]);
  const unsupported = video?.unsupported_examples ?? [];
  const analyzed = results.summary.analyzed_frames > 0;
  return (
    <Card className="card-pad" testId="classes">
      <CardHeader title={t.video.classesTitle} icon={<Tags className="size-5" />} />
      <div className="mt-4 space-y-4">
        <div>
          <p className="label-caps mb-2">{t.video.classCounts}</p>
          <p className="mb-2 text-xs text-muted" data-testid="classes-threshold-note">
            {t.video.reportThreshold(minScore.toFixed(2))}
          </p>
          {supported.size === 0 ? (
            <p className="text-sm text-muted">{t.common.none}</p>
          ) : (
            <ul className="divide-y divide-line">
              {[...supported].map((name) => (
                <li key={name} className="flex items-center justify-between py-1.5 text-sm" data-class-name={name}>
                  <span>{name}</span>
                  <span className="num font-semibold">
                    {analyzed ? f.int(counts[name] ?? 0) : t.common.notMeasured}
                    {analyzed && showLow && (lowCounts[name] ?? 0) > 0 && (
                      <span className="ml-2 text-xs font-normal text-warning" data-testid="class-low-count">
                        {t.video.lowScoreCount(lowCounts[name] ?? 0)}
                      </span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="label-caps mb-2">{t.video.unsupportedClasses}</p>
          <ul className="flex flex-wrap gap-1.5" data-testid="unsupported-classes">
            {unsupported.map((name) => (
              <li key={name}>
                <Pill tone="warning">{name}</Pill>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-muted">{t.video.unsupportedNote}</p>
        </div>
      </div>
    </Card>
  );
}

function Distribution({ results }: { results: JobResults }) {
  const { t } = useI18n();
  const f = useFormat();
  const s = results.summary;
  const total = Math.max(1, s.analyzed_frames);
  const decisions = Object.entries(s.decision_counts).sort((a, b) => b[1] - a[1]);
  const reasons = Object.entries(s.reason_counts).sort((a, b) => b[1] - a[1]);
  return (
    <Card className="card-pad" testId="decision-distribution">
      <CardHeader title={t.video.decisionsTitle} />
      <ul className="mt-4 space-y-2">
        {decisions.length === 0 && <li className="text-sm text-muted">{t.common.none}</li>}
        {decisions.map(([state, count]) => (
          <li key={state} className="text-sm">
            <div className="flex items-center justify-between gap-2">
              <Pill tone={DECISION_TONE[state as keyof typeof DECISION_TONE] ?? 'neutral'} dot>
                {t.states.decision[state as keyof typeof t.states.decision] ?? state}
              </Pill>
              <span className="num text-muted">
                {f.int(count)} · {f.pct(count / total, 0)}
              </span>
            </div>
          </li>
        ))}
      </ul>
      <p className="label-caps mt-5 mb-2">{t.video.reasonsTitle}</p>
      <ul className="space-y-1.5">
        {reasons.length === 0 && <li className="text-sm text-muted">{t.common.none}</li>}
        {reasons.map(([code, count]) => (
          <li key={code} className="flex items-start justify-between gap-3">
            <ReasonList codes={[code]} dense />
            <span className="num shrink-0 text-sm text-muted">{f.int(count)}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Downloads({ artifacts }: { artifacts: JobArtifact[] }) {
  const { t } = useI18n();
  const f = useFormat();
  const sorted = [...artifacts].sort((a, b) => ARTIFACT_ORDER.indexOf(a.name) - ARTIFACT_ORDER.indexOf(b.name));
  return (
    <Card className="card-pad" testId="downloads">
      <CardHeader title={t.video.downloadsTitle} icon={<Download className="size-5" />} />
      <ul className="mt-4 divide-y divide-line">
        {sorted.map((artifact) => {
          const url = safeRelativeUrl(artifact.url);
          return (
            <li key={artifact.name} className="flex items-center justify-between gap-3 py-2.5">
              <div className="flex min-w-0 items-center gap-2.5">
                <span className="text-muted" aria-hidden>
                  {ARTIFACT_ICONS[artifact.name] ?? <FileText className="size-4" />}
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{t.video.artifactNames[artifact.name] ?? artifact.name}</p>
                  <p className="num truncate text-xs text-muted">
                    {artifact.name} · {f.bytes(artifact.bytes)}
                  </p>
                </div>
              </div>
              {url ? (
                <div className="flex shrink-0 items-center gap-1">
                  {artifact.name === 'report.html' && (
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex size-8 items-center justify-center rounded-lg text-muted hover:bg-canvas hover:text-ink"
                      aria-label={`${t.common.open}: ${artifact.name}`}
                    >
                      <ExternalLink className="size-4" aria-hidden />
                    </a>
                  )}
                  <a
                    href={url}
                    download={artifact.name === 'input' ? undefined : artifact.name}
                    className="inline-flex h-8 items-center gap-1.5 rounded-[10px] border border-line px-3 text-sm font-medium hover:bg-canvas"
                    data-testid={`download-${artifact.name}`}
                  >
                    <Download className="size-3.5" aria-hidden />
                    <span className="sr-only">{artifact.name}</span>
                    <span aria-hidden>{artifact.name.split('.').pop()?.toUpperCase()}</span>
                  </a>
                </div>
              ) : (
                <span className="text-xs text-muted">{t.common.notAvailable}</span>
              )}
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

function InfoCard({ job, results }: { job: Job; results: JobResults }) {
  const { t } = useI18n();
  const f = useFormat();
  const video = job.video;
  const model = results.model ?? job.model;
  return (
    <Card className="card-pad" testId="result-info">
      <CardHeader title={t.video.infoTitle} icon={<Info className="size-5" />} />
      <div className="mt-3 flex flex-wrap gap-2">
        <SourcePill source={results.perception_source} extra={model?.model_id} testId="result-source" />
      </div>
      <KeyValueList
        className="mt-2"
        items={[
          { label: t.video.codec, value: video?.codec ?? '—' },
          { label: t.video.resolution, value: video ? `${video.width}×${video.height}` : '—' },
          { label: t.video.fpsAvg, value: f.num(video?.fps_avg ?? null, 2) },
          { label: t.video.rotation, value: video ? `${video.rotation}°` : '—' },
          { label: t.video.duration, value: video ? `${f.num(video.duration_s, 2)} ${t.units.s}` : '—' },
          { label: t.video.inputSize, value: f.bytes(job.input.bytes) },
          { label: `${t.video.sha256} (input)`, value: <CopyText value={job.input.sha256} display={shortHash(job.input.sha256)} /> },
          { label: t.video.model, value: model?.model_id ?? '—' },
          {
            label: `${t.video.sha256} (model)`,
            value: model?.sha256 ? <CopyText value={model.sha256} display={shortHash(model.sha256)} testId="result-model-sha" /> : '—',
          },
          { label: t.video.device, value: model?.device?.toUpperCase() ?? '—' },
          { label: t.video.precision, value: model?.precision ?? '—' },
        ]}
      />
    </Card>
  );
}

export function ResultDetails({
  job,
  results,
  onSeekTrack,
  canSeek,
  minScore,
  showLow,
}: {
  job: Job;
  results: JobResults;
  onSeekTrack: (firstT: number) => void;
  canSeek: boolean;
  minScore: number;
  showLow: boolean;
}) {
  const { t } = useI18n();
  return (
    <div className="space-y-6">
      <SummaryMetrics results={results} />
      <div className="grid gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <TracksTable results={results} onSeekTrack={onSeekTrack} canSeek={canSeek} minScore={minScore} />
        </div>
        <ClassesCard results={results} minScore={minScore} showLow={showLow} />
      </div>
      <div className="grid gap-6 lg:grid-cols-2 xl:grid-cols-3">
        <Distribution results={results} />
        <Downloads artifacts={results.artifacts} />
        <InfoCard job={job} results={results} />
      </div>
      <Card className="card-pad" testId="result-limitations">
        <CardHeader title={t.video.limitationsTitle} />
        <div className="mt-3">
          {results.limitations.length > 0 ? <LimitationsList items={results.limitations} /> : <Notice tone="neutral">—</Notice>}
        </div>
      </Card>
    </div>
  );
}
