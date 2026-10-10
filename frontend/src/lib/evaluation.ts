import type { Evaluation, EvaluationDataset } from '../api/types';
import type { Dict } from '../i18n/uz';
import { humanizeKey } from './format';

const CLEAN_TOKENS: Record<string, string> = { iou: 'IoU', f1: 'F1', ap50: 'AP50', map50: 'mAP50', ms: '(ms)', ttc: 'TTC', fps: 'FPS' };

/** Readable fallback for metric keys without a translation: tokens like iou/f1 keep their usual spelling. */
export function cleanKey(key: string): string {
  const words = key.split('_').filter(Boolean).map((word) => CLEAN_TOKENS[word] ?? word);
  return humanizeKey(words.join('_'));
}

/** Human-readable metric name: exact translations first, then "<prefix>_<class>" patterns, then a cleaned key. */
export function metricLabel(key: string, t: Dict): string {
  const exact = t.evaluation.metricLabels[key];
  if (exact) return exact;
  // Longest prefix first so ap50_95_person is not read as ap50 + "95_person".
  const prefixes = Object.keys(t.evaluation.metricPrefixes).sort((a, b) => b.length - a.length);
  for (const prefix of prefixes) {
    if (key.startsWith(`${prefix}_`)) {
      const cls = key.slice(prefix.length + 1);
      const name = t.evaluation.classNames[cls];
      if (name) return `${t.evaluation.metricPrefixes[prefix]} · ${name}`;
    }
  }
  return cleanKey(key);
}

/** e.g. "COCO val2017 ko‘cha to‘plami · 300 rasm · obyekt aniqlash"; null when the record names no dataset. */
export function datasetTitle(dataset: EvaluationDataset | null | undefined, sampleCount: number | null, t: Dict, formatInt: (n: number) => string): string | null {
  if (!dataset) return null;
  const parts = [t.evaluation.datasetNames[dataset.id] ?? dataset.name];
  if (sampleCount !== null && sampleCount > 0 && dataset.sample_unit !== 'scenario_run') {
    parts.push(`${formatInt(sampleCount)} ${t.evaluation.sampleUnits[dataset.sample_unit] ?? dataset.sample_unit}`);
  }
  const task = t.evaluation.datasetTasks[dataset.id];
  if (task) parts.push(task);
  return parts.join(' · ');
}

export function evaluationTitle(evaluation: Evaluation, t: Dict, formatInt: (n: number) => string): string | null {
  return datasetTitle(evaluation.dataset, evaluation.sample_count, t, formatInt);
}
