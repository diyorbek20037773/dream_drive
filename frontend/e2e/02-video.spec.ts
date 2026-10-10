import fs from 'node:fs';

import type { Page } from '@playwright/test';

import { envNumber, expect, openPage, requireEnv, saveScreenshot, test } from './fixtures';

const VIDEO_TIMEOUT = envNumber('DD_VIDEO_TIMEOUT_MS', 15 * 60_000);

function fixtureVideo(): string {
  const file = requireEnv('DD_FIXTURE_VIDEO');
  if (!fs.existsSync(file)) throw new Error(`DD_FIXTURE_VIDEO does not exist: ${file}`);
  return file;
}

async function uploadFixture(page: Page): Promise<string> {
  await openPage(page, '/video', 'page-video');
  await expect(page.getByTestId('upload-limits')).toBeVisible();
  await expect(page.getByTestId('upload-choose')).toBeEnabled({ timeout: 30_000 });
  const accepted = page.waitForResponse((r) => r.url().endsWith('/api/v1/videos') && r.request().method() === 'POST');
  await page.getByTestId('upload-input').setInputFiles(fixtureVideo());
  const response = await accepted;
  expect(response.status(), 'upload accepted (202)').toBe(202);
  const job = (await response.json()) as { id: string; state: string };
  await expect(page).toHaveURL(new RegExp(`/video/jobs/${job.id}$`));
  return job.id;
}

test.describe('video analysis (real inference)', () => {
  test('upload → progress → succeeded → annotated/original → timeline seek → downloads', async ({ page }, testInfo) => {
    test.setTimeout(VIDEO_TIMEOUT + 120_000);
    const jobId = await uploadFixture(page);

    // Queue/progress panel while the worker runs, then the result player.
    const statusPanel = page.getByTestId('job-status-panel');
    const player = page.getByTestId('result-player');
    await expect(statusPanel.or(player)).toBeVisible();
    if (await statusPanel.isVisible()) {
      await expect(page.getByTestId('job-elapsed')).toBeVisible();
      await saveScreenshot(page, testInfo, 'video-progress');
    }
    await expect(player).toBeVisible({ timeout: VIDEO_TIMEOUT });

    // Source label from the job's model identity: never a generic "AI".
    const video = page.getByTestId('video-player');
    await expect(video).toHaveAttribute('data-view', 'annotated');
    await expect(page.getByTestId('video-source-label')).toContainText('learned_model');
    await expect(page.getByTestId('result-metrics')).toBeVisible();
    await expect(page.getByTestId('metric-analyzed-frames').locator('[data-measured="true"]')).toBeVisible();
    await expect(page.getByTestId('unsupported-classes')).toBeVisible();

    // Annotated MP4 is playable (metadata loads through byte-range streaming).
    await expect
      .poll(async () => video.evaluate((el: HTMLVideoElement) => el.readyState), { timeout: 30_000 })
      .toBeGreaterThanOrEqual(1);

    // Toggle to the original upload and back.
    await page.getByTestId('view-original').click();
    await expect(page.getByTestId('video-player')).toHaveAttribute('data-view', 'original');
    await expect(page.getByTestId('video-source-label')).not.toContainText('learned_model');
    await page.getByTestId('view-annotated').click();
    await expect(page.getByTestId('video-player')).toHaveAttribute('data-view', 'annotated');
    await expect
      .poll(async () => page.getByTestId('video-player').evaluate((el: HTMLVideoElement) => el.readyState), { timeout: 30_000 })
      .toBeGreaterThanOrEqual(1);

    // Clicking a timeline event seeks the video to that analyzed frame (annotated: CFR frame k at (k+0.5)/fps).
    const seeks = page.getByTestId('timeline-event-seek');
    await expect(seeks.first()).toBeVisible();
    const target = seeks.nth((await seeks.count()) > 1 ? 1 : 0);
    const frameIndex = Number(await target.getAttribute('data-frame-index'));
    await target.click();
    await expect(page.getByTestId('playhead-label')).toHaveAttribute('data-frame-index', String(frameIndex));
    const timeline = (await (await page.request.get(`/api/v1/jobs/${jobId}/artifacts/timeline.json`)).json()) as {
      analysis_fps: number;
    };
    const expected = (frameIndex + 0.5) / timeline.analysis_fps;
    await expect
      .poll(async () => page.getByTestId('video-player').evaluate((el: HTMLVideoElement) => el.currentTime))
      .toBeCloseTo(expected, 1);

    // Downloads are owner-checked same-origin links that return the bytes.
    for (const name of ['annotated.mp4', 'report.json', 'report.html', 'timeline.json']) {
      const link = page.getByTestId(`download-${name}`);
      await expect(link).toBeVisible();
      const href = await link.getAttribute('href');
      expect(href, `${name} link`).toMatch(/^\/api\/v1\//);
      const response = await page.request.get(href as string);
      expect(response.status(), `${name} download`).toBe(200);
      expect((await response.body()).length, `${name} size`).toBeGreaterThan(0);
    }
    await saveScreenshot(page, testInfo, 'video-result');
  });

  test('a queued or running job can be cancelled and deleted', async ({ page }) => {
    test.setTimeout(5 * 60_000);
    const jobId = await uploadFixture(page);
    const cancel = page.getByTestId('job-status-cancel');
    await expect(cancel).toBeVisible();
    await cancel.click();
    const item = page.locator(`[data-testid="job-item"][data-job-id="${jobId}"]`);
    await expect(item).toHaveAttribute('data-job-state', /cancelling|cancelled/);
    await expect(item).toHaveAttribute('data-job-state', 'cancelled', { timeout: 120_000 });
    await expect(page.getByTestId('job-status-panel')).toHaveAttribute('data-job-state', 'cancelled');

    await item.getByTestId('job-delete').click();
    await item.getByTestId('job-delete-confirm').click();
    await expect(page).toHaveURL(/\/video$/);
    await expect(page.locator(`[data-testid="job-item"][data-job-id="${jobId}"]`)).toHaveCount(0);
    const gone = await page.request.get(`/api/v1/jobs/${jobId}`);
    expect(gone.status()).toBe(404);
  });
});
