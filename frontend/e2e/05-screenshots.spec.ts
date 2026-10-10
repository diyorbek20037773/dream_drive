import fs from 'node:fs';

import { envNumber, expect, openPage, requireEnv, saveScreenshot, test } from './fixtures';

/** Full-page screenshots of every page at the project's viewport (desktop 1440x900, tablet 768x1024, mobile 390x844). */
test.describe('responsive screenshots (real stack)', () => {
  const pages: [string, string, string][] = [
    ['overview', '/', 'page-overview'],
    ['video', '/video', 'page-video'],
    ['simulation', '/simulation', 'page-simulation'],
    ['evaluation', '/evaluation', 'page-evaluation'],
    ['system', '/system', 'page-system'],
  ];

  for (const [name, url, testId] of pages) {
    test(`${name} page`, async ({ page }, testInfo) => {
      await openPage(page, url, testId);
      // No horizontal page overflow: wide tables scroll inside their own containers.
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflow, 'horizontal page overflow (px)').toBeLessThanOrEqual(1);
      await saveScreenshot(page, testInfo, name);
    });
  }

  // Jobs and runs are session-owned (another test's session cannot see them), so this test creates its own.
  test('own analysed video and live run', async ({ page }, testInfo) => {
    const videoTimeout = envNumber('DD_VIDEO_TIMEOUT_MS', 600_000);
    test.setTimeout(videoTimeout + 180_000);
    await openPage(page, '/', 'page-overview');
    const csrf = (await page.context().cookies()).find((cookie) => cookie.name === 'dd_csrf')?.value;
    expect(csrf, 'dd_csrf cookie after session creation').toBeTruthy();
    const headers = { 'X-CSRF-Token': csrf as string };

    const fixture = requireEnv('DD_FIXTURE_VIDEO');
    const upload = await page.request.post('/api/v1/videos', {
      headers,
      multipart: { file: { name: 'clip.mp4', mimeType: 'video/mp4', buffer: fs.readFileSync(fixture) } },
    });
    expect(upload.status(), 'upload accepted (202)').toBe(202);
    const job = (await upload.json()) as { id: string };
    await expect
      .poll(async () => ((await (await page.request.get(`/api/v1/jobs/${job.id}`)).json()) as { state: string }).state, {
        timeout: videoTimeout,
        intervals: [2_000],
      })
      .toBe('succeeded');
    await openPage(page, `/video/jobs/${job.id}`, 'page-video');
    await expect(page.getByTestId('result-player')).toBeVisible();
    await saveScreenshot(page, testInfo, 'video-result');

    const sims = (await (await page.request.get('/api/v1/simulators')).json()) as {
      items: { id: string; default_start: string | null; road_graph: { nodes: { id: string; reachable_from_default: boolean }[] } | null }[];
    };
    const lite = sims.items.find((item) => item.id === 'lite');
    const start = lite?.default_start as string;
    const destination = lite?.road_graph?.nodes.find((node) => node.id !== start && node.reachable_from_default)?.id;
    expect(start && destination, 'reachable LiteRoad start and destination').toBeTruthy();
    const created = await page.request.post('/api/v1/runs', {
      headers,
      data: { simulator: 'lite', perception_source: 'synthetic_rgb_cv', start_node: start, destination_node: destination, scenario: 'normal', seed: 1 },
    });
    expect(created.status(), 'run accepted (202)').toBe(202);
    const run = (await created.json()) as { id: string };
    await openPage(page, `/simulation/runs/${run.id}`, 'page-simulation');
    await expect(page.getByTestId('decision-panel')).toBeVisible();
    await expect(page.getByTestId('sim-camera-img')).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(4_000); // let a few live frames and decisions arrive before the capture
    await saveScreenshot(page, testInfo, 'simulation-run');
    const stop = await page.request.post(`/api/v1/runs/${run.id}/control`, { headers, data: { action: 'stop' } });
    expect(stop.ok(), 'operator stop accepted').toBeTruthy();
  });
});
