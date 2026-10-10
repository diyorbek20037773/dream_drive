/**
 * DEV-ONLY layout inspection with route-mocked LAYOUT FIXTURES (see layout-fixtures.ts). This exists to look at
 * populated pages at 1440/768/390 px before the real stack is available; it proves nothing about the backend.
 */
import http from 'node:http';
import type { AddressInfo } from 'node:net';

import type { Page, Route, TestInfo } from '@playwright/test';
import { expect, test } from '@playwright/test';

import * as fx from './layout-fixtures';
import { collectPageErrors, devScreenshot, horizontalOverflow } from './helpers';

let sse: http.Server;
let ssePort = 0;

test.beforeAll(async () => {
  // A held-open event stream so the UI shows its normal "live" state (route.fulfill cannot keep a body open).
  sse = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' });
    res.write(': layout fixture stream\n\n');
    // Fixture decision events (maneuver changes), sent once the page has attached to the run.
    setTimeout(() => {
      fx.decisionEvents.forEach((event, index) => {
        const envelope = {
          schema_version: '1.0',
          event_id: `fx${index}`,
          timestamp_utc: new Date().toISOString(),
          resource_id: fx.runId,
          type: 'decision',
          payload: {
            kind: 'decision',
            run_id: fx.runId,
            frame_id: event.frame_id,
            state: event.state,
            reason_codes: event.reason_codes,
            command: null,
            sim_time_s: event.sim_time_s,
            maneuver: { primary: event.primary, longitudinal: event.longitudinal, lateral: 'straight', since_sim_time_s: event.sim_time_s },
          },
        };
        res.write(`event: decision\nid: 1-${index}\ndata: ${JSON.stringify(envelope)}\n\n`);
      });
    }, 2_000);
  });
  await new Promise<void>((resolve) => sse.listen(0, '127.0.0.1', resolve));
  ssePort = (sse.address() as AddressInfo).port;
});

test.afterAll(async () => {
  sse.closeAllConnections();
  await new Promise<void>((resolve) => sse.close(() => resolve()));
});

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
const notFound = (route: Route) =>
  json(route, { code: 'not_found', message: 'layout fixture', request_id: 'fixture', details: {} }, 404);

async function mockApi(page: Page, runState: 'running' | 'stopped' = 'running') {
  const run =
    runState === 'running'
      ? fx.run
      : { ...fx.run, state: 'stopped', outcome: 'stopped_before_obstacle', finished_utc: fx.run.created_utc };
  await page.route('**/health/ready', (route) => json(route, fx.health));
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    const method = request.method();
    if (pathname === '/api/v1/events') return route.continue({ url: `http://127.0.0.1:${ssePort}/events` });
    if (pathname === '/api/v1/session' && method === 'POST') return json(route, fx.session);
    if (pathname === '/api/v1/capabilities') return json(route, fx.capabilities);
    if (pathname === '/api/v1/jobs') return json(route, { items: fx.jobs, next_cursor: null });
    const jobMatch = /^\/api\/v1\/jobs\/([^/]+)(\/.*)?$/.exec(pathname);
    if (jobMatch) {
      const job = fx.jobs.find((item) => item.id === jobMatch[1]);
      if (!job) return notFound(route);
      const rest = jobMatch[2] ?? '';
      if (rest === '') return json(route, job);
      if (rest === '/results') return json(route, fx.results);
      if (rest === '/artifacts/timeline.json') return json(route, fx.timeline);
      return notFound(route);
    }
    if (pathname === '/api/v1/simulators') {
      return json(route, { items: fx.capabilities.simulation.simulators });
    }
    if (pathname === '/api/v1/runs') return json(route, { items: [run], next_cursor: null });
    if (pathname === `/api/v1/runs/${fx.runId}`) return json(route, run);
    if (pathname === `/api/v1/runs/${fx.runId}/frame`) {
      return route.fulfill({
        status: 200,
        contentType: 'image/svg+xml',
        headers: { 'X-Frame-Id': String(fx.liveFrameId), 'X-Sim-Time': '6.0', 'X-Perception-Source': 'synthetic_rgb_cv' },
        body: fx.frameSvg,
      });
    }
    if (pathname === `/api/v1/runs/${fx.runId}/report`) return json(route, fx.report);
    if (pathname === '/api/v1/evaluations') return json(route, { items: fx.evaluations });
    const evalMatch = /^\/api\/v1\/evaluations\/([^/]+)$/.exec(pathname);
    if (evalMatch) {
      const evaluation = fx.evaluations.find((item) => item.id === evalMatch[1]);
      return evaluation ? json(route, evaluation) : notFound(route);
    }
    return notFound(route);
  });
}

async function check(page: Page, testInfo: TestInfo, name: string) {
  expect(await horizontalOverflow(page), `${name}: horizontal overflow`).toBeLessThanOrEqual(1);
  await devScreenshot(page, testInfo, `fixture-${name}`);
}

test.describe('layout fixtures (dev only)', () => {
  test('overview and system', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await mockApi(page);
    await page.goto('/');
    await expect(page.getByTestId('capability-summary')).toBeVisible();
    await expect(page.getByTestId('sse-status').filter({ visible: true })).toHaveText('Jonli');
    await expect(page.getByTestId('recent-jobs').getByRole('link').nth(1)).toBeVisible();
    await check(page, testInfo, 'overview');
    await page.goto('/system');
    await expect(page.getByTestId('system-model-sha')).toBeVisible();
    await expect(page.getByTestId('system-models').getByText('rfdetr_nano')).toBeVisible();
    await expect(page.getByTestId('system-health-status')).toBeVisible();
    await check(page, testInfo, 'system');
    expect(errors).toEqual([]);
  });

  test('video: list, running job, failed job, result', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await mockApi(page);
    await page.goto('/video');
    await expect(page.getByTestId('job-list')).toBeVisible();
    await expect(page.getByTestId('upload-choose')).toBeEnabled();
    await check(page, testInfo, 'video-list');

    await page.goto(`/video/jobs/${fx.jobs[1]?.id}`);
    await expect(page.getByTestId('job-status-panel')).toHaveAttribute('data-job-state', 'running');
    await check(page, testInfo, 'video-running');

    await page.goto(`/video/jobs/${fx.jobs[2]?.id}`);
    await expect(page.getByTestId('job-failure-text')).toBeVisible();
    await check(page, testInfo, 'video-failed');

    await page.goto(`/video/jobs/${fx.jobs[0]?.id}`);
    await expect(page.getByTestId('result-player')).toBeVisible();
    await expect(page.getByTestId('video-source-label')).toHaveText('YOLOPv2 · learned_model · CPU');
    // Seeking via the events table moves the playhead to that analyzed frame.
    const seek = page.getByTestId('timeline-event-seek').nth(2);
    const index = await seek.getAttribute('data-frame-index');
    await seek.click();
    await expect(page.getByTestId('playhead-label')).toHaveAttribute('data-frame-index', index ?? '');
    await expect(page.getByTestId('current-decision-state')).toBeVisible();
    // Analysis-only recommendation derived from the decision state; no vehicle is controlled.
    await expect(page.getByTestId('current-recommendation')).toBeVisible();
    await expect(page.getByTestId('no-vehicle-controlled')).toBeVisible();
    const recs = await page
      .getByTestId('event-recommendation')
      .evaluateAll((nodes) => nodes.map((node) => node.getAttribute('data-recommendation')));
    expect(new Set(recs)).toEqual(new Set(['CONTINUE', 'SLOW_DOWN', 'STOP']));
    await check(page, testInfo, 'video-result');
    expect(errors).toEqual([]);
  });

  test('simulation: running run with camera overlay and map', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await mockApi(page, 'running');
    await page.goto('/simulation');
    // Re-attaches to the active run from the run list.
    await expect(page).toHaveURL(new RegExp(`/simulation/runs/${fx.runId}$`));
    await expect(page.getByTestId('sim-camera-img')).toHaveAttribute('data-frame-id', String(fx.liveFrameId));
    await expect(page.getByTestId('sim-camera-overlay')).toBeVisible();
    await expect(page.getByTestId('map-ego')).toBeVisible();
    await expect(page.getByTestId('sim-stop')).toBeEnabled();
    // Current maneuver: under the camera (above the fold on phones) and at the top of the decision panel.
    const strip = page.getByTestId('camera-maneuver');
    await expect(strip).toHaveAttribute('data-maneuver', 'SLOW_DOWN');
    await expect(strip).toHaveAttribute('data-frame-id', String(fx.liveFrameId));
    await expect(page.getByTestId('camera-maneuver-label')).toHaveText('Sekinlashish');
    const box = await strip.boundingBox();
    expect(box ? box.y + box.height : Infinity, 'maneuver strip above the fold').toBeLessThanOrEqual(page.viewportSize()?.height ?? 0);
    await expect(page.getByTestId('maneuver-indicator')).toHaveAttribute('data-maneuver', 'SLOW_DOWN');
    // Decision basis of the same frame, triggered input first with its effect.
    await expect(page.getByTestId('basis-frame')).toContainText(String(fx.liveFrameId));
    const cue = page.getByTestId('basis-triggered').locator('[data-basis-key="image_obstacle_cue"]');
    await expect(cue).toContainText(/0[.,]52/);
    await expect(cue).toContainText(/≥ 0[.,]35/);
    await expect(cue.getByTestId('basis-effect')).toHaveText('Sekinlashish');
    await expect(page.getByTestId('basis-item').first()).toHaveAttribute('data-frame-id', String(fx.liveFrameId));
    // History fed by the fixture event stream.
    await expect(page.getByTestId('maneuver-history-item')).toHaveCount(3);
    await expect(page.getByTestId('maneuver-history-item').first()).toHaveAttribute('data-maneuver', 'SLOW_DOWN');
    await check(page, testInfo, 'simulation-running');
    await page.getByTestId('lang-en').click();
    await expect(page.getByTestId('camera-maneuver-label')).toHaveText('Slowing down');
    await expect(page.getByTestId('basis-triggered').locator('[data-basis-key="image_obstacle_cue"]')).toContainText('Obstacle cue');
    await check(page, testInfo, 'simulation-running-en');
    expect(errors).toEqual([]);
  });

  test('simulation: setup and finished run report', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await mockApi(page, 'stopped');
    await page.goto('/simulation');
    await expect(page.getByTestId('sim-setup')).toBeVisible();
    await expect(page.getByTestId('map-node-n10')).toHaveAttribute('data-enabled', 'false');
    await page.getByTestId('map-node-n4').click();
    await expect(page.getByTestId('sim-destination-node')).toHaveValue('n4');
    await expect(page.getByTestId('sim-start')).toBeEnabled();
    await check(page, testInfo, 'simulation-setup');

    await page.goto(`/simulation/runs/${fx.runId}`);
    await expect(page.getByTestId('run-report')).toBeVisible();
    await expect(page.getByTestId('report-metrics')).toBeVisible();
    const rows = page.getByTestId('maneuver-sequence-row');
    await expect(rows).toHaveCount(5);
    await expect(rows.last()).toHaveAttribute('data-maneuver', 'STOP');
    await check(page, testInfo, 'simulation-report');
    expect(errors).toEqual([]);
  });

  test('evaluation: list, details and comparison', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await mockApi(page);
    await page.goto(`/evaluation/${fx.evaluations[0]?.id}`);
    await expect(page.getByTestId('evaluation-detail')).toBeVisible();
    await expect(page.getByTestId('ml-accuracy')).toContainText('not evaluated: no labeled subset');
    const boxes = page.getByTestId('evaluation-row').locator('input[type="checkbox"]');
    await boxes.nth(0).check();
    await boxes.nth(1).check();
    await expect(page.getByTestId('evaluation-compare')).toBeVisible();
    await check(page, testInfo, 'evaluation');
    expect(errors).toEqual([]);
  });

  test('english labels on the result page', async ({ page }, testInfo) => {
    await mockApi(page);
    await page.goto(`/video/jobs/${fx.jobs[0]?.id}`);
    await page.getByTestId('lang-en').click();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulation prototype');
    await expect(page.getByTestId('result-metrics')).toContainText('Analyzed frames');
    await check(page, testInfo, 'video-result-en');
  });
});
