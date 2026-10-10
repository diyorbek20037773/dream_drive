import type { Page } from '@playwright/test';

import { envNumber, expect, openPage, saveScreenshot, test } from './fixtures';

const SIM_TIMEOUT = envNumber('DD_SIM_TIMEOUT_MS', 180_000);
const MANEUVERS = /^(STOP|BRAKE|SLOW_DOWN|ACCELERATE|CRUISE|TURN_LEFT|TURN_RIGHT|HOLD)$/;

interface GraphNode {
  id: string;
  x: number;
  y: number;
}
interface SimulatorsBody {
  items: { id: string; default_start: string | null; road_graph: { nodes: GraphNode[]; edges: { from: string; to: string }[] } | null }[];
}

/**
 * Chooses the reachable destination farthest from the default start (straight-line), using the real /simulators
 * graph, so the obstacle placed ~25 m ahead lies on the route. Selection goes through the UI select.
 */
async function chooseFarthestDestination(page: Page): Promise<string> {
  const body = (await (await page.request.get('/api/v1/simulators')).json()) as SimulatorsBody;
  const lite = body.items.find((item) => item.id === 'lite');
  const graph = lite?.road_graph;
  expect(graph && lite?.default_start, 'LiteRoad graph and default start').toBeTruthy();
  const start = lite?.default_start as string;
  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  const reachable = new Set([start]);
  for (let grew = true; grew; ) {
    grew = false;
    for (const edge of edges) {
      for (const [a, b] of [[edge.from, edge.to], [edge.to, edge.from]] as const) {
        if (reachable.has(a) && !reachable.has(b)) {
          reachable.add(b);
          grew = true;
        }
      }
    }
  }
  const origin = nodes.find((node) => node.id === start) as GraphNode;
  const candidates = nodes.filter((node) => node.id !== start && reachable.has(node.id));
  expect(candidates.length, 'at least one reachable destination').toBeGreaterThan(0);
  candidates.sort((a, b) => Math.hypot(b.x - origin.x, b.y - origin.y) - Math.hypot(a.x - origin.x, a.y - origin.y));
  const destination = (candidates[0] as GraphNode).id;
  await page.getByTestId('sim-start-node').selectOption(start);
  await page.getByTestId('sim-destination-node').selectOption(destination);
  return destination;
}

/** Expects the setup page to be open; keeps the chosen start/destination. */
async function startRun(page: Page, scenario: string): Promise<string> {
  await expect(page.getByTestId('sim-setup')).toBeVisible();
  await expect(page.getByTestId('road-map')).toBeVisible();
  await page.getByTestId('sim-scenario').selectOption(scenario);
  await page.getByTestId('sim-perception').selectOption('synthetic_rgb_cv');
  const created = page.waitForResponse((r) => r.url().endsWith('/api/v1/runs') && r.request().method() === 'POST');
  await page.getByTestId('sim-start').click();
  const response = await created;
  expect(response.status(), 'run accepted (202)').toBe(202);
  const run = (await response.json()) as { id: string };
  await expect(page).toHaveURL(new RegExp(`/simulation/runs/${run.id}$`));
  return run.id;
}

test.describe('simulation (LiteRoad, simulator only)', () => {
  test('unreachable map nodes are visibly disabled and the learned mode states its reason', async ({ page }) => {
    await openPage(page, '/simulation', 'page-simulation');
    await expect(page.getByTestId('road-map')).toBeVisible();
    await expect(page.locator('[data-testid^="map-node-"][data-enabled="false"]').first()).toBeVisible();
    const learned = page.getByTestId('sim-perception').locator('option[value="learned_model"]');
    if ((await learned.count()) > 0 && (await learned.isDisabled())) {
      await expect(page.getByTestId('mode-reason-learned_model')).toBeVisible();
    }
  });

  test('start → camera frames update → pause → resume → stop', async ({ page }, testInfo) => {
    test.setTimeout(SIM_TIMEOUT + 60_000);
    await openPage(page, '/simulation', 'page-simulation');
    await chooseFarthestDestination(page);
    await startRun(page, 'normal');
    const toolbar = page.getByTestId('sim-toolbar');
    await expect(toolbar).toHaveAttribute('data-run-state', 'running', { timeout: 60_000 });

    // The camera shows actual frames whose identity advances.
    const img = page.getByTestId('sim-camera-img');
    await expect(img).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('sim-source-pill')).toContainText('synthetic_rgb_cv');
    const first = Number(await img.getAttribute('data-frame-id'));
    await expect.poll(async () => Number(await img.getAttribute('data-frame-id')), { timeout: 30_000 }).toBeGreaterThan(first);
    await expect(page.getByTestId('fsm-state')).toBeVisible();
    await expect(page.getByTestId('map-ego')).toBeVisible();
    await expect(page.getByTestId('map-route')).toBeVisible();

    // Current maneuver under the camera and in the decision panel, describing the frame the camera shows.
    const strip = page.getByTestId('camera-maneuver');
    await expect(strip).toBeVisible();
    await expect(strip).toHaveAttribute('data-maneuver', MANEUVERS);
    await expect(page.getByTestId('maneuver-indicator')).toHaveAttribute('data-maneuver', MANEUVERS);
    await expect(page.getByTestId('camera-maneuver-label')).not.toBeEmpty();
    // Once the run is moving the vehicle is not holding.
    await expect.poll(async () => strip.getAttribute('data-maneuver'), { timeout: 30_000 }).not.toBe('HOLD');
    // Decision basis items belong to the command's input frame.
    const basis = page.getByTestId('decision-basis');
    await expect(basis).toBeVisible();
    await expect(page.getByTestId('basis-item').first()).toBeAttached();
    // One atomic DOM read: the panel re-renders for every new frame, so two separate reads can straddle an update.
    const { basisFrame, itemFrames } = await basis.evaluate((node) => ({
      basisFrame: node.getAttribute('data-frame-id'),
      itemFrames: Array.from(node.querySelectorAll('[data-testid="basis-item"]')).map((item) => item.getAttribute('data-frame-id')),
    }));
    expect(itemFrames.length).toBeGreaterThan(0);
    expect(new Set(itemFrames), 'all basis items share one frame id').toEqual(new Set([basisFrame]));
    await expect(page.getByTestId('maneuver-history-item').first()).toBeVisible({ timeout: 30_000 });
    await saveScreenshot(page, testInfo, 'simulation-running');

    await page.getByTestId('sim-pause').click();
    await expect(toolbar).toHaveAttribute('data-run-state', 'paused');
    await expect(strip).toHaveAttribute('data-maneuver', 'HOLD', { timeout: 15_000 });
    await page.getByTestId('sim-resume').click();
    await expect(toolbar).toHaveAttribute('data-run-state', 'running');

    await page.getByTestId('sim-stop').click();
    await expect(page.getByTestId('sim-stop').or(page.getByTestId('sim-new'))).toBeVisible();
    await expect(toolbar).toHaveAttribute('data-run-state', 'stopped', { timeout: 60_000 });
    await expect(page.getByTestId('run-report')).toBeVisible();
    await expect(page.getByTestId('report-metrics')).toBeVisible({ timeout: 30_000 });
    await saveScreenshot(page, testInfo, 'simulation-stopped');
  });

  test('obstacle_stop scenario stops before the obstacle', async ({ page }, testInfo) => {
    test.setTimeout(SIM_TIMEOUT + 60_000);
    await openPage(page, '/simulation', 'page-simulation');
    await chooseFarthestDestination(page);
    await startRun(page, 'obstacle_stop');
    const toolbar = page.getByTestId('sim-toolbar');
    await expect(toolbar).toHaveAttribute('data-run-state', /stopped|arrived|failed/, { timeout: SIM_TIMEOUT });
    await expect(toolbar).toHaveAttribute('data-run-outcome', 'stopped_before_obstacle');
    await expect(page.getByTestId('report-metrics')).toBeVisible({ timeout: 30_000 });
    // The final state is a standstill and the report lists the maneuver sequence, slowing/braking before STOP.
    await expect(page.getByTestId('camera-maneuver')).toHaveAttribute('data-maneuver', 'STOP');
    const sequence = await page
      .getByTestId('maneuver-sequence-row')
      .evaluateAll((rows) => rows.map((row) => row.getAttribute('data-maneuver')));
    expect(sequence[sequence.length - 1]).toBe('STOP');
    const firstStop = sequence.indexOf('STOP');
    expect(sequence.slice(0, firstStop).some((code) => code === 'SLOW_DOWN' || code === 'BRAKE')).toBe(true);
    await saveScreenshot(page, testInfo, 'simulation-obstacle-stop');
  });
});
