/**
 * Release E2E against the REAL running stack (gateway + api + worker + inference + redis), e.g. `make e2e`.
 * Nothing is mocked here. Required env:
 *   DD_BASE_URL        gateway URL (default http://localhost:8080)
 *   DD_FIXTURE_VIDEO   path to a small licensed MP4 used for the upload flow (tests fail, not skip, without it)
 * Optional: DD_VIDEO_TIMEOUT_MS, DD_SIM_TIMEOUT_MS, DD_EVAL_TIMEOUT_MS.
 *
 * Flows that mutate shared server state (uploads, simulations, the scenario suite) run once, on the desktop project;
 * tablet and mobile run the shell checks and full-page screenshots. Files are numbered to keep a stable order.
 */
import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.DD_BASE_URL?.trim() || 'http://localhost:8080';
const viewportOnly = /(01-shell|05-screenshots)\.spec\.ts$/;

export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results/artifacts',
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
  ],
  use: {
    baseURL,
    screenshot: 'on',
    trace: 'retain-on-failure',
    video: 'off',
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },
  projects: [
    {
      name: 'desktop',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
    {
      name: 'tablet',
      testMatch: viewportOnly,
      use: { ...devices['Desktop Chrome'], viewport: { width: 768, height: 1024 }, hasTouch: true },
    },
    {
      name: 'mobile',
      testMatch: viewportOnly,
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
});
