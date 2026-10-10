/**
 * DEV-ONLY smoke (not a release gate): builds the bundle, serves it with `vite preview` WITHOUT a backend and checks
 * that the shell renders and that offline/error states are shown. `layout-fixtures.spec.ts` additionally renders
 * populated pages from clearly-labelled, hand-written layout fixtures (route-mocked) purely to inspect responsive
 * layout. Neither replaces the release E2E in ./e2e against the real stack.
 */
import { defineConfig, devices } from '@playwright/test';

const port = 4173;
// Set DD_DEV_BASE_URL to run the same smoke against an already running server (e.g. the built nginx web container,
// which also exercises the CSP and security headers). Otherwise `vite preview` is started.
const externalBase = process.env.DD_DEV_BASE_URL?.trim() || undefined;

export default defineConfig({
  testDir: './e2e-dev',
  outputDir: './test-results-dev/artifacts',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  reporter: [['list'], ['html', { outputFolder: 'playwright-report-dev', open: 'never' }]],
  webServer: externalBase
    ? undefined
    : {
        command: `npm run build && npx vite preview --host 127.0.0.1 --port ${port} --strictPort`,
        url: `http://127.0.0.1:${port}`,
        reuseExistingServer: false,
        timeout: 180_000,
        // No DD_DEV_API_TARGET: the preview server answers /api and /health with 502 like a gateway whose API is down.
        env: { DD_DEV_API_TARGET: '' },
      },
  use: {
    baseURL: externalBase ?? `http://127.0.0.1:${port}`,
    screenshot: 'on',
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'tablet', use: { ...devices['Desktop Chrome'], viewport: { width: 768, height: 1024 }, hasTouch: true } },
    {
      name: 'mobile',
      use: { ...devices['Desktop Chrome'], viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
    },
  ],
});
