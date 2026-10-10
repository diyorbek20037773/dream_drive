/**
 * Shared release-E2E fixtures: every test fails on browser console errors, uncaught page errors, or requests to a
 * localhost-style host other than the configured base URL (a compiled-in developer address would show up here).
 */
import fs from 'node:fs';
import path from 'node:path';

import type { Page, TestInfo } from '@playwright/test';
import { expect, test as base } from '@playwright/test';

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '0.0.0.0', '[::1]', '::1']);

export interface Guard {
  /** Console error texts that a test intentionally provokes (substring match). */
  allowConsole: (pattern: string) => void;
}

export const test = base.extend<{ guard: Guard }>({
  guard: [
    async ({ page, baseURL }, use) => {
      const base = new URL(baseURL ?? 'http://localhost:8080');
      const consoleErrors: string[] = [];
      const foreignLocal: string[] = [];
      const allowed: string[] = [];
      page.on('console', (message) => {
        if (message.type() === 'error') consoleErrors.push(message.text());
      });
      page.on('pageerror', (error) => consoleErrors.push(`pageerror: ${error.message}`));
      page.on('request', (request) => {
        const url = new URL(request.url());
        if (!url.protocol.startsWith('http')) return;
        if (LOCAL_HOSTS.has(url.hostname) && url.origin !== base.origin) foreignLocal.push(request.url());
      });
      await use({ allowConsole: (pattern) => allowed.push(pattern) });
      const unexpected = consoleErrors.filter((text) => !allowed.some((pattern) => text.includes(pattern)));
      expect(unexpected, 'browser console errors').toEqual([]);
      expect(foreignLocal, 'requests to localhost hosts other than the base URL').toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };

/** Full-page screenshot saved under test-results/screenshots/<project>/<name>.png (and attached to the report). */
export async function saveScreenshot(page: Page, testInfo: TestInfo, name: string): Promise<void> {
  const dir = path.join('test-results', 'screenshots', testInfo.project.name);
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${name}.png`);
  // Sticky header/sidebar render at the current scroll offset in full-page captures; capture from the top.
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: file, fullPage: true });
  await testInfo.attach(name, { path: file, contentType: 'image/png' });
}

/** Navigates and waits until the session exists and the shell rendered (SSE keeps the network busy, so no networkidle). */
export async function openPage(page: Page, url: string, pageTestId: string): Promise<void> {
  await page.goto(url);
  await expect(page.getByTestId('prototype-label').filter({ visible: true })).toBeVisible();
  await expect(page.getByTestId(pageTestId)).toBeVisible();
  await expect(page.getByTestId('offline-banner')).toHaveCount(0);
}

export function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} must be set for the release E2E suite (it is not optional and is never skipped)`);
  return value;
}

export function envNumber(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}
