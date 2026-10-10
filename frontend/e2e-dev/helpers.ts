import fs from 'node:fs';
import path from 'node:path';

import type { Page, TestInfo } from '@playwright/test';

export async function devScreenshot(page: Page, testInfo: TestInfo, name: string): Promise<void> {
  const dir = path.join('test-results-dev', 'screenshots', testInfo.project.name);
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${name}.png`);
  // Sticky header/sidebar render at the current scroll offset in full-page captures; capture from the top.
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: file, fullPage: true });
  await testInfo.attach(name, { path: file, contentType: 'image/png' });
}

/**
 * Collects uncaught page errors and Content-Security-Policy violations. Console resource errors (failed API calls)
 * are expected while the backend is absent and are not collected.
 */
export function collectPageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && /Content Security Policy/i.test(message.text())) errors.push(message.text());
  });
  return errors;
}

export async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
}
