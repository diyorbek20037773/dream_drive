/** DEV-ONLY: the built bundle with no backend behind it. Verifies the shell and the explicit offline/error states. */
import { expect, test } from '@playwright/test';

import { collectPageErrors, devScreenshot, horizontalOverflow } from './helpers';

test.describe('backend absent (dev smoke)', () => {
  test('shell renders with the offline state and retry', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await page.goto('/');
    await expect(page.getByTestId('page-overview')).toBeVisible();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulyatsion prototip');
    await expect(page.getByTestId('session-error')).toBeVisible();
    // vite preview answers 502 (gateway with API down); the nginx web container answers 503 service_unavailable.
    await expect(page.getByTestId('session-error')).toHaveAttribute('data-error-code', /service_offline|service_unavailable/);
    await expect(page.getByTestId('capabilities-error')).toBeVisible();
    await expect(page.getByTestId('sse-status').filter({ visible: true })).toBeVisible();
    // Retry re-attempts and keeps the explicit error (no fake data appears).
    const retried = page.waitForRequest((r) => r.url().endsWith('/api/v1/session') && r.method() === 'POST');
    await page.getByTestId('session-error-retry').click();
    await retried;
    await expect(page.getByTestId('session-error')).toBeVisible();
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    await devScreenshot(page, testInfo, 'offline-overview');
    expect(errors).toEqual([]);
  });

  test('video page disables upload with a reason', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await page.goto('/video');
    await expect(page.getByTestId('page-video')).toBeVisible();
    await expect(page.getByTestId('upload-choose')).toBeDisabled();
    await expect(page.getByTestId('upload-disabled-reason')).toBeVisible();
    await expect(page.getByTestId('session-pending-error')).toBeVisible();
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    await devScreenshot(page, testInfo, 'offline-video');
    expect(errors).toEqual([]);
  });

  test('simulation page keeps Start disabled with a reason', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await page.goto('/simulation');
    await expect(page.getByTestId('page-simulation')).toBeVisible();
    await expect(page.getByTestId('sim-start')).toBeDisabled();
    await expect(page.getByTestId('sim-start-reason')).toBeVisible();
    await expect(page.getByTestId('session-pending-error').first()).toBeVisible();
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    await devScreenshot(page, testInfo, 'offline-simulation');
    expect(errors).toEqual([]);
  });

  test('evaluation and system pages show service offline', async ({ page }, testInfo) => {
    const errors = collectPageErrors(page);
    await page.goto('/evaluation');
    await expect(page.getByTestId('page-evaluation')).toBeVisible();
    await expect(page.getByTestId('suite-run')).toBeDisabled();
    await expect(page.getByTestId('session-pending-error')).toBeVisible();
    await devScreenshot(page, testInfo, 'offline-evaluation');

    await page.goto('/system');
    await expect(page.getByTestId('page-system')).toBeVisible();
    await expect(page.getByTestId('system-offline')).toBeVisible();
    await expect(page.getByTestId('system-health-error')).toBeVisible();
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    await devScreenshot(page, testInfo, 'offline-system');
    expect(errors).toEqual([]);
  });

  test('language switch and navigation work offline', async ({ page }, testInfo) => {
    await page.goto('/');
    await page.getByTestId('lang-en').click();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulation prototype');
    await page.reload();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulation prototype');
    const drawer = testInfo.project.name !== 'desktop';
    if (drawer) {
      await page.getByTestId('nav-open').click();
      await expect(page.getByRole('dialog')).toBeVisible();
      await devScreenshot(page, testInfo, 'offline-mobile-nav');
    }
    await page.getByTestId('nav-system').filter({ visible: true }).click();
    await expect(page.getByTestId('page-system')).toBeVisible();
    if (drawer) await expect(page.getByRole('dialog')).toHaveCount(0);
    await page.getByTestId('lang-uz').click();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulyatsion prototip');
  });

  test('unknown routes show the not-found page', async ({ page }) => {
    await page.goto('/no-such-page');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Sahifa topilmadi');
  });
});
