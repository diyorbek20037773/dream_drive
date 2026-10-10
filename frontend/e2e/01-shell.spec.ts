import { expect, openPage, test } from './fixtures';

test.describe('shell (real stack)', () => {
  test('session and capabilities render', async ({ page }) => {
    const sessionResponse = page.waitForResponse((r) => r.url().endsWith('/api/v1/session') && r.request().method() === 'POST');
    const capsResponse = page.waitForResponse((r) => r.url().endsWith('/api/v1/capabilities'));
    await openPage(page, '/', 'page-overview');
    expect((await sessionResponse).status()).toBe(200);
    expect((await capsResponse).status()).toBe(200);

    await expect(page.getByTestId('capability-summary')).toBeVisible();
    await expect(page.getByTestId('overview-video-cap')).toBeVisible();
    await expect(page.getByTestId('build-commit').filter({ visible: true })).toBeVisible();
    await expect(page.getByTestId('provenance-legend')).toBeVisible();
    await expect(page.getByTestId('safety-note')).toBeVisible();
    // The live event stream connects with the session cookie.
    await expect(page.getByTestId('sse-status').filter({ visible: true })).toHaveText(/Jonli|Live/, { timeout: 30_000 });
    // The CSRF token is held in memory only.
    const stored = await page.evaluate(() => JSON.stringify({ ...window.localStorage }));
    expect(stored).not.toMatch(/csrf/i);
  });

  test('language switch is persisted and translates the shell', async ({ page }) => {
    await openPage(page, '/', 'page-overview');
    await expect(page.locator('html')).toHaveAttribute('lang', 'uz');
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulyatsion prototip');
    await page.getByTestId('lang-en').click();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulation prototype');
    await expect(page.locator('html')).toHaveAttribute('lang', 'en');
    await page.reload();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulation prototype');
    await page.getByTestId('lang-uz').click();
    await expect(page.getByTestId('prototype-label').filter({ visible: true })).toHaveText('Simulyatsion prototip');
  });

  test('every page is reachable from the navigation', async ({ page }, testInfo) => {
    await openPage(page, '/', 'page-overview');
    const mobile = testInfo.project.name === 'mobile' || testInfo.project.name === 'tablet';
    const targets: [string, string][] = [
      ['nav-video', 'page-video'],
      ['nav-simulation', 'page-simulation'],
      ['nav-evaluation', 'page-evaluation'],
      ['nav-system', 'page-system'],
      ['nav-overview', 'page-overview'],
    ];
    for (const [nav, pageId] of targets) {
      if (mobile) await page.getByTestId('nav-open').click();
      await page.getByTestId(nav).filter({ visible: true }).click();
      await expect(page.getByTestId(pageId)).toBeVisible();
    }
  });

  test('system page distinguishes readiness from offline', async ({ page }) => {
    await openPage(page, '/system', 'page-system');
    await expect(page.getByTestId('system-health-status')).toBeVisible();
    await expect(page.getByTestId('system-model-ready')).toBeVisible();
    await expect(page.getByTestId('system-offline')).toHaveCount(0);
  });
});
