import { expect, openPage, saveScreenshot, test } from './fixtures';

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

  test('latest succeeded job and latest run', async ({ page }, testInfo) => {
    await openPage(page, '/', 'page-overview');
    const jobs = (await (await page.request.get('/api/v1/jobs?limit=20')).json()) as { items: { id: string; state: string }[] };
    const job = jobs.items.find((item) => item.state === 'succeeded');
    expect(job, 'a succeeded job exists (run 02-video first)').toBeTruthy();
    await openPage(page, `/video/jobs/${job?.id}`, 'page-video');
    await expect(page.getByTestId('result-player')).toBeVisible();
    await saveScreenshot(page, testInfo, 'video-result');

    const runs = (await (await page.request.get('/api/v1/runs?limit=20')).json()) as { items: { id: string }[] };
    const run = runs.items[0];
    expect(run, 'a run exists (run 03-simulation first)').toBeTruthy();
    await openPage(page, `/simulation/runs/${run?.id}`, 'page-simulation');
    await expect(page.getByTestId('decision-panel')).toBeVisible();
    await saveScreenshot(page, testInfo, 'simulation-run');
  });
});
