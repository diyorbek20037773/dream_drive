import { envNumber, expect, openPage, saveScreenshot, test } from './fixtures';

const EVAL_TIMEOUT = envNumber('DD_EVAL_TIMEOUT_MS', 20 * 60_000);

test.describe('evaluation', () => {
  test('ML accuracy is shown exactly as reported and the scenario suite runs', async ({ page }, testInfo) => {
    test.setTimeout(EVAL_TIMEOUT + 60_000);
    await openPage(page, '/evaluation', 'page-evaluation');
    await expect(page.getByTestId('ml-accuracy')).toBeVisible();
    await expect(page.getByTestId('evaluation-list')).toBeVisible();

    await page.getByTestId('suite-seeds').fill('1');
    const created = page.waitForResponse((r) => r.url().endsWith('/api/v1/evaluations') && r.request().method() === 'POST');
    await page.getByTestId('suite-run').click();
    const response = await created;
    expect(response.status(), 'suite accepted (202)').toBe(202);
    const evaluation = (await response.json()) as { id: string };
    await expect(page).toHaveURL(new RegExp(`/evaluation/${evaluation.id}$`));
    await expect(page.getByTestId('evaluation-detail')).toBeVisible();

    const row = page.locator('[data-testid="evaluation-row"]').first();
    await expect(row).toHaveAttribute('data-evaluation-state', /succeeded|failed|interrupted/, { timeout: EVAL_TIMEOUT });
    await expect(row).toHaveAttribute('data-evaluation-state', 'succeeded');
    await expect(page.getByTestId('evaluation-metrics')).toBeVisible();
    const samples = await page.getByTestId('evaluation-samples').textContent();
    expect(Number((samples ?? '').replace(/\D/g, ''))).toBeGreaterThan(0);
    await saveScreenshot(page, testInfo, 'evaluation-suite');
  });
});
