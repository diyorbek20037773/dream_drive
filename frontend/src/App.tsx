import { BrowserRouter, Link, Route, Routes } from 'react-router';

import { AppShell } from './components/layout/AppShell';
import { PageHeader } from './components/ui/Card';
import { I18nProvider, useI18n } from './i18n';
import { EvaluationPage } from './pages/evaluation/EvaluationPage';
import { OverviewPage } from './pages/OverviewPage';
import { SimulationPage } from './pages/simulation/SimulationPage';
import { SystemPage } from './pages/SystemPage';
import { VideoPage } from './pages/video/VideoPage';
import { AppProvider } from './state/app';

function NotFoundPage() {
  const { t } = useI18n();
  return (
    <div>
      <PageHeader title={t.notFound.title} subtitle={t.notFound.body} />
      <Link to="/" className="font-medium text-accent underline-offset-4 hover:underline">
        {t.notFound.home}
      </Link>
    </div>
  );
}

export function App() {
  return (
    <I18nProvider>
      <AppProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<OverviewPage />} />
              <Route path="video" element={<VideoPage />} />
              <Route path="video/jobs/:jobId" element={<VideoPage />} />
              <Route path="simulation" element={<SimulationPage />} />
              <Route path="simulation/runs/:runId" element={<SimulationPage />} />
              <Route path="evaluation" element={<EvaluationPage />} />
              <Route path="evaluation/:evaluationId" element={<EvaluationPage />} />
              <Route path="system" element={<SystemPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AppProvider>
    </I18nProvider>
  );
}
