import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import type { Connect, Plugin } from 'vite';
import { defineConfig } from 'vite';

/**
 * Developer-only API wiring. The production bundle always calls relative `/api/v1` URLs and never contains a host:
 * the gateway (Caddy) routes `/api/*` to the backend. For local development set DD_DEV_API_TARGET (for example the
 * compose gateway) and the dev/preview server proxies `/api` and `/health` to it.
 */
const apiTarget = process.env.DD_DEV_API_TARGET?.trim() || undefined;

/**
 * Without DD_DEV_API_TARGET the dev/preview server answers API paths with an empty 502, which is exactly what the
 * gateway returns while the API container is down. This keeps the "service offline" UI path testable locally
 * instead of letting the SPA fallback answer API calls with index.html.
 */
function apiOfflinePlugin(): Plugin {
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    const url = req.url ?? '';
    if (url.startsWith('/api/') || url.startsWith('/health/')) {
      res.statusCode = 502;
      res.setHeader('Cache-Control', 'no-store');
      res.end();
      return;
    }
    next();
  };
  return {
    name: 'dreamdrive-dev-api-offline',
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

const proxy = apiTarget
  ? {
      '/api': { target: apiTarget, changeOrigin: false },
      '/health': { target: apiTarget, changeOrigin: false },
    }
  : undefined;

export default defineConfig({
  plugins: [react(), tailwindcss(), ...(apiTarget ? [] : [apiOfflinePlugin()])],
  build: {
    target: 'es2022',
    // Never inline assets as data: URIs; the CSP only allows same-origin fonts/images.
    assetsInlineLimit: 0,
    sourcemap: false,
    chunkSizeWarningLimit: 700,
  },
  server: { proxy },
  preview: { proxy },
});
