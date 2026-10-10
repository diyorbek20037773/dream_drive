# DreamDrive web (frontend)

React 18 + TypeScript (strict) + Vite + Tailwind CSS v4 + react-router 7. Uzbek Latin is the default UI language
(`src/i18n/uz.ts`), English in `src/i18n/en.ts`. The bundle only calls relative `/api/v1` and `/health` URLs; the
gateway routes them. No API host is compiled in.

## Scripts (Node 22)

| Command | Purpose |
|---|---|
| `npm ci` | install from the lockfile |
| `npm run dev` | dev server; set `DD_DEV_API_TARGET=http://127.0.0.1:8080` to proxy `/api` and `/health` to a running gateway |
| `npm run build` | type-check + production build into `dist/` |
| `npm run typecheck` / `npm run lint` | `tsc --noEmit` (app + tooling) / ESLint flat config, zero warnings |
| `npm run gen:types` | regenerate `src/api/schema.d.ts` from `src/api/openapi.json` (exported by the backend) |
| `npm run test:e2e` | release E2E against the **real** stack (see below) |
| `npm run test:e2e:dev` | dev-only smoke: built bundle, no backend, plus route-mocked layout fixtures (not a release gate) |

## Release E2E (`e2e/`, `playwright.config.ts`)

Runs against the running gateway with nothing mocked. Env: `DD_BASE_URL` (default `http://localhost:8080`),
`DD_FIXTURE_VIDEO` (required; the upload tests fail without it), optional `DD_VIDEO_TIMEOUT_MS`, `DD_SIM_TIMEOUT_MS`,
`DD_EVAL_TIMEOUT_MS`. Projects: desktop 1440x900 (all flows), tablet 768x1024 and mobile 390x844 (shell + screenshots).
Every test fails on browser console errors, page errors or requests to a localhost host other than the base URL.
Full-page screenshots: `test-results/screenshots/<project>/`.

## Dev-only smoke (`e2e-dev/`, `playwright.dev.config.ts`)

Starts `vite preview` with the API answering 502 (as the gateway does when the API is down) and checks the offline
states. `DD_DEV_BASE_URL=http://127.0.0.1:18080` runs the same checks against an already running web container
(exercises nginx CSP and headers). `layout-fixtures.spec.ts` renders populated pages from hand-written, clearly
labelled fixtures purely for responsive-layout inspection; it is not evidence of backend behaviour.

## Image

`docker build -f frontend/Dockerfile -t dreamdrive-web:dev .` (context = repo root; `frontend/Dockerfile.dockerignore`
limits the context). nginx-unprivileged on :8080 as uid 101, SPA fallback, immutable caching for hashed assets,
`no-cache` for `index.html`, CSP `script-src 'self'` with `frame-ancestors 'none'`, works with a read-only root
filesystem and a tmpfs `/tmp`. `/api/*` and `/health/*` answer 503 JSON here because only the gateway routes the API.
Build args: `GIT_COMMIT`, `BUILD_VERSION`, `BUILD_DIRTY`, `BUILD_DATE` (OCI labels).
