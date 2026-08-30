// Server-side only (used from Route Handlers, never imported by client
// components) -- API_BASE_URL is http://api:8000 inside docker-compose's
// network, http://localhost:8000 for local `npm run dev`. Keeping this
// fetch server-side means it never reaches the browser bundle, so the
// FastAPI service needs no CORS configuration.
export function apiBaseUrl(): string {
  return process.env.API_BASE_URL ?? "http://localhost:8000";
}
