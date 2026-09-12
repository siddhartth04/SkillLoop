# Deployment

## Quick start (Docker)
```bash
docker build -t skillloop .
docker run -p 7331:7331 -v skillloop-data:/data -e SKILLLOOP_LOG=json skillloop
curl localhost:7331/health
```

## Configuration (environment variables)
| var | default | purpose |
|---|---|---|
| `SKILLLOOP_HOME` | `~/.skillloop` | data directory (SQLite + skill files). Mount a volume here. |
| `SKILLLOOP_PORT` | `7331` | HTTP port |
| `SKILLLOOP_LOG` | off | `json` or `text` — structured logs with request ids |
| `SKILLLOOP_PROVIDER` | auto | `anthropic` \| `openai` \| `manual` \| `fake` |
| `OPENAI_BASE_URL` | — | for OpenRouter / Groq / Ollama / vLLM |
| `OPENAI_API_KEY` / `OPENAI_API_KEYS` | — | single key or comma-separated pool (rotates on 429) |
| `SKILLLOOP_MODEL` + per-role overrides | — | see README |
| `SKILLLOOP_REQUIRE_APPROVAL` | 0 | human approval before any skill goes active |

## Observability
With `SKILLLOOP_LOG=json`, every operation emits one structured event carrying a `request_id`, so a single
`learn`/`recall` is traceable end to end. Key events: `learn.start/ok/error`, `trace.sanitized` (redaction and
size stats), `recall` (task + hits), `skill.promote`/`skill.demote` (skill + version). `GET /metrics` returns
episode success rate, library size, promotion/demotion counts, regression rate.

## Health & lifecycle
- `GET /health` — readiness: confirms the DB answers, returns version + library size (503 if the DB is down).
- `GET /version` — build version.
- **Graceful shutdown**: SIGTERM/SIGINT stop accepting new requests, let in-flight ones finish (daemon
  threads + WAL), then close the DB. Container `HEALTHCHECK` is wired to `/health`.

## Database: migration, backup, recovery
- Schema is versioned via SQLite `user_version`; `skillloop.migrate.migrate()` runs pending migrations
  automatically on startup and is idempotent.
- Backup (safe while running): `python -c "from skillloop.migrate import backup; backup('$SKILLLOOP_HOME/skillloop.db')"`
  uses SQLite's online backup API. Schedule it against the volume.
- Restore: stop writers, then `from skillloop.migrate import restore; restore(backup, db_path)`.

## Concurrency & resource limits
- One SQLite connection per thread (WAL, `busy_timeout=30000`, `synchronous=NORMAL`). Verified: 8 concurrent
  threads x 25 interleaved learn+recall, 0 errors.
- Single-host by design: SQLite on a local/attached volume, not a shared network filesystem. For multiple
  server instances, front one writer or shard by tenant — cross-host SQLite is not supported.
- Trace size is hard-capped at 200k serialized chars (enforced with a closing assertion), so one hostile
  trace cannot exhaust memory or context.
- Retrieval is an in-process scan: ~9 ms/query at 200 skills, linear. Add an index beyond a few thousand.

## Security posture
Traces carry untrusted tool output. Layers: secret redaction before storage, size caps, regex injection scan,
an LLM injection judge on risky skills, quarantine-by-default with human-approval mode, and provenance on
every recall. Red-team suite: `pytest tests/test_redteam.py`. Do NOT expose the HTTP server to untrusted
networks without an auth proxy — it has no built-in authentication (documented limitation).
