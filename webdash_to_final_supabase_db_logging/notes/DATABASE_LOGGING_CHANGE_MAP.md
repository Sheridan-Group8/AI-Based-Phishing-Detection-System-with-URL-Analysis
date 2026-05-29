# Database And Logging Change Map

This is the focused map from `webdash copy` to the final Supabase version.

## Webdash Copy Baseline

`webdash copy/app.py` stores runtime data locally/in memory:

- `scan_results = {}` is global process memory.
- `_live_emails`, `_graph_token`, `_graph_user`, and `_live_mode` are global.
- `LOG_FILE = Path(__file__).parent / "phishguard_log.txt"`.
- `_log_session_event()` always writes login/logout details to that file.
- `/api/log` and `/api/log/download` read the local file directly.
- Scan results are not persisted to a database.
- Sender DNA is not persisted in Supabase.
- No RLS-backed user data model exists.

## Final Supabase Version

The final version adds a database-backed model while keeping email body content
out of the database.

### Supabase Schema

Add `supabase/migrations/0001_initial_schema.sql`.

Tables:

- `sender_profiles`: per-user sender behavior profiles.
- `scan_history`: per-user scan result metadata.
- `starred_emails`: per-user starred message IDs.
- `user_accounts`: connected mailbox names/emails only, no tokens.
- `user_settings`: per-user UI preferences.
- `threat_reports`: append-only cross-user domain reports.
- `schema_migrations`: migration bookkeeping.

View:

- `threat_domain_summary`: aggregate report counts by domain.

Important security rule:

- Email body content, attachments, and raw mailbox data are not stored.
- Outlook tokens are not stored in app tables.
- RLS scopes user-owned data by `auth.uid()`.

### Backend Supabase Plumbing

Port these concepts from `backend_reference/app.py`:

- `_load_supabase_config()`
  Reads `static/js/supabase-config.js` so Flask and the frontend use the same
  project URL and anon key.

- `_request_jwt()`
  Reads the Supabase access token from the `Authorization: Bearer ...` request
  header.

- `_jwt_user_id(jwt)`
  Decodes the JWT subject so backend inserts can include `user_id` or
  `reporter_id`.

- `_supabase_request(...)`
  Makes PostgREST calls to `/rest/v1/...` using the anon key plus the user's
  JWT. RLS handles ownership checks.

### Scan History Persistence

In `webdash copy`, scans end at:

```python
scan_results[idx] = result
```

In the final version, scan results still stay in session memory for UI speed,
but metadata is also inserted into Supabase:

- Build metadata with `_scan_history_row(...)`.
- Add `user_id` from `_jwt_user_id(jwt)`.
- Insert into `scan_history`.

The row stores only metadata:

- `message_id`
- `sender_domain`
- `prediction`
- `confidence`
- `signals`
- `scanned_at`

It does not store message body text.

### Sender DNA Persistence

In the final version:

- `_sender_profiles` is only an in-process cache.
- `_persist_sender_profile_to_supabase(...)` upserts profiles into
  `sender_profiles`.
- `/api/sender-dna/<path:sender_addr>` builds the profile and persists it when
  a Supabase JWT is available.

### Threat Reports / Local History

The final version keeps a local offline fallback:

- `local_scan_history.json`

But explicit user reports also insert into Supabase:

- `/api/report-sender`
- table: `threat_reports`
- fields: `reporter_id`, `domain`, `category`

This is the database-backed replacement for purely local/community history.

### Logging Changes

`webdash copy` writes `phishguard_log.txt` beside the project code and logging
is always active.

The final version changes that:

- `LOG_FILE = USER_DATA_DIR / "phishguard_log.txt"`
- `_LOGGING_ENABLED = os.environ.get("PHISHGUARD_LOG") == "1"`
- `_log_session_event()` returns immediately unless logging is enabled.
- `/api/log` and `/api/log/download` are gated by origin/launch-secret checks.
- `/api/log/clear` is added.
- `.env.example` documents `PHISHGUARD_LOG`.

This is a privacy improvement because login/logout logs include user names,
email addresses, and scan counts.

### Frontend Changes Required

Add script loading in `templates/index.html`:

```html
<script src="/static/js/vendor/supabase.js"></script>
<script src="/static/js/supabase-config.js"></script>
<script src="/static/js/supabase-client.js"></script>
```

Then update frontend API calls to:

- fetch the Supabase session.
- attach `Authorization: Bearer <access_token>` when available.
- call `/api/csrf` and attach `X-CSRF-Token` for mutating routes.
- use `window.pg.starred` for starred email persistence.
- use `window.pg.accounts` for connected account metadata.
- use final logging/settings UI behavior from `frontend_reference/app.js`.

## Replacement Shortcut

For the fastest move to final behavior, replace these `webdash copy` files:

- `app.py` with `backend_reference/app.py`
- `static/js/app.js` with `frontend_reference/app.js`
- `static/css/style.css` with `frontend_reference/style.css`
- `templates/index.html` with `frontend_reference/index.html`

Then add:

- `supabase/`
- `static/js/supabase-client.js`
- `static/js/supabase-config.js`
- `static/js/vendor/supabase.js`
- final `.env.example` and `requirements.txt` as needed

That replacement brings in more than database/logging changes, but it matches
the final Supabase project structure.
