# Webdash -> Final Supabase Database/Logging Migration Kit

This folder contains the files and change notes needed to move the database
and logging parts of `webdash copy` toward the final Supabase-backed version.

## What This Adds

- Supabase schema for persisted app data.
- Row-level security for per-user data.
- Supabase browser client files.
- Backend reference implementation from the final version.
- Frontend reference implementation from the final version.
- Logging changes that move logs out of the project folder and make logging
  opt-in.

## Main Files

- `supabase/migrations/0001_initial_schema.sql`
  Creates `sender_profiles`, `scan_history`, `starred_emails`,
  `user_accounts`, `user_settings`, `threat_reports`, RLS policies, grants,
  and the `threat_domain_summary` materialized view.

- `static/js/supabase-config.example.js`
  Template for your Supabase URL and anon key. Copy it into the app as
  `static/js/supabase-config.js`.

- `static/js/supabase-client.js`
  Final version's Supabase wrapper. It exposes `window.pg.supabase`,
  `window.pg.accounts`, `window.pg.starred`, and auth/settings helpers.

- `static/js/vendor/supabase.js`
  Vendored Supabase JavaScript client used by the final app.

- `backend_reference/app.py`
  Final backend reference. This contains the Supabase persistence and revised
  logging implementation.

- `frontend_reference/app.js`, `frontend_reference/index.html`,
  `frontend_reference/style.css`
  Final frontend reference files that pair with the Supabase-backed backend.

- `notes/DATABASE_LOGGING_CHANGE_MAP.md`
  Practical map of what changed from `webdash copy` and where to port it.

## Suggested Apply Order

1. Create a Supabase project.
2. Run `supabase/migrations/0001_initial_schema.sql` in the Supabase SQL editor.
3. Copy `static/js/supabase-config.example.js` to
   `webdash copy/static/js/supabase-config.js` and fill in your project URL and
   anon key.
4. Copy `static/js/supabase-client.js` and `static/js/vendor/supabase.js` into
   `webdash copy/static/js/`.
5. Use `backend_reference/app.py` as the source of truth for backend database
   and logging behavior.
6. Use the frontend reference files to wire Supabase auth, Authorization
   headers, starred emails, settings, and logging controls into the UI.

For the quickest path to the final behavior, replace the matching files in
`webdash copy` with the reference files. For a narrower/manual port, follow the
change map in `notes/DATABASE_LOGGING_CHANGE_MAP.md`.
