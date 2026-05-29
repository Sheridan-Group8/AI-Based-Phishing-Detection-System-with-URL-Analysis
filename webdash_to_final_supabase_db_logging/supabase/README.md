# Supabase setup for PhishGuard

This folder holds the database schema migrations for the PhishGuard
Supabase backend. Supabase does NOT ship a built-in migration tool; this
folder is the source of truth for the schema.

## Files

- `migrations/0001_initial_schema.sql` — initial tables, indexes,
  triggers, and RLS policies. Idempotent except where noted.

## How to apply

1. Open your project at https://supabase.com/dashboard.
2. Left sidebar → **SQL Editor**.
3. Click **New query**.
4. Paste the entire contents of the migration file.
5. Click **Run**.
6. Verify in **Table Editor** that all six tables exist with RLS enabled
   (look for the green shield icon next to each table name).

## Future migrations

When adding a new migration:

1. Create `migrations/0002_<short_name>.sql`.
2. Increment the `INSERT INTO schema_migrations (version) VALUES (N)` at
   the bottom.
3. Apply via SQL Editor the same way.
4. Commit the file to git so the schema history is replayable from
   scratch on a fresh project.

## Verifying RLS is correct

After applying, run this in the SQL Editor to confirm every user-data
table has RLS turned on:

```sql
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
```

Every row except `schema_migrations` should show `rowsecurity = true`.

To prove RLS actually isolates users, sign in as two different test
accounts in two browser sessions and verify that each only sees their own
rows in the Table Editor.

## What is NOT in Supabase

For privacy:

- Outlook OAuth access tokens (live in Flask `SessionState` memory only)
- Email body content, headers, attachments
- The pickled ML model

If you ever find yourself writing one of those into a Supabase table,
stop and reconsider the design.
