-- ============================================================================
-- PhishGuard — Initial Supabase schema
-- ============================================================================
-- Migration: 0001
-- Applies to a fresh Supabase project. Idempotent where possible; should be
-- run in the Supabase SQL Editor with the "Run" button.
--
-- Tables created:
--   sender_profiles   — per-user behavioral fingerprints (replaces sender_dna.json)
--   scan_history      — per-user scan results (replaces local_scan_history.json)
--   starred_emails    — per-user starred messages (replaces pg-starred localStorage)
--   user_accounts     — per-user Outlook account list (names + emails ONLY, no tokens)
--   user_settings     — per-user UI preferences (theme, font size, etc.)
--   threat_reports    — append-only cross-user reputation reports
--   threat_domain_summary  — materialized view aggregating threat_reports
--
-- Security model:
--   • Every user-data table has RLS enabled.
--   • Policies restrict each row to its owner via auth.uid() = user_id.
--   • threat_reports is the one cross-user table: any authenticated user
--     can SELECT all reports; INSERT only with their own reporter_id;
--     no UPDATE or DELETE (append-only by policy + lack of grant).
--   • The Outlook access token is NEVER persisted here. Only Supabase's
--     own auth.users + auth.identities tables hold the provider_token.
--
-- DO NOT EVER store: email body content, attachments, raw URLs from
-- emails, or anything that would let someone reconstruct a user's mailbox.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Schema versioning
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ----------------------------------------------------------------------------
-- Helper: touch updated_at on row update
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ----------------------------------------------------------------------------
-- sender_profiles
-- ----------------------------------------------------------------------------
CREATE TABLE sender_profiles (
    user_id      UUID  NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    sender_email TEXT  NOT NULL,
    profile      JSONB NOT NULL,
    email_count  INT   NOT NULL DEFAULT 0 CHECK (email_count >= 0),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, sender_email)
);

CREATE INDEX idx_sender_profiles_recent
    ON sender_profiles(user_id, updated_at DESC);

CREATE TRIGGER sender_profiles_touch
    BEFORE UPDATE ON sender_profiles
    FOR EACH ROW EXECUTE FUNCTION public.touch_updated_at();


-- ----------------------------------------------------------------------------
-- scan_history
-- ----------------------------------------------------------------------------
CREATE TABLE scan_history (
    id            BIGSERIAL PRIMARY KEY,
    user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    message_id    TEXT NOT NULL,
    sender_domain TEXT,
    prediction    SMALLINT NOT NULL CHECK (prediction IN (0, 1)),
    confidence    REAL     NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    scanned_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signals       JSONB
);

CREATE INDEX idx_scan_history_user_time
    ON scan_history(user_id, scanned_at DESC);

CREATE INDEX idx_scan_history_user_domain
    ON scan_history(user_id, sender_domain);


-- ----------------------------------------------------------------------------
-- starred_emails
-- ----------------------------------------------------------------------------
CREATE TABLE starred_emails (
    user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    starred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, message_id)
);


-- ----------------------------------------------------------------------------
-- user_accounts  (names + emails only — NO tokens, ever)
-- ----------------------------------------------------------------------------
CREATE TABLE user_accounts (
    user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    email        TEXT NOT NULL,
    display_name TEXT,
    added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, email)
);


-- ----------------------------------------------------------------------------
-- user_settings  (theme, font size, sidebar state, future flags)
-- ----------------------------------------------------------------------------
CREATE TABLE user_settings (
    user_id    UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    data       JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER user_settings_touch
    BEFORE UPDATE ON user_settings
    FOR EACH ROW EXECUTE FUNCTION public.touch_updated_at();


-- ----------------------------------------------------------------------------
-- threat_reports  (append-only, cross-user)
-- ----------------------------------------------------------------------------
CREATE TABLE threat_reports (
    id          BIGSERIAL PRIMARY KEY,
    reporter_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    domain      TEXT NOT NULL,
    category    TEXT NOT NULL CHECK (category IN ('phishing', 'safe')),
    reported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (reporter_id, domain, category)
);

CREATE INDEX idx_threat_reports_domain
    ON threat_reports(domain);


-- ----------------------------------------------------------------------------
-- threat_domain_summary  (materialized view aggregating threat_reports)
-- Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY threat_domain_summary;
-- Run that periodically (e.g. nightly cron) or after big report batches.
-- ----------------------------------------------------------------------------
CREATE MATERIALIZED VIEW threat_domain_summary AS
SELECT
    domain,
    COUNT(*) FILTER (WHERE category = 'phishing') AS phishing_count,
    COUNT(*) FILTER (WHERE category = 'safe')     AS safe_count,
    COUNT(DISTINCT reporter_id)                   AS reporter_count,
    MAX(reported_at)                              AS last_reported
FROM threat_reports
GROUP BY domain;

CREATE UNIQUE INDEX idx_threat_summary_domain
    ON threat_domain_summary(domain);

GRANT SELECT ON threat_domain_summary TO authenticated;


-- ============================================================================
-- Auto-provision user_settings row on signup
-- ============================================================================
-- Why: the app reads user_settings via SELECT then patches with UPDATE.
-- Pre-creating an empty row means client code never has to handle "row missing"
-- as a special case.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.user_settings (user_id, data)
    VALUES (NEW.id, '{}'::jsonb)
    ON CONFLICT (user_id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();


-- ============================================================================
-- Row Level Security
-- ============================================================================
-- Every table that holds per-user data gets RLS turned on. Without these
-- policies, the anon key (which ships in the Electron bundle) would let any
-- attacker read every user's data. RLS is mandatory, not optional.
-- ============================================================================

ALTER TABLE sender_profiles  ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_history     ENABLE ROW LEVEL SECURITY;
ALTER TABLE starred_emails   ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_accounts    ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_settings    ENABLE ROW LEVEL SECURITY;
ALTER TABLE threat_reports   ENABLE ROW LEVEL SECURITY;

-- Per-user tables: users only see / mutate their own rows
CREATE POLICY "own sender_profiles"
    ON sender_profiles FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "own scan_history"
    ON scan_history FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "own starred_emails"
    ON starred_emails FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "own user_accounts"
    ON user_accounts FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "own user_settings"
    ON user_settings FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- threat_reports: authenticated users can SELECT all rows (so the app can
-- look up domain reputation) and INSERT rows attributed to themselves.
-- No UPDATE / DELETE policies → append-only.
CREATE POLICY "auth users read all threat reports"
    ON threat_reports FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "auth users insert own threat report"
    ON threat_reports FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = reporter_id);


-- ============================================================================
-- Role grants
-- ============================================================================
-- RLS filters WHICH ROWS a role can touch; GRANT controls WHETHER a role
-- can touch the table at all. With dashboard "Auto-expose new tables" OFF,
-- Supabase does not run the usual auto-grant step, so we issue these
-- explicitly. Without them, the renderer hits 42501 "permission denied
-- for table X" before RLS even gets a chance to filter.
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON
    public.sender_profiles,
    public.scan_history,
    public.starred_emails,
    public.user_accounts,
    public.user_settings
TO authenticated;

-- threat_reports is append-only by RLS; SELECT + INSERT only.
GRANT SELECT, INSERT ON public.threat_reports TO authenticated;

-- BIGSERIAL primary keys (scan_history.id, threat_reports.id) need the
-- authenticated role to read/advance the underlying sequence.
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;


-- ============================================================================
-- Record migration applied
-- ============================================================================
INSERT INTO schema_migrations (version) VALUES (1)
ON CONFLICT (version) DO NOTHING;
