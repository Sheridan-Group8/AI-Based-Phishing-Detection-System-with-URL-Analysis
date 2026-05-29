/* ============================================================================
   Supabase project configuration

   Copy this file to:
     webdash copy/static/js/supabase-config.js

   Then fill in values from:
     Supabase Dashboard -> Project Settings -> API

   The anon key is public by design. Never put the service_role key here.
   RLS policies in supabase/migrations/0001_initial_schema.sql protect data.
   ============================================================================ */

window.SUPABASE_CONFIG = {
    SUPABASE_URL: 'https://YOUR_PROJECT_REF.supabase.co',
    SUPABASE_ANON_KEY: 'YOUR_ANON_PUBLIC_KEY',
};
