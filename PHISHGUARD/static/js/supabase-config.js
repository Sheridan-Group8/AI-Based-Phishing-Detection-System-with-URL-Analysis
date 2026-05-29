/* ============================================================================
   Supabase project configuration

   How to fill this in:
     1. Open your Supabase project dashboard.
     2. Project Settings → API.
     3. Copy "Project URL" into SUPABASE_URL.
     4. Copy the "anon public" key into SUPABASE_ANON_KEY.

   Both values are PUBLIC by design — they ship in the renderer bundle
   and any user can read them. RLS is what protects the data, not these
   values. Treat them like the Azure client ID: visible, but not secret.

   NEVER paste the service_role key here. It bypasses RLS.
   ============================================================================ */

window.SUPABASE_CONFIG = {
    SUPABASE_URL: 'https://tbjloagdkhytdnfqeonr.supabase.co',
    SUPABASE_ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRiamxvYWdka2h5dGRuZnFlb25yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkyMjUzNDQsImV4cCI6MjA5NDgwMTM0NH0.69slWYLpVPDb5hoCncNGiUlDhpSCi37h9D2CkL71eoI',
};
