/* ============================================================================
   PhishGuard Supabase singleton client

   Loaded AFTER vendor/supabase.js (which exposes window.supabase) and
   supabase-config.js (which sets window.SUPABASE_CONFIG).

   Exposes window.pg as the app-wide namespace:
     window.pg.supabase  — the underlying supabase-js client
     window.pg.auth      — auth helpers (signIn, signOut, currentUser, onChange)

   app.js consumes from window.pg, NOT directly from window.supabase, so the
   indirection lets us swap the backend later if needed without churning
   every call site.
   ============================================================================ */

(function () {
    'use strict';

    if (!window.supabase || typeof window.supabase.createClient !== 'function') {
        console.error('[PhishGuard] supabase-js UMD bundle did not load — '
                      + 'check static/js/vendor/supabase.js is reachable.');
        return;
    }
    if (!window.SUPABASE_CONFIG || !window.SUPABASE_CONFIG.SUPABASE_URL) {
        console.error('[PhishGuard] SUPABASE_CONFIG not populated — '
                      + 'fill in static/js/supabase-config.js.');
        return;
    }

    const { SUPABASE_URL, SUPABASE_ANON_KEY } = window.SUPABASE_CONFIG;

    // Guard against shipping the placeholder by accident.
    if (SUPABASE_URL.includes('YOUR_PROJECT_REF') ||
        SUPABASE_ANON_KEY.includes('YOUR_ANON_KEY')) {
        console.warn('[PhishGuard] Supabase config still has placeholder values. '
                     + 'Edit static/js/supabase-config.js to point at your project.');
    }

    const client = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: {
            // Persist the session in localStorage so refreshing the renderer
            // doesn't kick the user out. supabase-js handles refresh tokens
            // automatically.
            persistSession: true,
            autoRefreshToken: true,
            // OAuth redirects come back via the renderer URL; let supabase-js
            // pull the token out automatically when it sees it in the hash.
            detectSessionInUrl: true,
            // 'implicit' flow returns the access token directly in the URL
            // hash. Avoids the PKCE code-exchange step that, in Electron,
            // can fail because the popup's localStorage may be isolated
            // from the main window where the code-verifier was stashed.
            flowType: 'implicit',
        },
    });

    window.pg = window.pg || {};
    window.pg.supabase = client;

    /* Helpers for the user_accounts table. The "accounts" here are
       Outlook mailboxes the PhishGuard user has connected — names +
       emails only, no tokens (those live in Flask SessionState memory).
       Wrapped here so app.js doesn't deal with user_id / upsert. */
    window.pg.accounts = {
        async list() {
            const { data, error } = await client
                .from('user_accounts')
                .select('email, display_name')
                .order('added_at', { ascending: true });
            if (error) throw error;
            return data.map((row) => ({
                email: row.email,
                name: row.display_name || '',
            }));
        },
        async add({ email, name }) {
            const { data: u } = await client.auth.getUser();
            if (!u || !u.user) throw new Error('not signed in');
            const { error } = await client
                .from('user_accounts')
                .upsert({
                    user_id: u.user.id,
                    email: email,
                    display_name: name || null,
                });
            if (error) throw error;
        },
        async remove(email) {
            const { error } = await client
                .from('user_accounts')
                .delete()
                .eq('email', email);
            if (error) throw error;
        },
    };

    /* Helpers for the starred_emails table. Centralised here so app.js
       doesn't have to know about user_id, RLS, or upsert semantics — it
       just calls `pg.starred.add(messageId)`. */
    window.pg.starred = {
        async list() {
            const { data, error } = await client
                .from('starred_emails')
                .select('message_id');
            if (error) throw error;
            return data.map((row) => row.message_id);
        },
        async add(messageId) {
            const { data: u } = await client.auth.getUser();
            if (!u || !u.user) throw new Error('not signed in');
            const { error } = await client
                .from('starred_emails')
                .upsert({ user_id: u.user.id, message_id: messageId });
            if (error) throw error;
        },
        async remove(messageId) {
            const { error } = await client
                .from('starred_emails')
                .delete()
                .eq('message_id', messageId);
            if (error) throw error;
        },
    };

    window.pg.auth = {
        signIn: (email, password) =>
            client.auth.signInWithPassword({ email, password }),
        /* Sign in via the Supabase Azure OAuth provider.
           - skipBrowserRedirect: we open the URL ourselves in a popup
             (Electron child window) so the main renderer doesn't navigate
             away from the app.
           - scopes: Mail.ReadWrite + User.Read for Outlook access;
             offline_access so Supabase stashes a refresh token.
           - redirectTo: back to our renderer URL. supabase-js detects
             the session in the URL hash via detectSessionInUrl: true.
           Returns { data, error }; caller opens data.url in a window. */
        signInWithMicrosoft: async (overrides) => {
            overrides = overrides || {};
            return client.auth.signInWithOAuth({
                provider: 'azure',
                options: {
                    scopes: 'email Mail.ReadWrite User.Read offline_access',
                    redirectTo: overrides.redirectTo || window.location.origin,
                    skipBrowserRedirect: true,
                    /* prompt=select_account → Microsoft shows the
                       account picker. Caller (external-browser flow)
                       overrides redirectTo to point at Flask's
                       /auth/external-callback so the browser hands
                       the tokens back to the desktop app. */
                    queryParams: {
                        prompt: 'select_account',
                    },
                },
            });
        },
        /* scope: 'local' only clears the local session — no server round
           trip. Fast and reliable even when the Supabase Auth endpoint
           is slow or the access token is already expired server-side.
           The user's JWT remains valid until its natural expiry (~1h),
           which is fine for desktop apps where the client + server are
           the same machine. */
        signOut: () => client.auth.signOut({ scope: 'local' }),
        currentUser: async () => {
            const { data } = await client.auth.getUser();
            return data && data.user ? data.user : null;
        },
        currentSession: async () => {
            const { data } = await client.auth.getSession();
            return data && data.session ? data.session : null;
        },
        onChange: (cb) => client.auth.onAuthStateChange(cb),
    };

    // Tiny boot diagnostic — useful while bringing the integration up.
    client.auth.getSession().then(({ data }) => {
        if (data && data.session) {
            console.log('[PhishGuard] Supabase session restored for '
                        + data.session.user.email);
        } else {
            console.log('[PhishGuard] No active Supabase session.');
        }
    });

    /* OAuth callback popup. Detected via URL — auth params land here
       regardless of whether window.opener works in this Electron build.
       Poll for the session to arrive (supabase-js exchanges the PKCE
       code internally) and close as soon as it does. */
    window.pg.isOAuthCallback = (
        window.location.hash.includes('access_token') ||
        window.location.search.includes('code=') ||
        window.location.hash.includes('error')
    );

    if (window.pg.isOAuthCallback) {
        console.log('[PhishGuard] OAuth callback popup detected. '
                    + 'URL search=' + window.location.search.slice(0, 80)
                    + ' hash=' + window.location.hash.slice(0, 80));

        let polls = 0;
        const maxPolls = 30; // 15s total
        const poll = setInterval(async () => {
            polls++;
            try {
                const { data, error } = await client.auth.getSession();
                if (error) {
                    console.warn('[PhishGuard] popup getSession error:', error);
                }
                if (data && data.session) {
                    console.log('[PhishGuard] popup: session acquired for '
                                + data.session.user.email + ', handing off');
                    clearInterval(poll);
                    // Hand the session over to the main window via
                    // postMessage. This works even if the popup's
                    // localStorage is isolated from the main window
                    // (Electron sometimes partitions child-window storage).
                    try {
                        if (window.opener && !window.opener.closed) {
                            window.opener.postMessage({
                                type: 'pg-oauth-session',
                                session: data.session,
                            }, window.location.origin);
                            console.log('[PhishGuard] popup: postMessage sent to opener');
                        }
                    } catch (e) {
                        console.warn('[PhishGuard] popup: postMessage failed:', e);
                    }
                    setTimeout(() => { try { window.close(); } catch (_e) {} }, 200);
                    return;
                }
            } catch (e) {
                console.warn('[PhishGuard] popup poll threw:', e);
            }
            if (polls >= maxPolls) {
                clearInterval(poll);
                console.error('[PhishGuard] popup timed out — no session after 15s. '
                              + 'URL was ' + window.location.href.slice(0, 200));
                console.error('[PhishGuard] supabase keys in localStorage:',
                              Object.keys(localStorage).filter(k => k.includes('supabase') || k.startsWith('sb-')));
            }
        }, 500);
    }

    /* Main-window listener for the popup's postMessage handoff. Installs
       the received session into this client even if localStorage didn't
       bridge across windows. */
    if (!window.pg.isOAuthCallback) {
        window.addEventListener('message', async (event) => {
            if (event.origin !== window.location.origin) return;
            const msg = event.data;
            if (!msg || msg.type !== 'pg-oauth-session') return;
            if (!msg.session || !msg.session.access_token) return;
            try {
                const { error } = await client.auth.setSession({
                    access_token: msg.session.access_token,
                    refresh_token: msg.session.refresh_token || '',
                });
                if (error) {
                    console.warn('[PhishGuard] setSession from popup failed:', error);
                } else {
                    console.log('[PhishGuard] session installed from popup postMessage');
                }
            } catch (e) {
                console.warn('[PhishGuard] setSession threw:', e);
            }
        });
    }
})();
