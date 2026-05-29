# Microsoft OAuth via Supabase — setup steps

This is the only Supabase integration step that needs dashboard work by
you, not code. Once you've done these steps, ping me and I'll swap the
email/password sign-in modal for a single "Sign in with Microsoft" button
and wire Flask to use the `provider_token` from the Supabase session for
Outlook Graph API calls.

## What this gives you

Before: two separate sign-ins —
1. PhishGuard email/password (Supabase identity)
2. Click Connect → Microsoft OAuth in a popup → Outlook access

After: one sign-in —
- Click "Sign in with Microsoft" → Microsoft consent →
  Supabase issues your session JWT AND passes back the Outlook
  access/refresh tokens in the same session.

Cleaner UX, one identity, simpler code on our side. Same security model
(PKCE, public client, no client secret in the app).

---

## Step 1 — Azure: add a Web redirect URI for Supabase

Your existing Azure app registration has a "Mobile and desktop
applications" redirect (`http://localhost:5050/auth/callback`). That
stays — Supabase needs a **Web** platform URI too.

1. Go to https://portal.azure.com → **Microsoft Entra ID** → **App registrations** → your PhishGuard app.
2. Left sidebar → **Authentication**.
3. **Add a platform** → **Web**.
4. **Redirect URI:**
   `https://tbjloagdkhytdnfqeonr.supabase.co/auth/v1/callback`
   (copy this exact URL — it's your Supabase project's callback)
5. **Implicit grant** and **Hybrid flows** — both stay UNCHECKED.
6. Click **Configure**.
7. **Front-channel logout URL** — leave blank.
8. Save.

Your app should now have BOTH platforms listed under Authentication:
- Mobile and desktop applications (existing): `http://localhost:5050/auth/callback`
- Web (new): `https://tbjloagdkhytdnfqeonr.supabase.co/auth/v1/callback`

---

## Step 2 — Azure: create a client secret

Supabase's server-side OAuth flow requires a client secret (this is the
"Web platform" rule we noted before — Supabase is the confidential
client, the desktop app is not).

1. Same Azure app registration → **Certificates & secrets** → **Client secrets**.
2. **New client secret**.
3. **Description:** `Supabase OAuth (PhishGuard)`.
4. **Expires:** 24 months (or whatever your policy allows).
5. Click **Add**.
6. **IMMEDIATELY copy the "Value"** (not the Secret ID). You won't see
   it again after leaving this page.

The secret will live in Supabase's backend only — it does NOT ship in
the Electron app, which is what makes this safe.

---

## Step 3 — Azure: API permissions (verify, don't recreate)

These should already be set from your earlier Outlook OAuth setup:
- `Mail.ReadWrite` (Delegated)
- `User.Read` (Delegated)

Add if missing:
- `offline_access` (Delegated) — lets Supabase store a refresh token so
  Outlook access doesn't expire after an hour.

Grant admin consent if your tenant requires it (most personal accounts
don't).

---

## Step 4 — Supabase: enable Azure as an Auth provider

1. https://supabase.com/dashboard → your project → **Authentication** → **Providers**.
2. Find **Azure** in the list. Expand it.
3. **Enable Sign in with Azure** — toggle ON.
4. Fill in:
   - **Azure Tenant URL:** `https://login.microsoftonline.com/common` (or your specific tenant ID if you want to restrict to one org)
   - **Application (Client) ID:** the UUID from your Azure app registration's Overview page (same one currently in PhishGuard's Connect modal)
   - **Secret Value:** paste the client secret from Step 2
5. **Callback URL:** verify it matches what you used in Step 1 (Supabase shows it on this page — copy from here back to Azure if needed).
6. Click **Save**.

---

## Step 5 — Supabase: verify Site URL + redirect URLs

1. **Authentication** → **URL Configuration**.
2. **Site URL:** `http://localhost:5050` (the Electron renderer origin)
3. **Redirect URLs (allow list):**
   - `http://localhost:5050/**`
   - `http://127.0.0.1:5050/**`
4. Save.

These are the URLs Supabase will redirect back to AFTER Microsoft auth.
Anything not in this list is rejected, which prevents the token-stealing
"open redirect" class of attack.

---

## Step 6 — ping me

Once these six steps are done, tell me. I'll:
- Add a "Sign in with Microsoft" button to the renderer that calls
  `pg.auth.signInWithOAuth({ provider: 'azure', scopes: 'Mail.ReadWrite User.Read offline_access' })`.
- Remove the email/password modal (or keep it as a fallback for dev).
- Wire Flask to receive the `provider_token` and use it for Graph API
  calls instead of the existing per-PhishGuard PKCE flow.
- Verify end-to-end: clicking "Sign in with Microsoft" results in a
  Microsoft consent popup, then the app boots with both Supabase
  identity AND Outlook inbox access.

---

## What to do if something breaks

| Symptom | Likely cause |
|---|---|
| `AADSTS50011 redirect_uri mismatch` after Microsoft consent | Azure Web redirect URI doesn't exactly match Supabase callback URL. Recheck both. |
| `unauthorized_client` error | Client secret expired, or you pasted the Secret ID instead of the Value. Recreate. |
| Supabase fires the callback but no `provider_token` in session | `offline_access` scope wasn't granted. Add it in Azure → API permissions. |
| `invalid_grant` mid-session, several hours in | Refresh token expired (didn't grant offline_access) OR the user changed their Microsoft password. Sign out and back in. |
| Microsoft consent screen says "needs admin approval" | Your tenant requires admin consent for `Mail.ReadWrite`. Either request it or use a personal Microsoft account that doesn't have this policy. |

---

## What does NOT change about your security model

- The Outlook access token is still treated as sensitive — Supabase
  stores the refresh token encrypted at rest, and the access token is
  only handed to the renderer briefly during sign-in. From there it
  flows to Flask `SessionState` memory exactly like today.
- The Supabase anon key + client ID stay public, as designed.
- RLS still scopes every per-user table to `auth.uid()`.
- The client_secret you generated in Step 2 lives ONLY in the Supabase
  dashboard. It is never sent to the Electron renderer or Flask. If it
  leaks, rotate it (Azure → Certificates & secrets → delete + create
  new + paste into Supabase). No user action needed for rotation.
