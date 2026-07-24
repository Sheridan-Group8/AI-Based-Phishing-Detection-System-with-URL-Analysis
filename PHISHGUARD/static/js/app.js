/* ================================================================
   PhishGuard Web Dashboard — Complete SPA JavaScript
   ================================================================ */

/* ======================== STATE ======================== */

/** Parse a JSON blob from localStorage without crashing the app if the value
 * was corrupted (malicious extension, aborted write, etc.). */
function safeJSONFromLocalStorage(key, fallback) {
    try {
        const raw = localStorage.getItem(key);
        if (!raw) return fallback;
        const parsed = JSON.parse(raw);
        // Guard against null/boolean/number when we expect an object or array
        if (parsed === null || typeof parsed !== typeof fallback) return fallback;
        return parsed;
    } catch (_e) {
        try { localStorage.removeItem(key); } catch (_) {}
        return fallback;
    }
}

const state = {
    emails: [],
    selectedIdx: null,           // display-only array index into state.emails
    scanResults: {},             // keyed by email.id (message id) — NOT by array index
    stats: { scanned: 0, threats: 0, safe: 0, dangerous: 0, suspicious: 0, safeTier: 0 },
    navView: 'inbox',
    theme: 'cupertino',
    mode: 'dark',
    fontSize: 11,
    compactList: false,
    inlineVisible: false,
    detailReportOpen: false,
    /* Starred emails are now stored in Supabase (starred_emails table)
       and synced into this map on sign-in. The map shape stays the same
       — { messageId: true } — so the existing render/toggle code is a
       thin patch on top. */
    starredEmails: {},
    /* Saved Outlook accounts (names + emails, NO tokens) — also synced
       from Supabase (user_accounts) on sign-in. */
    savedAccounts: [],
    detectionRules: {
        brand_name_tokens: {},
        brand_domains: [],
        risky_tlds: [],
        trusted_domains: [],
    },
    user: null,                  // current Supabase user (set by onChange)
    supabaseJwt: null,           // forwarded to Flask so it can call Supabase as the user
    /* Stash for provider tokens returned by the Supabase PKCE code exchange.
       They stay in memory only long enough to be forwarded to Flask. */
    pendingProviderToken: null,
    csrfToken: '',               // fetched from /api/csrf before any POST
    launchSecret: '',            // from window.electron (passed via preload) if present
};

/* ======================== SECURE FETCH ======================== */

/** Fetch the CSRF token for this session. Must succeed before any mutation. */
async function ensureCsrfToken() {
    if (state.csrfToken) return state.csrfToken;
    try {
        const r = await fetch('/api/csrf', { credentials: 'same-origin' });
        const data = await r.json();
        state.csrfToken = data.csrf || '';
    } catch (e) {
        console.error('Could not fetch CSRF token', e);
        state.csrfToken = '';
    }
    return state.csrfToken;
}

/** Force a fresh CSRF fetch. Needed after the OAuth popup flow, because
 * the popup's load of "/" may have replaced our pg_sid cookie with a new
 * one — the cached CSRF token then no longer matches Flask's session. */
async function refreshCsrfToken() {
    state.csrfToken = '';
    return ensureCsrfToken();
}

/** Thin fetch wrapper that automatically attaches the CSRF header on
 * mutating requests and the launch secret (if available) on every request.
 * All mutating API calls in this file go through apiFetch rather than fetch.
 */
async function apiFetch(url, opts) {
    opts = opts || {};
    const method = (opts.method || 'GET').toUpperCase();
    const headers = new Headers(opts.headers || {});
    if (method !== 'GET' && method !== 'HEAD') {
        const token = await ensureCsrfToken();
        if (token) headers.set('X-CSRF-Token', token);
    }
    if (state.launchSecret) headers.set('X-Launch-Secret', state.launchSecret);
    /* Forward the Supabase JWT so Flask can call Supabase as the
       authenticated user (writes to scan_history, sender_profiles, etc.
       go through RLS using this token, not the service_role key). */
    if (state.supabaseJwt) {
        headers.set('Authorization', 'Bearer ' + state.supabaseJwt);
    }
    opts.headers = headers;
    opts.credentials = opts.credentials || 'same-origin';
    return fetch(url, opts);
}

/* Hover tooltips for any element carrying a [data-tip] attribute. Rendered at
   body level (position: fixed) so they never clip inside scrolling panels like
   the analysis rail, and clamped to the viewport. */
(function initTips() {
    let tipEl = null;
    function show(el) {
        if (!tipEl) {
            tipEl = document.createElement('div');
            tipEl.className = 'pg-tip';
            document.body.appendChild(tipEl);
        }
        tipEl.textContent = el.getAttribute('data-tip') || '';
        tipEl.style.display = 'block';
        const r = el.getBoundingClientRect();
        const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
        let left = Math.max(8, Math.min(r.left + r.width / 2 - tw / 2, window.innerWidth - tw - 8));
        let top = r.top - th - 9;
        if (top < 8) top = r.bottom + 9;           // flip below if no room above
        tipEl.style.left = left + 'px';
        tipEl.style.top = top + 'px';
        requestAnimationFrame(() => { if (tipEl) tipEl.classList.add('show'); });
    }
    function hide() {
        if (tipEl) { tipEl.classList.remove('show'); tipEl.style.display = 'none'; }
    }
    document.addEventListener('mouseover', (e) => {
        const el = e.target.closest && e.target.closest('[data-tip]');
        if (el) show(el);
    });
    document.addEventListener('mouseout', (e) => {
        const el = e.target.closest && e.target.closest('[data-tip]');
        if (el) hide();
    });
})();

/* ======================== MESSAGE-ID HELPERS ======================== */

/** Safely resolve an array index to a stable message id. */
function emailIdFromIdx(idx) {
    if (idx === null || idx === undefined) return null;
    const email = state.emails[idx];
    return email ? (email.id || email.messageId || null) : null;
}

/** Retrieve the scan result for an array index (translates idx → id). */
function scanResultFor(idx) {
    const id = emailIdFromIdx(idx);
    return id ? state.scanResults[id] : undefined;
}

/** Check whether an email at this index has been scanned. */
function isScanned(idx) {
    return Boolean(scanResultFor(idx));
}

/* ======================== URL EXTRACTION (ReDoS-safe) ====================== */

/** Extract http(s) URLs from an arbitrary string without catastrophic
 * backtracking. Any input is truncated to 200 KB and each URL match to
 * 2048 chars before further work.
 */
const MAX_BODY_FOR_REGEX = 200_000;
const URL_REGEX_SAFE = /https?:\/\/[^\s<>"'`]{1,2048}/gi;

function extractUrlsSafe(text) {
    if (!text) return [];
    const capped = text.length > MAX_BODY_FOR_REGEX ? text.slice(0, MAX_BODY_FOR_REGEX) : text;
    const matches = capped.match(URL_REGEX_SAFE) || [];
    // Strip trailing punctuation without another regex pass
    return matches.map(u => u.replace(/[.,;:!?)>\]]+$/, ''));
}

/* ======================== EMAIL HTML SANITIZER ======================== */

/** Very defensive allowlist sanitizer for rich-HTML email preview.
 *
 * Even though the iframe runs with sandbox="" (no scripts, no same-origin)
 * and a strict CSP that blocks remote resources, we still strip tracking
 * vectors client-side before handing content to srcdoc. That keeps a hostile
 * email from leaking read receipts through background images declared in a
 * style attribute or from reshaping into a phishing form.
 */
const SAFE_TAGS = new Set([
    'a', 'b', 'strong', 'i', 'em', 'u', 's', 'br', 'p', 'div', 'span',
    'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'table', 'thead', 'tbody', 'tr', 'td', 'th', 'hr',
    'small', 'sub', 'sup', 'font', 'center',
]);
const SAFE_ATTRS = new Set(['align', 'colspan', 'rowspan']);  // minimal set

function sanitizeEmailHtml(raw) {
    if (!raw) return '';
    // Build a detached DOM (documents created via DOMParser don't execute scripts)
    const parsed = new DOMParser().parseFromString('<div>' + raw + '</div>', 'text/html');
    const root = parsed.body.firstElementChild;
    if (!root) return '';
    walkAndClean(root);
    /* Defang URL text inside the iframe content. The iframe is fully sandboxed
       (no allow-same-origin, no allow-scripts) and a CSP blocks remote loads,
       so the user can't actually click through — but the visible URL text is
       still copy-pasteable. Defanging it makes the rendered preview consistent
       with the plain-text view. */
    defangTextNodes(root);
    return root.innerHTML;
}

function defangTextNodes(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    let n;
    while ((n = walker.nextNode())) nodes.push(n);
    for (const node of nodes) {
        const text = node.nodeValue;
        if (!text || !/https?:\/\//i.test(text)) continue;
        node.nodeValue = text.replace(URL_REGEX_SAFE, (m) => defangUrl(m));
    }
}

function walkAndClean(node) {
    // Iterate over a snapshot so we can remove while we go
    const children = Array.from(node.children);
    for (const child of children) {
        const tag = child.tagName.toLowerCase();
        // Drop anything not on the allowlist outright (script, img, link, style,
        // iframe, form, input, object, embed, meta, base, audio, video, etc.)
        if (!SAFE_TAGS.has(tag)) {
            // Preserve text but drop the element itself
            const text = document.createTextNode(child.textContent || '');
            child.replaceWith(text);
            continue;
        }
        // Strip every attribute except the tiny whitelist
        const attrs = Array.from(child.attributes);
        for (const a of attrs) {
            const name = a.name.toLowerCase();
            if (name.startsWith('on')) { child.removeAttribute(a.name); continue; }
            if (name === 'style') { child.removeAttribute(a.name); continue; }
            if (tag === 'a' && name === 'href') {
                const v = (a.value || '').trim();
                // Only allow http/https/mailto links
                if (!/^(https?:|mailto:)/i.test(v)) {
                    child.removeAttribute(a.name);
                } else {
                    // Neutralise the link — the iframe blocks navigation anyway,
                    // but we also strip target so it can't try to open windows.
                    child.setAttribute('rel', 'noopener noreferrer');
                    child.removeAttribute('target');
                }
                continue;
            }
            if (!SAFE_ATTRS.has(name)) {
                child.removeAttribute(a.name);
            }
        }
        walkAndClean(child);
    }
}

/* ======================== CONSTANTS ======================== */
const AVATAR_COLORS = [
    '#FF6B6B', '#FF8E53', '#FFC947', '#51CF66',
    '#20C997', '#339AF0', '#5C7CFA', '#845EF7',
    '#CC5DE8', '#F06595', '#22B8CF', '#FF922B'
];

const RISKY_EXTENSIONS = [
    '.exe', '.bat', '.cmd', '.com', '.msi', '.scr', '.pif', '.vbs',
    '.js', '.wsf', '.docm', '.xlsm', '.pptm', '.jar', '.ps1', '.reg'
];

const URGENCY_WORDS = [
    'urgent', 'immediately', 'act now', 'expire', 'suspended',
    'verify your', 'confirm your', 'within 24 hours', 'limited time',
    'action required', 'account will be', 'unauthorized', 'unusual activity'
];

const MONEY_WORDS = [
    'wire transfer', 'bank account', 'credit card', 'payment',
    'invoice', 'bitcoin', 'cryptocurrency', 'prize', 'lottery',
    'million dollars', 'inheritance', 'beneficiary', 'fee'
];

const THREAT_WORDS = [
    'legal action', 'police', 'arrest', 'lawsuit', 'penalty',
    'terminate', 'suspend', 'revoke', 'deactivate', 'close your account'
];

const SECURITY_TIPS = [
    'Select an email and run a scan to receive a comprehensive threat analysis.',
    'Hover over links in the email body to preview their destination before clicking.',
    'SPF, DKIM, and DMARC authentication checks verify the sender\'s identity.',
    'Emails that create artificial urgency are a hallmark of phishing campaigns.',
    'Always verify URLs independently before interacting with them.',
    'Mismatched display names and email addresses are a common impersonation tactic.',
    'Legitimate organizations will never request credentials via email.',
    'Inconsistent grammar and formatting in professional correspondence may indicate fraud.',
];

/* ======================== UTILITY FUNCTIONS ======================== */

function hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return Math.abs(hash);
}

function getInitials(name) {
    if (!name) return '?';
    const parts = name.trim().split(/\s+/);
    if (parts.length === 1) {
        return parts[0].charAt(0).toUpperCase();
    }
    return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
}

function getAvatarColor(address) {
    if (!address) return AVATAR_COLORS[0];
    return AVATAR_COLORS[hashCode(address) % AVATAR_COLORS.length];
}

function formatDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const emailDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());

    if (emailDate.getTime() === today.getTime()) {
        return 'Today';
    }

    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    if (emailDate.getTime() === yesterday.getTime()) {
        return 'Yesterday';
    }

    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return months[date.getMonth()] + ' ' + date.getDate();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

function truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.substring(0, max) + '...' : str;
}

function extractDomain(email) {
    if (!email) return '';
    const atIdx = email.lastIndexOf('@');
    if (atIdx === -1) return email;
    return email.substring(atIdx + 1).toLowerCase();
}

function domainMatches(domain, allowedDomains) {
    const clean = String(domain || '').toLowerCase().replace(/\.$/, '');
    return allowedDomains.some(base => {
        const expected = String(base || '').toLowerCase().replace(/\.$/, '');
        return clean === expected || clean.endsWith('.' + expected);
    });
}

async function loadDetectionRules() {
    try {
        const res = await fetch('/api/detection-rules', { credentials: 'same-origin' });
        if (!res.ok) return;
        const rules = await res.json();
        state.detectionRules = {
            brand_name_tokens: rules.brand_name_tokens || {},
            brand_domains: Array.isArray(rules.brand_domains) ? rules.brand_domains : [],
            risky_tlds: Array.isArray(rules.risky_tlds) ? rules.risky_tlds : [],
            trusted_domains: Array.isArray(rules.trusted_domains) ? rules.trusted_domains : [],
        };
    } catch (e) {
        console.warn('[PhishGuard] detection rules not loaded:', e);
    }
}

/* Convert a URL into a visually defanged form so it cannot be copy-pasted
   and accidentally clicked downstream. http(s):// → hxxp(s)://, every dot
   becomes [.], and @ becomes [@]. The original URL stays in data-url so
   the user can deliberately reveal it. */
function defangUrl(url) {
    if (!url) return '';
    return String(url)
        .replace(/^https:\/\//i, 'hxxps://')
        .replace(/^http:\/\//i, 'hxxp://')
        .replace(/\./g, '[.]')
        .replace(/@/g, '[@]');
}

function getExtension(filename) {
    if (!filename) return '';
    const dotIdx = filename.lastIndexOf('.');
    if (dotIdx === -1) return '';
    return filename.substring(dotIdx).toLowerCase();
}

function isRiskyExtension(filename) {
    const ext = getExtension(filename);
    return RISKY_EXTENSIONS.includes(ext);
}

/* Write text into a single-line truncated element and only mark it as
   tooltip-worthy when the content actually overflows. The detection
   has to happen after the layout settles, so we defer one frame. We
   also stash the full text on `title` so screen readers / keyboard
   users get the same info. The actual hover popup is rendered by
   #pgOverflowTip below — a CSS ::after on the element doesn't work
   because `overflow: hidden` (needed for the ellipsis) would clip it. */
function setTextWithTooltipIfTruncated(el, fullText) {
    if (!el) return;
    el.textContent = fullText || '';
    el.removeAttribute('title');
    el.removeAttribute('data-tip');
    el.classList.remove('has-overflow-tip');
    requestAnimationFrame(() => {
        if (el.scrollWidth > el.clientWidth + 1) {
            el.setAttribute('title', fullText || '');
            el.setAttribute('data-tip', fullText || '');
            el.classList.add('has-overflow-tip');
        }
    });
}

/* Singleton floating tooltip. Lives at <body> level so it escapes any
   overflow-clipping ancestor (which is exactly the issue that breaks a
   pseudo-element approach for a truncated element). One element is
   reused; we just reposition + retext it on each hover. */
(function installOverflowTooltip() {
    if (document.getElementById('pgOverflowTip')) return;
    const tip = document.createElement('div');
    tip.id = 'pgOverflowTip';
    document.body.appendChild(tip);

    function show(target) {
        const text = target.getAttribute('data-tip') || target.textContent || '';
        tip.textContent = text;
        // Make it measurable before positioning.
        tip.classList.add('visible');
        const r = target.getBoundingClientRect();
        const tr = tip.getBoundingClientRect();
        // Default: 8px above the target, left-aligned with it.
        let top = r.top - tr.height - 8;
        let left = r.left;
        // Flip below the target if there's no room above.
        if (top < 8) top = r.bottom + 8;
        // Clamp horizontally so the tip can't shoot off-screen.
        const margin = 8;
        if (left + tr.width > window.innerWidth - margin) {
            left = window.innerWidth - tr.width - margin;
        }
        if (left < margin) left = margin;
        tip.style.top = top + 'px';
        tip.style.left = left + 'px';
    }
    function hide() { tip.classList.remove('visible'); }

    document.addEventListener('mouseover', (ev) => {
        const target = ev.target.closest('.has-overflow-tip');
        if (target) show(target);
    });
    document.addEventListener('mouseout', (ev) => {
        const target = ev.target.closest('.has-overflow-tip');
        if (target) hide();
    });
    // Hide on scroll / resize since the target moved.
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide);
})();

function setStatus(msg, type) {
    const dot = document.getElementById('statusDot');
    const msgEl = document.getElementById('statusMsg');
    dot.className = 'status-dot';
    if (type === 'scanning') dot.classList.add('scanning');
    else if (type === 'error') dot.classList.add('error');
    msgEl.textContent = msg;
}

function randomTip() {
    const el = document.getElementById('securityTip');
    if (!el) return;
    const span = el.querySelector('span');
    if (span) {
        span.textContent = SECURITY_TIPS[Math.floor(Math.random() * SECURITY_TIPS.length)];
    }
}

/* ======================== API CALLS ======================== */

async function loadEmails() {
    setStatus('Loading emails...', 'scanning');

    // Show skeleton placeholders while fetching — mirror the real
    // table columns (star | subject | date | sender | score ring) so
    // the layout doesn't jump when real rows arrive.
    const skeletonHtml = Array(7).fill(
        '<div class="email-row skeleton-row">' +
        '<span class="etr-star"><span class="skeleton sk-dot"></span></span>' +
        '<span class="etr-subject"><span class="skeleton sk-bar"></span></span>' +
        '<span class="etr-date"><span class="skeleton sk-bar sk-bar-sm"></span></span>' +
        '<span class="etr-sender"><span class="skeleton sk-bar"></span></span>' +
        '<span class="etr-score"><span class="skeleton sk-ring"></span></span>' +
        '</div>'
    ).join('');
    document.getElementById('emailList').innerHTML = skeletonHtml;

    try {
        const res = await fetch('/api/emails');
        if (!res.ok) throw new Error('Failed to load emails');
        const data = await res.json();
        state.emails = data.emails || [];
        // DO NOT clear scanResults here — they're keyed by message_id
        // and stay valid across folder switches (inbox → junk → inbox)
        // and refreshes. Only explicit sign-out clears them.
        state.selectedIdx = null;
        updateStatsFromResults();
        listAnimateNext = true;
        renderEmailList();
        document.getElementById('emailPreview').style.display = 'none';
        document.getElementById('dashboardView').style.display = '';
        setStatus('Ready — ' + state.emails.length + ' emails loaded', '');
    } catch (err) {
        console.error('loadEmails error:', err);
        renderEmailList();
        setStatus('Error loading emails', 'error');
    }
}

async function scanEmail(idx) {
    const messageId = emailIdFromIdx(idx);
    if (!messageId) {
        setStatus('Could not resolve message — reload the list', 'error');
        return;
    }
    setStatus('Analyzing message...', 'scanning');

    const overlay = document.getElementById('scanOverlay');
    const scanLine = document.getElementById('scanLine');
    const scanGlow = document.getElementById('scanGlow');
    overlay.className = 'scan-overlay';
    scanLine.className = 'scan-line scanning';
    scanGlow.className = 'scan-glow scanning';

    try {
        const res = await apiFetch('/api/messages/' + encodeURIComponent(messageId) + '/scan',
                                    { method: 'POST' });
        const data = await res.json();

        scanLine.className = 'scan-line';
        scanGlow.className = 'scan-glow';

        if (data.error) {
            console.error('Scan error:', data.error);
            setStatus('Scan failed: ' + data.error, 'error');
            return;
        }
        if (!data.scanned_at) data.scanned_at = new Date().toISOString();
        state.scanResults[messageId] = data;

        const isPhishing = data.prediction === 1;
        overlay.className = 'scan-overlay ' + (isPhishing ? 'scan-complete-danger' : 'scan-complete-safe');
        setTimeout(() => { overlay.className = 'scan-overlay'; }, 600);

        updateStatsFromResults();
        renderEmailList();
        if (state.selectedIdx === idx) {
            showScanResults(idx);
        }
        setStatus('Analysis complete', '');

        // Kick off attachment analysis in the background (non-blocking).
        // Doesn't await — the scan UX shouldn't wait on VT's rate limit.
        const email = state.emails[idx];
        if (email && email.attachments && email.attachments.length > 0) {
            analyzeEmailAttachments(email);
        }
    } catch (err) {
        scanLine.className = 'scan-line';
        scanGlow.className = 'scan-glow';
        console.error('scanEmail error:', err);
        setStatus('Scan failed', 'error');
    }
}

async function scanAll() {
    const total = state.emails.length;
    if (total === 0) return;

    const progressEl = document.getElementById('scanProgress');
    const barEl = document.getElementById('scanProgressBar');
    progressEl.style.display = 'block';
    barEl.style.width = '0%';
    setStatus('Scanning all emails...', 'scanning');

    let scanned = 0;
    for (let i = 0; i < total; i++) {
        const messageId = emailIdFromIdx(i);
        if (!messageId) { scanned++; continue; }
        if (state.scanResults[messageId]) {
            scanned++;
            barEl.style.width = Math.round((scanned / total) * 100) + '%';
            continue;
        }
        try {
            const res = await apiFetch('/api/messages/' + encodeURIComponent(messageId) + '/scan',
                                        { method: 'POST' });
            const data = await res.json();
            if (!data.error) {
                if (!data.scanned_at) data.scanned_at = new Date().toISOString();
                state.scanResults[messageId] = data;
            }
        } catch (e) { /* skip failed */ }
        scanned++;
        barEl.style.width = Math.round((scanned / total) * 100) + '%';
        setStatus('Scanning ' + scanned + ' of ' + total + '...', 'scanning');
    }

    updateStatsFromResults();
    renderEmailList();
    if (state.selectedIdx !== null && scanResultFor(state.selectedIdx)) {
        showScanResults(state.selectedIdx);
    }
    setStatus('Batch analysis complete', '');

    setTimeout(() => {
        barEl.style.width = '100%';
        setTimeout(() => {
            progressEl.style.display = 'none';
            barEl.style.width = '0%';
        }, 500);
    }, 300);
}

async function getReputation(domain, idx) {
    try {
        const res = await fetch('/api/reputation/' + encodeURIComponent(domain));
        if (!res.ok) return null;
        const data = await res.json();
        return data;
    } catch (err) {
        console.error('getReputation error:', err);
        return null;
    }
}

/* ======================== STATS ======================== */

function updateStatsFromResults() {
    let scanned = 0, threats = 0, safe = 0;
    // Sidebar Scan Summary tiers (by risk score): Dangerous / Suspicious / Safe.
    // Count over the emails actually IN the current list (not every stored scan
    // result) so the counts match the Threats/Suspicious/Safe views — otherwise
    // restored or junk-folder results inflate the numbers.
    let dangerous = 0, suspicious = 0, safeTier = 0;
    for (let i = 0; i < state.emails.length; i++) {
        const r = scanResultFor(i);
        if (!r) continue;
        scanned++;
        if (r.prediction === 1) threats++; else safe++;
        const rs = riskScoreOf(r);
        if (rs >= 65) dangerous++;
        else if (rs >= 40) suspicious++;
        else safeTier++;
    }
    state.stats = { scanned, threats, safe, dangerous, suspicious, safeTier };
    renderStats();
}

function renderStats() {
    const setTxt = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    setTxt('statScannedNum', state.stats.scanned);
    setTxt('statThreatsNum', state.stats.threats);
    setTxt('statSafeNum', state.stats.safe);

    setTxt('dashScanned', state.stats.scanned);
    setTxt('dashThreats', state.stats.threats);
    setTxt('dashSafe', state.stats.safe);

    // Sidebar Scan Summary
    setTxt('sumDangerous', state.stats.dangerous || 0);
    setTxt('sumSuspicious', state.stats.suspicious || 0);
    setTxt('sumSafe', state.stats.safeTier || 0);

    updateDashboardInsights();
    updateSecurityScore();
}

function updateDashboardInsights() {
    const wrap = document.getElementById('dashInsights');
    if (!wrap) return;
    const scanned = state.stats.scanned || 0;
    if (!scanned) {
        wrap.style.display = 'none';
        return;
    }
    wrap.style.display = '';

    const dangerous = state.stats.dangerous || 0;
    const suspicious = state.stats.suspicious || 0;
    const safe = state.stats.safeTier || 0;
    const pct = (n) => Math.max(0, Math.min(100, Math.round((n / scanned) * 100))) + '%';
    const setWidth = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.style.width = value;
    };
    const setTxt = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };
    setWidth('dashBarDanger', pct(dangerous));
    setWidth('dashBarWarn', pct(suspicious));
    setWidth('dashBarSafe', pct(safe));
    setTxt('dashLegDanger', dangerous);
    setTxt('dashLegWarn', suspicious);
    setTxt('dashLegSafe', safe);

    const threatCounts = {};
    let lastScan = null;
    for (let i = 0; i < state.emails.length; i++) {
        const result = scanResultFor(i);
        if (!result) continue;
        const tsRaw = result.scanned_at || result.scannedAt || result.timestamp;
        const ts = tsRaw ? new Date(tsRaw) : null;
        if (ts && !isNaN(ts.getTime()) && (!lastScan || ts > lastScan)) lastScan = ts;
        const threats = (result.assessment && result.assessment.threats) || [];
        for (const t of threats) {
            const title = (t && t.title) ? String(t.title) : '';
            if (!title) continue;
            threatCounts[title] = (threatCounts[title] || 0) + 1;
        }
    }

    const list = document.getElementById('dashThreatsList');
    if (list) {
        const top = Object.entries(threatCounts)
            .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
            .slice(0, 3);
        list.innerHTML = top.length
            ? top.map(([name, count]) => '<div class="dash-threat-row"><span>' + escapeHtml(name) + '</span><b>' + count + '</b></div>').join('')
            : '<div class="dash-empty-note">No recurring threat patterns yet.</div>';
    }

    const last = document.getElementById('dashLastScan');
    if (last) {
        last.textContent = lastScan
            ? 'Last scan ' + lastScan.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
            : 'Last scan just now';
    }
}

/* ======================== #10 INBOX SECURITY SCORE ======================== */

function updateSecurityScore() {
    const total = state.emails.length;
    const scanned = state.stats.scanned;
    const threats = state.stats.threats;
    const wrap = document.getElementById('securityScoreWrap');
    if (!wrap) return;

    /* Only show once every visible email has a scan result. */
    if (total === 0 || scanned < total) {
        wrap.style.display = 'none';
        return;
    }

    wrap.style.display = '';

    const threatRatio = scanned > 0 ? threats / scanned : 0;
    let score = Math.round((1 - threatRatio) * 100);
    score = Math.max(0, Math.min(100, score));

    const ring = document.getElementById('secScoreRingFill');
    const valEl = document.getElementById('secScoreVal');
    const labelEl = document.getElementById('secScoreLabel');

    const circ = 2 * Math.PI * 42;
    const filled = (score / 100) * circ;
    ring.setAttribute('stroke-dasharray', filled.toFixed(1) + ', ' + circ.toFixed(1));

    let color = 'var(--success)';
    let label = 'Inbox Protected';
    if (score < 50) { color = 'var(--danger)'; label = 'Inbox At Risk'; }
    else if (score < 75) { color = 'var(--warning)'; label = 'Review Recommended'; }

    ring.setAttribute('stroke', color);
    valEl.textContent = score;
    valEl.style.color = color;
    labelEl.textContent = label;
}

/* ======================== #8 STAR/PIN EMAILS ======================== */

async function toggleStar(idx) {
    const email = state.emails[idx];
    if (!email) return;
    const emailId = email.id || String(idx);

    if (!state.user) {
        setStatus('Sign in to star emails', 'error');
        return;
    }

    const wasStarred = !!state.starredEmails[emailId];

    // Optimistic local update — instant feedback, sync to DB after.
    if (wasStarred) {
        delete state.starredEmails[emailId];
    } else {
        state.starredEmails[emailId] = true;
    }
    renderEmailList();

    try {
        if (wasStarred) {
            await window.pg.starred.remove(emailId);
        } else {
            await window.pg.starred.add(emailId);
        }
    } catch (e) {
        // Revert local state on failure so the UI matches the DB.
        if (wasStarred) {
            state.starredEmails[emailId] = true;
        } else {
            delete state.starredEmails[emailId];
        }
        renderEmailList();
        setStatus('Could not sync star — see console', 'error');
        console.warn('[PhishGuard] toggleStar failed:', e);
    }
}

function isStarred(idx) {
    const emailId = state.emails[idx] ? (state.emails[idx].id || idx) : idx;
    return !!state.starredEmails[emailId];
}

/* ======================== #13 CONTACT CARDS ======================== */

function showContactCard(email, anchorEl) {
    closeContactCard();
    const senderName = email.sender_name || email.sender || 'Unknown';
    const senderAddr = email.sender || '';
    const domain = extractDomain(senderAddr);
    const initials = getInitials(senderName);
    const avatarColor = getAvatarColor(senderAddr);

    /* Count emails from this sender */
    let totalFromSender = 0, threatsFromSender = 0;
    state.emails.forEach((e, i) => {
        if ((e.sender || '') === senderAddr) {
            totalFromSender++;
            const r = scanResultFor(i);
            if (r && r.prediction === 1) threatsFromSender++;
        }
    });
    const safeFromSender = totalFromSender - threatsFromSender;

    /* Domain trust */
    const domainTrusted = ['company.com','amazon.com','chase.com','atlassian.net'].some(d => domain.endsWith(d));
    const dotColor = domainTrusted ? 'var(--success)' : 'var(--warning)';
    const domainLabel = domainTrusted ? 'Trusted domain' : 'Unverified domain';

    const rect = anchorEl.getBoundingClientRect();
    const top = Math.min(rect.bottom + 8, window.innerHeight - 300);
    const left = Math.min(rect.left, window.innerWidth - 340);

    const container = document.getElementById('contactCardContainer');
    container.innerHTML = '<div class="contact-card-overlay" id="contactCardOverlay"></div>' +
        '<div class="contact-card" style="top:' + top + 'px;left:' + left + 'px;">' +
        '<div class="contact-card-header">' +
        '<div class="contact-card-avatar" style="background:' + avatarColor + '">' + escapeHtml(initials) + '</div>' +
        '<div><div class="contact-card-name">' + escapeHtml(senderName) + '</div>' +
        '<div class="contact-card-email">' + escapeHtml(senderAddr) + '</div></div></div>' +
        '<div class="contact-card-stats">' +
        '<div class="contact-stat"><div class="contact-stat-num">' + totalFromSender + '</div><div class="contact-stat-label">Emails</div></div>' +
        '<div class="contact-stat"><div class="contact-stat-num" style="color:var(--success)">' + safeFromSender + '</div><div class="contact-stat-label">Safe</div></div>' +
        '<div class="contact-stat"><div class="contact-stat-num" style="color:var(--danger)">' + threatsFromSender + '</div><div class="contact-stat-label">Threats</div></div>' +
        '</div>' +
        '<div class="contact-card-domain"><div class="domain-dot" style="background:' + dotColor + '"></div>' + escapeHtml(domainLabel) + ' — ' + escapeHtml(domain) + '</div>' +
        '</div>';

    document.getElementById('contactCardOverlay').addEventListener('click', closeContactCard);
}

function closeContactCard() {
    document.getElementById('contactCardContainer').innerHTML = '';
}

/* ======================== #5 SPLASH SCREEN ======================== */

function dismissSplash() {
    const splash = document.getElementById('splashScreen');
    if (!splash) return;
    splash.classList.add('fade-out');
    setTimeout(() => { splash.remove(); }, 600);
}

/* ======================== #3 EMAIL PRIORITY HEURISTIC ======================== */

function isPriorityWarn(email) {
    /* Quick pre-scan heuristic — flags suspicious emails before AI scan */
    if (!email) return false;
    const headers = email.headers || {};
    const spf = (headers.spf || '').toLowerCase();
    const dkim = (headers.dkim || '').toLowerCase();
    const dmarc = (headers.dmarc || '').toLowerCase();
    const authFailed = spf === 'fail' || dkim === 'fail' || dmarc === 'fail';
    const subject = (email.subject || '').toLowerCase();
    const sender = (email.sender || '').toLowerCase();
    const hasUrgency = URGENCY_WORDS.some(w => subject.includes(w));
    const configuredTlds = state.detectionRules.risky_tlds || [];
    const suspiciousTlds = configuredTlds.length
        ? configuredTlds.map(t => t.charAt(0) === '.' ? t : '.' + t)
        : ['.xyz', '.top', '.click', '.buzz', '.net'];
    const hasSuspiciousTld = suspiciousTlds.some(t => sender.endsWith(t));
    const domain = extractDomain(sender);
    const brandTokens = state.detectionRules.brand_name_tokens || {};
    const fallbackBrands = {
        paypal: 'paypal.com', microsoft: 'microsoft.com', apple: 'apple.com',
        amazon: 'amazon.com', chase: 'chase.com', google: 'google.com',
    };
    const brands = Object.keys(brandTokens).length ? brandTokens : fallbackBrands;
    const hasMismatch = Object.entries(brands).some(([token, expected]) =>
        sender.includes(token) && !domainMatches(domain, [expected]));

    return (authFailed && (hasUrgency || hasSuspiciousTld)) || hasMismatch;
}

/* ======================== #13 SEARCH AUTOCOMPLETE ======================== */

function showSearchAutocomplete(query) {
    const ac = document.getElementById('searchAutocomplete');
    if (!query) { ac.style.display = 'none'; return; }

    let html = '';
    const q = query.toLowerCase();

    /* Filter hints */
    const filters = [
        { label: 'from:', desc: 'Search by sender' },
        { label: 'is:threat', desc: 'Show phishing emails' },
        { label: 'is:safe', desc: 'Show safe emails' },
        { label: 'is:unread', desc: 'Show unread emails' },
        { label: 'is:scanned', desc: 'Show scanned emails' },
        { label: 'has:attachment', desc: 'Has attachments' },
    ];
    const matchingFilters = filters.filter(f => f.label.includes(q) && f.label !== q);
    if (matchingFilters.length > 0) {
        html += '<div class="search-ac-section"><div class="search-ac-label">Filters</div>';
        for (const f of matchingFilters.slice(0, 4)) {
            html += '<div class="search-ac-item" data-ac-value="' + f.label + '"><kbd>' + f.label + '</kbd> <span>' + f.desc + '</span></div>';
        }
        html += '</div>';
    }

    /* Matching senders */
    const seenSenders = new Set();
    const matchingSenders = [];
    for (const email of state.emails) {
        const name = email.sender_name || '';
        const addr = email.sender || '';
        const key = addr.toLowerCase();
        if (seenSenders.has(key)) continue;
        if (name.toLowerCase().includes(q) || addr.toLowerCase().includes(q)) {
            seenSenders.add(key);
            matchingSenders.push({ name, addr });
        }
    }
    if (matchingSenders.length > 0) {
        html += '<div class="search-ac-section"><div class="search-ac-label">Senders</div>';
        for (const s of matchingSenders.slice(0, 5)) {
            const color = getAvatarColor(s.addr);
            html += '<div class="search-ac-item" data-ac-value="from:' + escapeHtml(s.addr) + '">' +
                '<span class="ac-sender-dot" style="background:' + color + '"></span>' +
                '<span>' + escapeHtml(s.name) + '</span>' +
                '<span style="color:var(--text3);font-size:11px;">' + escapeHtml(s.addr) + '</span></div>';
        }
        html += '</div>';
    }

    if (!html) { ac.style.display = 'none'; return; }

    ac.innerHTML = html;
    ac.style.display = 'block';

    ac.querySelectorAll('.search-ac-item').forEach(item => {
        item.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const val = item.getAttribute('data-ac-value');
            document.getElementById('searchInput').value = val;
            ac.style.display = 'none';
            renderEmailList();
        });
    });
}

/* ======================== EMAIL LIST RENDERING ======================== */

/* When true, the next renderEmailList() staggers the rows in with a
   short cascade. Set by loadEmails/loadJunkEmails/setNav (fresh list
   contexts) and cleared after one render so re-renders triggered by
   selection or starring don't replay the animation. */
let listAnimateNext = false;

function renderEmailList() {
    const container = document.getElementById('emailList');
    const searchVal = document.getElementById('searchInput').value.toLowerCase().trim();

    let filtered = state.emails.map((email, idx) => ({ email, idx }));

    if (searchVal) {
        const filters = [];
        let freeText = '';

        // Parse filter tokens
        const tokens = searchVal.split(/\s+/);
        for (const token of tokens) {
            const colonIdx = token.indexOf(':');
            if (colonIdx > 0) {
                const key = token.substring(0, colonIdx).toLowerCase();
                const val = token.substring(colonIdx + 1).toLowerCase();
                filters.push({ key, val });
            } else {
                freeText += (freeText ? ' ' : '') + token;
            }
        }

        filtered = filtered.filter(({ email, idx }) => {
            // Apply filters
            for (const f of filters) {
                if (f.key === 'from') {
                    const sender = (email.sender_name || '').toLowerCase() + ' ' + (email.sender || '').toLowerCase();
                    if (!sender.includes(f.val)) return false;
                } else if (f.key === 'has' && f.val === 'attachment') {
                    if (!email.hasAttachments) return false;
                } else if (f.key === 'is') {
                    if (f.val === 'unread' && email.isRead !== false) return false;
                    if (f.val === 'read' && email.isRead === false) return false;
                    const __sr = scanResultFor(idx);
                    if (f.val === 'scanned' && !__sr) return false;
                    if ((f.val === 'threat' || f.val === 'phishing') && (!__sr || __sr.prediction !== 1)) return false;
                    if (f.val === 'safe' && (!__sr || __sr.prediction !== 0)) return false;
                }
            }
            // Apply free text search
            if (freeText) {
                const subject = (email.subject || '').toLowerCase();
                const sender = (email.sender_name || email.sender || '').toLowerCase();
                if (!subject.includes(freeText) && !sender.includes(freeText)) return false;
            }
            return true;
        });
    }

    // Threats / Suspicious / Safe match the colored labels the user sees:
    // red (>=65) / yellow (40-64) / green (<40), by the same risk score.
    if (state.navView === 'threats') {
        filtered = filtered.filter(({ idx }) => {
            const r = scanResultFor(idx);
            return r && riskScoreOf(r) >= 65;
        });
    } else if (state.navView === 'suspicious') {
        filtered = filtered.filter(({ idx }) => {
            const r = scanResultFor(idx);
            if (!r) return false;
            const v = riskScoreOf(r);
            return v >= 40 && v < 65;
        });
    } else if (state.navView === 'safe') {
        filtered = filtered.filter(({ idx }) => {
            const r = scanResultFor(idx);
            return r && riskScoreOf(r) < 40;
        });
    } else if (state.navView === 'starred') {
        filtered = filtered.filter(({ idx }) => isStarred(idx));
    }

    const _navTitles = { inbox: 'Inbox', threats: 'Threats', suspicious: 'Suspicious', safe: 'Safe', starred: 'Starred', junk: 'Junk' };
    const _lt = document.getElementById('listTitle');
    if (_lt) _lt.textContent = _navTitles[state.navView] || 'Inbox';
    const _lc = document.getElementById('listCount');
    if (_lc) _lc.textContent = filtered.length;

    if (filtered.length === 0) {
        container.innerHTML = '<div style="padding:30px 16px;text-align:center;color:var(--text3);font-size:12px;">' +
            (searchVal ? 'No emails match your search.' :
             state.navView === 'threats' ? 'No threats detected.' :
             state.navView === 'suspicious' ? 'No suspicious emails.' :
             state.navView === 'safe' ? 'No safe emails yet.' :
             state.navView === 'junk' ? 'Junk folder is empty.' :
             'No emails loaded.') +
            '</div>';
        return;
    }

    let html = '';
    let renderPos = 0;
    for (const { email, idx } of filtered) {
        const senderName = email.sender_name || email.sender || 'Unknown';
        const subject = email.subject || '(no subject)';
        const date = formatDate(email.date);
        const isUnread = email.isRead === false;
        const isSelected = state.selectedIdx === idx;
        const scanResult = scanResultFor(idx);

        const rsv = scanResult ? riskScoreOf(scanResult) : null;
        const tier = rsv === null ? 'none' : (rsv >= 65 ? 'danger' : (rsv >= 40 ? 'warn' : 'safe'));
        const STAT = {
            danger: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
            warn: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
            safe: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
            none: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg>'
        };
        const statusIcon = '<span class="er-status er-' + tier + '">' + STAT[tier] + '</span>';
        const badge = rsv === null ? '' : '<span class="er-score er-' + tier + '">' + rsv + '</span>';

        const starred = isStarred(idx);
        const starSvg = starred
            ? '<svg width="13" height="13" viewBox="0 0 24 24" fill="var(--warning)" stroke="var(--warning)" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'
            : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';

        const priorityWarn = scanResult && scanResult.prediction === 1;
        const preview = (email.bodyPreview || email.body || '').replace(/\s+/g, ' ').trim();

        const enterAttrs = listAnimateNext
            ? ' row-enter" style="animation-delay:' + Math.min(renderPos * 24, 360) + 'ms'
            : '';
        renderPos++;

        html += '<div class="email-row' +
            (isUnread ? ' unread' : '') +
            (isSelected ? ' selected' : '') +
            (priorityWarn ? ' priority-warn' : '') +
            ' tier-' + tier +
            enterAttrs +
            '" data-idx="' + idx + '">' +
            statusIcon +
            '<div class="er-main">' +
                '<div class="er-top">' +
                    '<span class="er-sender etr-sender-link" data-sender-idx="' + idx + '">' + escapeHtml(truncate(senderName, 24)) + '</span>' +
                    '<span class="er-meta">' + badge + '<span class="er-date">' + escapeHtml(date) + '</span></span>' +
                '</div>' +
                '<div class="er-subject">' + escapeHtml(truncate(subject, 60)) + '</div>' +
                '<div class="er-preview">' + escapeHtml(truncate(preview, 76)) + '</div>' +
            '</div>' +
            '<div class="er-actions">' +
                '<button class="star-btn' + (starred ? ' starred' : '') + '" data-star-idx="' + idx + '" title="Star">' + starSvg + '</button>' +
                '<button class="etr-action-btn" data-quick="scan" data-idx="' + idx + '" title="Scan"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/></svg></button>' +
                '<button class="etr-action-btn" data-quick="junk" data-idx="' + idx + '" title="Move to junk"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg></button>' +
            '</div>' +
            '</div>';
    }

    container.innerHTML = html;
    listAnimateNext = false;

    container.querySelectorAll('.email-row').forEach(row => {
        row.addEventListener('click', (e) => {
            /* Don't select email if clicking star or quick action */
            if (e.target.closest('.star-btn') || e.target.closest('.etr-action-btn') || e.target.closest('.etr-sender-link') || e.target.closest('.etr-action-left') || e.target.closest('.etr-action-right')) return;
            const idx = parseInt(row.getAttribute('data-idx'), 10);
            selectEmail(idx);
        });
    });

    /* #8 Star buttons */
    container.querySelectorAll('.star-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.getAttribute('data-star-idx'), 10);
            toggleStar(idx);
        });
    });

    /* #7 Quick action buttons */
    container.querySelectorAll('.etr-action-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.getAttribute('data-idx'), 10);
            const action = btn.getAttribute('data-quick');
            if (action === 'scan') scanEmail(idx);
            else if (action === 'junk') moveToJunk(idx);
        });
    });

    /* #13 Sender name → contact card */
    container.querySelectorAll('.etr-sender-link').forEach(el => {
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(el.getAttribute('data-sender-idx'), 10);
            const email = state.emails[idx];
            if (email) showContactCard(email, el);
        });
    });
}

/* ======================== EMAIL SELECTION & PREVIEW ======================== */

function selectEmail(idx) {
    state.selectedIdx = idx;
    state.inlineVisible = false;

    renderEmailList();
    showEmail(idx);
}

function showEmail(idx) {
    const email = state.emails[idx];
    if (!email) return;

    document.getElementById('dashboardView').style.display = 'none';
    document.getElementById('dashboardView').classList.remove('dash-enter');
    const previewEl = document.getElementById('emailPreview');
    previewEl.style.display = 'flex';
    previewEl.classList.remove('preview-enter');
    void previewEl.offsetWidth; /* force reflow to restart animation */
    previewEl.classList.add('preview-enter');

    document.getElementById('previewSubject').textContent = email.subject || '(no subject)';

    const senderName = email.sender_name || email.sender || 'Unknown';
    const senderAddr = email.sender || '';
    const initials = getInitials(senderName);
    const avatarColor = getAvatarColor(senderAddr);

    const avatarEl = document.getElementById('previewAvatar');
    avatarEl.textContent = initials;
    avatarEl.style.background = avatarColor;

    setTextWithTooltipIfTruncated(document.getElementById('previewSender'), senderName);
    setTextWithTooltipIfTruncated(document.getElementById('previewEmail'), senderAddr);
    document.getElementById('previewDate').textContent = formatDate(email.date);

    // Hide elements on email switch
    document.getElementById('replyWarning').style.display = 'none';
    document.getElementById('mismatchAlert').style.display = 'none';
    document.getElementById('scanBrief').style.display = 'none';

    renderAttachments(email);
    // Attachment metadata isn't in the email-list fetch — pull it lazily the
    // first time an email with attachments is opened, then re-render.
    if (email.hasAttachments && !email._attLoaded) {
        email._attLoaded = true;
        fetch('/api/messages/' + encodeURIComponent(email.id) + '/attachments',
              { credentials: 'same-origin' })
            .then(r => r.json())
            .then(d => {
                email.attachments = d.attachments || [];
                renderAttachments(email);
                // refresh the rail (attachment count / Files bar) if scanned
                if (state.selectedIdx === idx && scanResultFor(idx)) showScanResults(idx);
            })
            .catch(() => {});
    }

    const bodyEl = document.getElementById('previewBody');
    const bodyHtmlIframe = document.getElementById('previewBodyHtml');
    const bodyToggle = document.getElementById('bodyToggle');
    const bodyText = email.body || '';
    const bodyHtml = email.bodyHtml || '';

    /* Plain text view — cap body size before running the URL regex. */
    const cappedBody = bodyText.length > MAX_BODY_FOR_REGEX
        ? bodyText.slice(0, MAX_BODY_FOR_REGEX)
        : bodyText;
    bodyEl.innerHTML = escapeHtml(cappedBody).replace(/\n/g, '<br>');
    bodyEl.innerHTML = bodyEl.innerHTML.replace(URL_REGEX_SAFE, (url) => {
        /* Real URL stays in data-url; the visible text is defanged so users
           can read where the link points without being able to copy a working
           URL out of the preview. The hover popup has an explicit reveal. */
        const realSafe = escapeHtml(url);
        const defangedSafe = escapeHtml(defangUrl(url));
        return '<span class="body-link" data-url="' + realSafe + '">' + defangedSafe + '</span>';
    });

    /* Rich HTML toggle — only show if HTML body exists */
    if (bodyHtml) {
        bodyToggle.style.display = 'flex';
        /* Reset to plain text view */
        document.getElementById('bodyTogglePlain').classList.add('active');
        document.getElementById('bodyToggleHtml').classList.remove('active');
        bodyEl.style.display = '';
        bodyHtmlIframe.style.display = 'none';
        /* Store HTML for when user toggles */
        bodyHtmlIframe.setAttribute('data-html', bodyHtml);
    } else {
        bodyToggle.style.display = 'none';
        bodyEl.style.display = '';
        bodyHtmlIframe.style.display = 'none';
    }

    const scanBtn = document.getElementById('scanBtn');
    state.inlineVisible = false;
    state.detailReportOpen = false;

    if (scanResultFor(idx)) {
        showScanResults(idx);   // fills report + a "Rescan" button (via showScanBrief)
    } else {
        scanBtn.style.display = '';
        scanBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/></svg> Scan for Phishing';
        clearAnalysisRail();
    }
}

function renderAttachments(email) {
    const container = document.getElementById('previewAttachments');
    const list = document.getElementById('attachmentsList');
    const attachments = email.attachments || [];

    if (attachments.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    const vtMap = email._vt_results || {};
    let html = '';
    for (const att of attachments) {
        const name = typeof att === 'string' ? att : (att.name || att.filename || 'file');
        const risk = (att && typeof att === 'object' && att.risk)
            ? att.risk
            : (isRiskyExtension(name) ? 'dangerous' : 'safe');
        const risky = risk !== 'safe';
        const reason = (att && typeof att === 'object' && att.reason) ? att.reason : '';
        const tierCls = risk === 'dangerous' ? ' risky' : (risk === 'caution' ? ' caution' : '');
        const vt = vtMap[name];
        let vtBadge = '';
        if (vt) {
            if (vt.status === 'scanning') {
                vtBadge = '<span class="vt-status vt-scanning" title="Analyzing on VirusTotal">scanning…</span>';
            } else if (vt.status === 'deepscanning') {
                vtBadge = '<span class="vt-status vt-scanning" title="Uploading & analyzing on VirusTotal">'
                    + (vt.phase === 'analyzing' ? 'analyzing…' : 'uploading…') + '</span>';
            } else if (vt.status === 'pending_timeout') {
                vtBadge = '<span class="vt-status vt-unknown" title="' + escapeHtml(vt.message || '') + '">'
                    + 'analyzing…</span>';
            } else if (vt.analyzed && typeof vt.malicious === 'number' && vt.malicious > 0) {
                vtBadge = '<span class="vt-status vt-malicious" title="Click for details">'
                    + vt.malicious + '/' + (vt.total || 0) + ' flagged</span>';
            } else if (vt.analyzed && vt.found) {
                vtBadge = '<span class="vt-status vt-clean" title="' + (vt.total || 0) + ' engines, none flagged">'
                    + 'clean</span>';
            } else if (vt.analyzed && !vt.found) {
                vtBadge = '<span class="vt-status vt-unknown" title="Hash not in VirusTotal database">'
                    + 'unknown</span>'
                    + '<button class="vt-deepscan" data-att="' + escapeHtml(name)
                    + '" title="Upload this file to VirusTotal for a fresh scan. Sends the file&#39;s contents to VirusTotal.">Deep scan</button>';
            } else if (vt.configured === false || vt.enabled === false) {
                vtBadge = '<span class="vt-status vt-disabled" title="' + escapeHtml(vt.message || '') + '">'
                    + 'VT off</span>';
            } else if (vt.error) {
                vtBadge = '<span class="vt-status vt-error" title="' + escapeHtml(vt.error) + '">'
                    + 'scan error</span>';
            }
        }
        html += '<span class="attachment-badge' + tierCls + '"' +
            (reason ? ' title="' + escapeHtml(reason) + '"' : '') + '>' +
            '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>' +
            (risky ? '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>' : '') +
            escapeHtml(name) +
            vtBadge +
            '</span>';
    }
    list.innerHTML = html;
    // Wire the per-file "Deep scan" buttons (shown on VT "unknown" results).
    list.querySelectorAll('.vt-deepscan').forEach(btn => {
        btn.addEventListener('click', () => deepScanAttachment(email, btn.dataset.att));
    });
}

/* User-initiated DEEP scan of one attachment: upload it to VirusTotal for a
   fresh analysis (only reached from the "unknown" badge). This sends the file
   CONTENTS to VT, so it never runs automatically. Polls until the analysis
   completes, then renders the verdict like any other VT result. */
async function deepScanAttachment(email, name) {
    const messageId = email.id || email.messageId;
    if (!messageId) return;
    const confirmed = window.confirm(
        'Upload “' + name + '” to VirusTotal?\n\n'
        + 'The complete attachment will leave this device and be handled under '
        + 'VirusTotal’s privacy and retention policies.'
    );
    if (!confirmed) return;
    const att = (email.attachments || []).find(
        a => (typeof a === 'string' ? a : (a.name || a.filename)) === name);
    const attId = (att && typeof att === 'object' && (att.id || att.name)) || name;
    email._vt_results = email._vt_results || {};
    email._vt_results[name] = { status: 'deepscanning', phase: 'uploading' };
    renderAttachments(email);

    const base = '/api/messages/' + encodeURIComponent(messageId)
        + '/attachments/' + encodeURIComponent(attId) + '/deepscan';
    let data;
    try {
        const res = await apiFetch(base, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirm_upload: true }),
        });
        data = await res.json();
    } catch (e) {
        email._vt_results[name] = { error: 'upload failed' };
        renderAttachments(email);
        return;
    }
    // Hash turned out to be known, or the upload failed / not possible.
    if (data.error || data.configured === false || data.analyzed === false
        || (data.uploaded === false && data.analyzed)) {
        email._vt_results[name] = data;
        renderAttachments(email);
        return;
    }
    // Uploaded → poll the analysis until VT finishes (spaced to respect the
    // 4-req/min free tier).
    const analysisId = data.analysis_id;
    email._vt_results[name] = { status: 'deepscanning', phase: 'analyzing' };
    renderAttachments(email);
    for (let i = 0; i < 6; i++) {
        await new Promise(r => setTimeout(r, 18000));
        try {
            const res = await apiFetch(base + '/' + encodeURIComponent(analysisId),
                { method: 'GET' });
            const pd = await res.json();
            if (pd.analyzed || pd.error) {
                email._vt_results[name] = pd;
                renderAttachments(email);
                return;
            }
        } catch (e) { /* transient — keep polling */ }
    }
    email._vt_results[name] = { status: 'pending_timeout',
        message: 'Still analyzing on VirusTotal — reopen the email shortly to see the result.' };
    renderAttachments(email);
}

/* Analyze every attachment on this email against VirusTotal.
   Sequential (not parallel) so we don't blow through the 4-req/min
   free-tier rate limit. Results are cached on the email object so
   they survive re-renders. */
async function analyzeEmailAttachments(email) {
    if (!email || !email.attachments || email.attachments.length === 0) return;
    const messageId = email.id || email.messageId;
    if (!messageId) return;

    email._vt_results = email._vt_results || {};

    // Mark all as scanning so the UI shows progress immediately.
    for (const att of email.attachments) {
        const name = typeof att === 'string' ? att : (att.name || att.filename || 'file');
        if (!email._vt_results[name] || !email._vt_results[name].analyzed) {
            email._vt_results[name] = { status: 'scanning' };
        }
    }
    renderAttachments(email);

    for (const att of email.attachments) {
        const name = typeof att === 'string' ? att : (att.name || att.filename || 'file');
        const attId = (typeof att === 'object' && (att.id || att.name)) || name;
        try {
            const res = await apiFetch(
                '/api/messages/' + encodeURIComponent(messageId)
                + '/attachments/' + encodeURIComponent(attId) + '/analyze',
                { method: 'POST' }
            );
            const data = await res.json();
            email._vt_results[name] = data;
        } catch (e) {
            email._vt_results[name] = { error: 'request failed' };
            console.warn('[PhishGuard] attachment analyze failed for ' + name + ':', e);
        }
        renderAttachments(email);
    }
}

/* ======================== SCAN RESULTS DISPLAY ======================== */

/* Single source of truth for the 0-100 risk score shown by the ring, the
   brief, and the verdict card — so they can never show different numbers for
   the same email. Prefer the backend risk_score; fall back to a verdict-
   derived value for older restored scans that only carry prediction +
   confidence. */
function riskScoreOf(result) {
    if (!result) return 0;
    if (result.risk_score !== undefined) return result.risk_score;
    if (result.confidence !== undefined) {
        return result.prediction === 1
            ? Math.round(result.confidence * 100)
            : Math.round((1 - result.confidence) * 100);
    }
    return 0;
}

function showScanBrief(idx) {
    const result = scanResultFor(idx);
    if (!result) return;

    const email = state.emails[idx];
    const isPhishing = result.prediction === 1;
    const riskScore = riskScoreOf(result);

    const briefEl = document.getElementById('scanBrief');
    const iconEl = document.getElementById('scanBriefIcon');
    const textEl = document.getElementById('scanBriefText');

    briefEl.className = 'scan-brief ' + (isPhishing ? 'brief-danger' : 'brief-safe');
    iconEl.textContent = isPhishing ? '\u26A0\uFE0F' : '\u2705';

    // Use the assessment's own one-line summary so the brief, the report, and
    // the score all tell the same story. Fall back for older restored scans.
    const a = result.assessment;
    const reason = (a && a.summary)
        ? a.summary
        : (isPhishing ? 'This message matches known phishing patterns.'
                      : 'No phishing indicators were found.');

    textEl.textContent = (isPhishing ? 'Threat Identified' : 'No Threats Detected') +
        ' (risk score ' + riskScore + '/100). ' + reason;

    briefEl.style.display = 'flex';

    // Reply warning
    const replyWarn = document.getElementById('replyWarning');
    if (isPhishing) {
        replyWarn.style.display = 'flex';
        document.getElementById('replyWarningText').textContent =
            'Security Alert: This message has been identified as a potential phishing threat. Do not reply, open links, or download attachments.';
    } else {
        replyWarn.style.display = 'none';
    }

    // Keep a "Rescan" button visible so the user can re-scan; the detailed
    // report lives in the always-visible analysis rail.
    const _sb = document.getElementById('scanBtn');
    _sb.style.display = '';
    _sb.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg> Rescan';
}

function showScanResults(idx) {
    const result = scanResultFor(idx);
    if (!result) return;
    const email = state.emails[idx];

    // Quick summary banner above the body, then the analysis rail + links.
    showScanBrief(idx);
    renderAssessment(result, email);
    checkContactMismatch(email);
}

/* ── Unified security report ──────────────────────────────────────────────
   One renderer for the whole detailed report: a verdict header + three
   expandable dimension cards (Content / Links / Sender) + recommended
   actions. Everything is driven by result.assessment, so the report can
   never disagree with the score. */
const _ICON_WARN = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
const _ICON_CHECK = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';

function _dimFindingsHtml(findings) {
    if (!findings || !findings.length) return '';
    return '<ul class="dim-findings">' +
        findings.map(f => '<li>' + escapeHtml(f) + '</li>').join('') + '</ul>';
}

function _dimLinksHtml(dim) {
    let h = _dimFindingsHtml(dim.findings);
    if (dim.links && dim.links.length) {
        h += '<div class="dim-links">';
        for (const l of dim.links) {
            const shown = (typeof defangUrl === 'function') ? defangUrl(l.url) : l.url;
            h += '<div class="dim-link dim-' + l.level + '">' +
                '<div class="dim-link-url">' + escapeHtml(shown) + '</div>' +
                '<div class="dim-link-foot"><span class="dim-link-badge">' + l.score + '</span>' +
                '<span class="dim-link-reason">' + escapeHtml(l.reason) + '</span></div>' +
                '</div>';
        }
        h += '</div>';
    }
    return h;
}

function _dimCardHtml(name, dim, detailHtml, open) {
    return '<div class="dim-card dim-' + dim.level + (open ? ' open' : '') + '">' +
        '<button class="dim-head" type="button">' +
            '<span class="dim-name">' + name + '</span>' +
            '<span class="dim-cap">' + escapeHtml(dim.summary || '') + '</span>' +
            '<span class="dim-meter"><span class="dim-meter-fill" style="width:' + dim.score + '%"></span></span>' +
            '<span class="dim-score">' + dim.score + '</span>' +
            '<svg class="dim-chev" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>' +
        '</button>' +
        '<div class="dim-detail">' + detailHtml + '</div>' +
    '</div>';
}

const _ICON_INFO_S = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';

/* Render the right-hand ANALYSIS RAIL: Threat Score + Header Analysis +
   Threats Found + Recommended action. Driven entirely by result.assessment. */
function renderAssessment(result, email) {
    const rail = document.getElementById('railContent');
    const railEmpty = document.getElementById('railEmpty');
    if (!rail) return;
    const a = result.assessment;
    const isP = result.prediction === 1;
    const rs = riskScoreOf(result);
    const tier = rs >= 65 ? 'tier-danger' : (rs >= 40 ? 'tier-warn' : 'tier-safe');

    let html = '';
    // Threat score
    html += '<div class="rail-section rail-score ' + tier + '">' +
        '<div class="rail-cap">Threat Score</div>' +
        '<div class="rail-score-num">' + rs + '</div>' +
        '<div class="rail-score-tag">' + (rs >= 65 ? 'High threat' : (rs >= 40 ? 'Caution advised' : 'Low risk')) + '</div>' +
    '</div>';

    // Risk breakdown — the Content / Links / Sender bars that make up the score
    const dims = (a && a.dimensions) || null;
    if (dims) {
        const lvlClass = (lv) => lv === 'high' ? 'lvl-high' : (lv === 'elevated' ? 'lvl-elevated' : 'lvl-low');
        const bar = (label, dim, tip) => {
            const sc = Math.max(0, Math.min(100, dim.score | 0));
            return '<div class="rb-row">' +
                '<span class="rb-label has-tip" data-tip="' + escapeHtml(tip) + '">' + label + '</span>' +
                '<span class="rb-meter"><span class="rb-fill ' + lvlClass(dim.level) +
                '" style="width:' + sc + '%"></span></span>' +
                '<span class="rb-val">' + sc + '</span></div>';
        };
        html += '<div class="rail-section"><div class="rail-cap">Risk Breakdown</div>' +
            bar('Content', dims.content, 'How much the email’s wording reads like phishing — the AI text model plus urgency, credential and money-lure detection.') +
            bar('Links', dims.links, 'The riskiest link in the email, from the URL model, structural red flags, and threat intelligence (Google Safe Browsing, urlscan, VirusTotal, URLhaus).') +
            bar('Sender', dims.sender, 'Reputation of the sender’s domain and the IP address the mail was actually sent from.') +
            (dims.auth ? bar('Auth', dims.auth, 'Whether the email proved who it’s from — SPF/DKIM/DMARC, domain alignment and Reply-To checks.') : '') +
            (dims.files ? bar('Files', dims.files, 'Risk from attachments, judged by file type and VirusTotal verdict.') : '') +
        '</div>';
    }

    const providerOutcomes = (
        result.threat_intel && result.threat_intel.provider_outcomes
    ) || {};
    const providerOutcomeLabels = {
        flagged: 'Flagged',
        not_flagged: 'Not flagged',
        not_found: 'Not found',
        not_checked: 'Not checked',
        provider_unavailable: 'Provider unavailable',
    };
    const providerIds = Object.keys(PROVIDER_DISPLAY_NAMES);
    if (providerIds.some(id => providerOutcomes[id])) {
        html += '<div class="rail-section"><div class="rail-cap">Remote Provider Checks</div>';
        for (const id of providerIds) {
            const item = providerOutcomes[id] || { outcome: 'not_checked' };
            const outcome = item.outcome || 'not_checked';
            const title = item.reason ? ' title="' + escapeHtml(item.reason) + '"' : '';
            html += '<div class="provider-check-row"><span>'
                + escapeHtml(PROVIDER_DISPLAY_NAMES[id]) + '</span><span class="provider-outcome outcome-'
                + escapeHtml(outcome) + '"' + title + '>'
                + escapeHtml(providerOutcomeLabels[outcome] || 'Not checked') + '</span></div>';
        }
        html += '</div>';
    }

    // Header analysis
    const hr = (email && email.headers) || result.header_result || {};
    const authState = (v) => {
        v = String(v || '').toLowerCase();
        if (v === 'pass') return ['Pass', 'ha-pass'];
        if (v === 'fail') return ['Fail', 'ha-fail'];
        return ['Missing', 'ha-none'];
    };
    const spf = authState(hr.spf), dkim = authState(hr.dkim), dmarc = authState(hr.dmarc);
    const anyFail = [spf, dkim, dmarc].some(x => x[1] === 'ha-fail');
    const allPass = [spf, dkim, dmarc].every(x => x[1] === 'ha-pass');
    const domainAuth = anyFail ? ['Fail', 'ha-fail'] : (allPass ? ['Pass', 'ha-pass'] : ['Partial', 'ha-none']);
    const links = (a && a.dimensions && a.dimensions.links && a.dimensions.links.links) || [];
    const linksCount = links.length || ((result.url_analysis && result.url_analysis.urls_found || []).length);
    const attachCount = (email && email.attachments && email.attachments.length) || 0;
    const haRow = (label, st, tip) => {
        const lbl = tip
            ? '<span class="ha-label has-tip" data-tip="' + escapeHtml(tip) + '">' + label + '</span>'
            : '<span class="ha-label">' + label + '</span>';
        return '<div class="ha-row">' + lbl + '<span class="ha-val ' + st[1] + '">' + st[0] + '</span></div>';
    };
    // Deeper sender-identity checks from the Auth dimension: does the visible
    // From align with the authenticated domain, and does Reply-To point elsewhere.
    const authDim = (a && a.dimensions && a.dimensions.auth) || {};
    const authDetail = authDim.detail || {};
    const alignSt = authDetail.aligned === true ? ['Aligned', 'ha-pass']
        : (authDetail.aligned === false ? ['Misaligned', 'ha-fail'] : ['Unknown', 'ha-none']);
    const replySt = authDetail.reply_to_mismatch ? ['Mismatch', 'ha-fail'] : ['OK', 'ha-pass'];
    const providerSignal = authDetail.provider_signal || null;
    const providerSt = providerSignal
        ? [providerSignal.label || 'Flagged',
            providerSignal.status === 'fail' ? 'ha-fail' : 'ha-warn']
        : null;
    html += '<div class="rail-section"><div class="rail-cap">Header Analysis</div>' +
        haRow('Domain Auth', domainAuth, 'The combined result of the email’s SPF, DKIM and DMARC checks — “Pass” means all that were present passed.') +
        haRow('DKIM', dkim, 'DKIM (DomainKeys Identified Mail): a cryptographic signature proving the message genuinely came from the domain and wasn’t altered in transit.') +
        haRow('SPF', spf, 'SPF (Sender Policy Framework): checks the server that sent the mail is one the domain authorised to send on its behalf.') +
        haRow('DMARC', dmarc, 'DMARC: ties the SPF/DKIM results to the visible “From” domain and tells receivers what to do when they fail.') +
        '<div class="ha-sep"></div>' +
        haRow('Domain Alignment', alignSt, 'Whether the visible “From” domain matches the domain that actually passed authentication. A mismatch is a common sign of spoofing.') +
        haRow('Reply-To', replySt, 'Whether a reply would go to a different domain than the sender. Phishers often set Reply-To to an address they control.') +
        (providerSt ? haRow('Provider Signal', providerSt, providerSignal.message || 'Microsoft mail protection added an additional authentication or spam-confidence signal.') : '') +
        '<div class="ha-sep"></div>' +
        haRow('Links Checked', [linksCount, 'ha-num'], 'How many links were extracted from this email and checked against the threat-intelligence sources.') +
        haRow('Attachments', [attachCount, 'ha-num'], 'How many file attachments this email has. Each is judged by type and checked on VirusTotal.') +
    '</div>';

    // Threats found
    const threats = (a && a.threats) || [];
    if (threats.length) {
        html += '<div class="rail-section"><div class="rail-cap">Threats Found (' + threats.length + ')</div>';
        for (const t of threats) {
            html += '<div class="threat-card threat-' + (t.severity || 'medium') + '">' +
                '<div class="threat-title"><span class="threat-dot"></span>' + escapeHtml(t.title) + '</div>' +
                '<div class="threat-desc">' + escapeHtml(t.desc) + '</div></div>';
        }
        html += '</div>';
    }

    // Recommended action
    const actions = (a && a.actions) || [];
    html += '<div class="rail-section rail-reco"><div class="reco-head">' + _ICON_INFO_S + ' Recommended action</div>';
    if (actions.length) {
        html += '<ul>' + actions.map(x => '<li>' + escapeHtml(x) + '</li>').join('') + '</ul>';
    } else {
        html += '<div class="reco-safe">No action needed — this message looks safe.</div>';
    }
    html += '</div>';

    rail.innerHTML = html;
    rail.style.display = '';
    if (railEmpty) railEmpty.style.display = 'none';

    renderReadingLinks(result);
}

/* Reset the rail to its empty state (no scan yet for the open email). */
function clearAnalysisRail() {
    const rail = document.getElementById('railContent');
    const railEmpty = document.getElementById('railEmpty');
    if (rail) { rail.innerHTML = ''; rail.style.display = 'none'; }
    if (railEmpty) railEmpty.style.display = '';
    const pl = document.getElementById('previewLinks');
    if (pl) pl.style.display = 'none';
}

/* Reading-pane LINKS section — the links found in the email body. */
function renderReadingLinks(result) {
    const wrap = document.getElementById('previewLinks');
    const list = document.getElementById('previewLinksList');
    const cnt = document.getElementById('previewLinksCount');
    if (!wrap || !list) return;
    const a = result && result.assessment;
    const links = (a && a.dimensions && a.dimensions.links && a.dimensions.links.links) || [];
    if (!links.length) { wrap.style.display = 'none'; return; }
    if (cnt) cnt.textContent = 'Links (' + links.length + ')';
    list.innerHTML = links.map(l => {
        const shown = (typeof defangUrl === 'function') ? defangUrl(l.url) : l.url;
        const cls = l.level === 'high' ? 'pl-danger' : (l.level === 'elevated' ? 'pl-warn' : 'pl-safe');
        // Per-source threat-intel results (Google Safe Browsing / urlscan / etc.)
        const checks = (l.factors && l.factors.checks) || [];
        const badges = checks.map(c => {
            const rc = (c.result === 'flagged' || c.result === 'malicious') ? 'plc-bad'
                : (c.result === 'suspicious' ? 'plc-warn'
                : (c.result === 'unknown' ? 'plc-neutral' : 'plc-ok'));
            return '<span class="pl-check ' + rc + '">' + escapeHtml(c.src) +
                ' <b>' + escapeHtml(c.result) + '</b></span>';
        }).join('');
        return '<div class="pl-item ' + cls + '"><div class="pl-url">' + escapeHtml(shown) +
            '</div><div class="pl-reason">' + escapeHtml(l.reason || '') + '</div>' +
            (badges ? '<div class="pl-checks">' + badges + '</div>' : '') + '</div>';
    }).join('');
    wrap.style.display = '';
}

/* ======================== CONTACT MISMATCH (Feature 5) ======================== */

function checkContactMismatch(email) {
    const alert = document.getElementById('mismatchAlert');
    const alertText = document.getElementById('mismatchAlertText');

    const name = (email.sender_name || '').toLowerCase();
    const addr = (email.sender || '').toLowerCase();
    const domain = addr.split('@')[1] || '';

    const configuredBrands = state.detectionRules.brand_name_tokens || {};
    const brands = Object.keys(configuredBrands).length
        ? Object.fromEntries(Object.entries(configuredBrands).map(([brand, domain]) => [brand, [domain]]))
        : {
            'microsoft': ['microsoft.com', 'outlook.com', 'live.com', 'hotmail.com'],
            'apple': ['apple.com', 'icloud.com', 'me.com'],
            'google': ['google.com', 'gmail.com', 'youtube.com'],
            'amazon': ['amazon.com', 'amazon.co.uk', 'amazonaws.com'],
            'paypal': ['paypal.com'],
            'chase': ['chase.com', 'jpmorgan.com'],
            'bank of america': ['bankofamerica.com', 'bofa.com'],
            'wells fargo': ['wellsfargo.com'],
            'netflix': ['netflix.com'],
            'facebook': ['facebook.com', 'meta.com', 'fb.com'],
            'instagram': ['instagram.com'],
            'linkedin': ['linkedin.com'],
            'twitter': ['twitter.com', 'x.com'],
        };

    let mismatch = false;
    let brandName = '';
    let expectedDomains = [];

    for (const [brand, domains] of Object.entries(brands)) {
        if (name.includes(brand)) {
            if (!domainMatches(domain, domains)) {
                mismatch = true;
                brandName = brand.charAt(0).toUpperCase() + brand.slice(1);
                expectedDomains = domains;
                break;
            }
        }
    }

    if (mismatch) {
        alert.style.display = 'flex';
        alertText.textContent = 'Sender Mismatch: This message identifies as ' + brandName + ' but originates from ' + domain + ' (legitimate domain: ' + expectedDomains[0] + ')';
    } else {
        alert.style.display = 'none';
    }
}

/* ======================== THEME SYSTEM ======================== */

function setTheme(name) {
    state.theme = name;
    document.documentElement.setAttribute('data-theme', name);
    localStorage.setItem('pg-theme', name);

    document.querySelectorAll('.theme-swatch').forEach(sw => {
        sw.classList.toggle('selected', sw.getAttribute('data-theme') === name);
    });
}

function toggleMode() {
    state.mode = state.mode === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-mode', state.mode);
    localStorage.setItem('pg-mode', state.mode);

    const isLight = state.mode === 'light';
    document.getElementById('modeToggleTop').checked = isLight;
    document.getElementById('modeToggleModal').checked = isLight;
}

function setMode(mode) {
    state.mode = mode;
    document.documentElement.setAttribute('data-mode', mode);
    localStorage.setItem('pg-mode', mode);

    const isLight = mode === 'light';
    document.getElementById('modeToggleTop').checked = isLight;
    document.getElementById('modeToggleModal').checked = isLight;
}

function setFontSize(size) {
    state.fontSize = size;
    document.documentElement.style.setProperty('--font-size', size + 'px');
    localStorage.setItem('pg-fontSize', size);
    document.getElementById('fontSizeVal').textContent = size;
    document.getElementById('fontSlider').value = size;
}

function setListDensity(compact) {
    state.compactList = !!compact;
    document.documentElement.classList.toggle('compact-list', state.compactList);
    localStorage.setItem('pg-compactList', state.compactList ? '1' : '0');
    const densityToggle = document.getElementById('densityToggle');
    if (densityToggle) densityToggle.checked = state.compactList;
}

function loadPreferences() {
    const savedTheme = localStorage.getItem('pg-theme');
    const savedMode = localStorage.getItem('pg-mode');
    const savedFontSize = localStorage.getItem('pg-fontSize');
    const savedCompactList = localStorage.getItem('pg-compactList');

    // 'indigo' is the default (no override → mode-block accents). Older saved
    // values like 'cupertino' simply fall through to the default palette.
    setTheme(savedTheme || 'indigo');

    if (savedMode) setMode(savedMode);
    else setMode('dark');

    if (savedFontSize) setFontSize(parseInt(savedFontSize, 10));
    else setFontSize(11);

    setListDensity(savedCompactList === '1');
}

/* ======================== SETTINGS MODAL ======================== */

function openSettings() {
    clearProviderKeyInputs();
    document.getElementById('settingsModal').style.display = 'flex';
    checkAuthStatus();
    refreshProviderSettings();
}

function closeSettings() {
    clearProviderKeyInputs();
    document.getElementById('settingsModal').style.display = 'none';
}

const PROVIDER_DISPLAY_NAMES = {
    virustotal: 'VirusTotal',
    google_safe_browsing: 'Google Safe Browsing',
    urlscan: 'urlscan.io',
    abuseipdb: 'AbuseIPDB',
};

function clearProviderKeyInputs() {
    document.querySelectorAll('[data-provider-key]').forEach(input => {
        input.value = '';
    });
}

function setProviderFeedback(card, message, type) {
    const element = card && card.querySelector('[data-provider-feedback]');
    if (!element) return;
    element.textContent = message || '';
    element.className = 'provider-feedback'
        + (type === 'error' ? ' feedback-error' : (type === 'ok' ? ' feedback-ok' : ''));
}

function providerSettingsBridge() {
    const bridge = window.electron && window.electron.providerSettings;
    return bridge && typeof bridge.get === 'function' ? bridge : null;
}

async function refreshProviderSettings() {
    const cards = document.querySelectorAll('.provider-card[data-provider]');
    const notice = document.getElementById('providerStorageNotice');
    const bridge = providerSettingsBridge();
    let local = null;
    let backend = null;

    try {
        const requests = [apiFetch('/api/settings').then(r => r.json())];
        if (bridge) requests.push(bridge.get());
        const values = await Promise.all(requests);
        backend = values[0];
        local = values[1] || null;
    } catch (_error) {
        if (notice) {
            notice.textContent = 'Provider settings are temporarily unavailable.';
            notice.classList.add('storage-warning');
        }
        return;
    }

    const secure = local && local.ok && local.secureStorage;
    if (notice) {
        notice.classList.toggle('storage-warning', !secure || !secure.available);
        if (!bridge) {
            notice.textContent = 'User API keys can only be managed in the Electron desktop app.';
        } else if (!secure || !secure.available) {
            notice.textContent = (secure && secure.reason)
                ? secure.reason + ' Existing deployer keys can still be enabled or disabled.'
                : 'Protected credential storage is unavailable. User keys cannot be saved.';
        } else {
            notice.textContent = 'User keys are encrypted by the operating system and kept on this device.';
        }
    }

    const backendProviders = Object.fromEntries(
        ((backend && backend.providers) || []).map(item => [item.id, item])
    );
    const localProviders = (local && local.ok && local.providers) || {};
    const statusLabels = {
        enabled: 'Enabled',
        disabled: 'Disabled',
        not_configured: 'Not configured',
        unavailable: 'Unavailable',
        rate_limited: 'Rate-limited',
    };

    cards.forEach(card => {
        const id = card.dataset.provider;
        const server = backendProviders[id] || {
            configured: false,
            enabled: false,
            status: 'unavailable',
            reason: 'No status received',
            key_source: 'none',
        };
        const stored = localProviders[id] || {};
        const status = card.querySelector('[data-provider-status]');
        const source = card.querySelector('[data-provider-source]');
        const toggle = card.querySelector('[data-provider-enabled]');
        const input = card.querySelector('[data-provider-key]');
        const save = card.querySelector('[data-provider-save]');
        const remove = card.querySelector('[data-provider-remove]');

        status.textContent = statusLabels[server.status] || 'Unavailable';
        status.className = 'provider-status status-' + (server.status || 'unavailable');
        status.title = server.reason || '';
        source.textContent = stored.userKeyConfigured
            ? (stored.userKeyAvailable === false ? 'Stored key unavailable' : 'User key')
            : (server.key_source === 'deployer' ? 'Deployer key' : '');
        toggle.checked = Boolean(server.enabled);
        toggle.disabled = !bridge || !server.configured;
        input.disabled = !bridge || !secure || !secure.available;
        save.disabled = input.disabled;
        input.placeholder = stored.userKeyConfigured
            ? (stored.userKeyAvailable === false
                ? 'Stored key unavailable; enter to replace'
                : '•••••••• (stored; enter to replace)')
            : (server.key_source === 'deployer'
                ? '•••••••• (deployer key; enter to override)'
                : 'Enter API key');
        remove.hidden = !bridge || !stored.userKeyConfigured;
        card.dataset.configured = server.configured ? '1' : '0';
        if (server.reason && server.status !== 'enabled') {
            setProviderFeedback(card, server.reason, server.status === 'unavailable' ? 'error' : '');
        } else {
            setProviderFeedback(card, '', '');
        }
    });
}

async function saveProviderKey(card) {
    const bridge = providerSettingsBridge();
    if (!bridge || !card) return;
    const id = card.dataset.provider;
    const input = card.querySelector('[data-provider-key]');
    const key = input.value.trim();
    input.value = '';
    if (!key) {
        setProviderFeedback(card, 'Enter an API key to save.', 'error');
        return;
    }
    setProviderFeedback(card, 'Saving…', '');
    let response;
    try {
        response = await bridge.update(id, { key, enabled: true });
    } catch (_error) {
        setProviderFeedback(card, 'Could not save key.', 'error');
        return;
    }
    if (!response || !response.ok) {
        setProviderFeedback(card, (response && response.reason) || 'Could not save key.', 'error');
        return;
    }
    setProviderFeedback(card, 'Key saved on this device.', 'ok');
    await new Promise(resolve => setTimeout(resolve, 80));
    await refreshProviderSettings();
}

async function toggleProvider(card, enabled) {
    const bridge = providerSettingsBridge();
    if (!bridge || !card) return;
    const toggle = card.querySelector('[data-provider-enabled]');
    toggle.disabled = true;
    let response;
    try {
        response = await bridge.update(
            card.dataset.provider,
            { enabled: Boolean(enabled) }
        );
    } catch (_error) {
        response = null;
    }
    if (!response || !response.ok) {
        setProviderFeedback(card, (response && response.reason) || 'Could not update provider.', 'error');
    }
    await new Promise(resolve => setTimeout(resolve, 80));
    await refreshProviderSettings();
}

async function removeProviderKey(card) {
    const bridge = providerSettingsBridge();
    if (!bridge || !card) return;
    const name = PROVIDER_DISPLAY_NAMES[card.dataset.provider] || 'provider';
    if (!window.confirm('Remove your locally stored ' + name + ' key? A deployer key, if present, will be used instead.')) {
        return;
    }
    let response;
    try {
        response = await bridge.clearKey(card.dataset.provider);
    } catch (_error) {
        response = null;
    }
    if (!response || !response.ok) {
        setProviderFeedback(card, (response && response.reason) || 'Could not remove key.', 'error');
        return;
    }
    await new Promise(resolve => setTimeout(resolve, 80));
    await refreshProviderSettings();
}

/* ======================== OUTLOOK CONNECTION ======================== */

/* Saved Outlook accounts now live in Supabase (user_accounts table),
   loaded into state.savedAccounts on Supabase sign-in and cleared on
   sign-out. The localStorage `pg-accounts` key is no longer used. */

function setReconnectBannerVisible(visible) {
    const banner = document.getElementById('reconnectBanner');
    if (banner) banner.style.display = visible ? 'flex' : 'none';
}

async function checkAuthStatus() {
    try {
        const r = await fetch('/api/auth/status');
        const data = await r.json();
        const connectBtn = document.getElementById('connectBtn');
        const userLabel = document.getElementById('topbarUser');
        const statusEl = document.getElementById('connectionStatus');
        const profileSection = document.getElementById('sidebarProfile');

        // Reveal the "Connect Outlook" recovery button whenever we have
        // a Supabase session but Flask says no Outlook is connected.
        const reconnectBtn = document.getElementById('pgReconnectBtn');
        const needsReconnect = Boolean(state.user && !data.connected);
        if (reconnectBtn) {
            reconnectBtn.style.display = needsReconnect ? 'inline-flex' : 'none';
        }
        setReconnectBannerVisible(needsReconnect);

        if (data.connected) {
            connectBtn.style.display = 'none';   // legacy button, always hidden
            userLabel.textContent = data.user_name || 'Connected';
            if (statusEl) statusEl.textContent = 'Connected as ' + (data.user_email || data.user_name);

            /* Show profile card */
            profileSection.style.display = 'block';
            document.getElementById('profileName').textContent = data.user_name || 'User';
            document.getElementById('profileEmail').textContent = data.user_email || '';
            document.getElementById('popupCurrentName').textContent = data.user_name || 'User';
            document.getElementById('popupCurrentEmail').textContent = data.user_email || '';

            /* Sidebar account card (mockup) */
            const _acctEmail = document.getElementById('sbAcctEmail');
            if (_acctEmail) _acctEmail.textContent = data.user_email || data.user_name || 'Connected';
            const _acctAv = document.getElementById('sbAcctAvatar');
            if (_acctAv) _acctAv.textContent = ((data.user_name || data.user_email || 'A').trim()[0] || 'A').toUpperCase();
            const _acctSt = document.getElementById('sbAcctStatus');
            if (_acctSt) _acctSt.innerHTML = '<span class="sb-acct-dot"></span>Connected';

            /* Try loading profile photo */
            const avatar = document.getElementById('profileAvatar');
            const popupAvatar = document.getElementById('popupCurrentAvatar');
            const img = new Image();
            img.onload = () => {
                avatar.innerHTML = '';
                avatar.appendChild(img);
                const img2 = img.cloneNode(true);
                popupAvatar.innerHTML = '';
                popupAvatar.appendChild(img2);
            };
            img.src = '/api/auth/photo?t=' + Date.now();

            /* Persist this Outlook account to Supabase so it shows up
               in the account-switcher popup on every device. We only
               write when the user has an active Supabase session — if
               they haven't signed into PhishGuard yet, the list is
               local-only. */
            if (data.user_email && state.user) {
                const exists = state.savedAccounts.find(a => a.email === data.user_email);
                if (!exists) {
                    try {
                        await window.pg.accounts.add({
                            email: data.user_email,
                            name: data.user_name,
                        });
                        state.savedAccounts.push({
                            email: data.user_email,
                            name: data.user_name || '',
                        });
                    } catch (e) {
                        console.warn('[PhishGuard] could not save Outlook account to Supabase:', e);
                    }
                }
            }
            renderPopupAccounts(data.user_email);
        } else {
            // Legacy connect path is dead — Microsoft sign-in modal is
            // the only entry. Keep the button hidden no matter what.
            connectBtn.style.display = 'none';
            userLabel.textContent = '';
            if (statusEl) statusEl.textContent = '';
            profileSection.style.display = 'none';
            closeProfilePopup();
            const _ae = document.getElementById('sbAcctEmail');
            if (_ae) _ae.textContent = 'Not connected';
            const _as = document.getElementById('sbAcctStatus');
            if (_as) _as.innerHTML = '<span class="sb-acct-dot off"></span>Offline';
        }
    } catch (e) { /* ignore */ }
}

function renderPopupAccounts(currentEmail) {
    const container = document.getElementById('popupAccounts');
    const others = state.savedAccounts.filter(a => a.email !== currentEmail);
    if (others.length === 0) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = others.map(a =>
        '<div class="profile-popup-account" data-email="' + escapeHtml(a.email) + '">' +
        '<div class="profile-popup-avatar"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>' +
        '<div class="profile-popup-info">' +
        '<div class="profile-popup-name">' + escapeHtml(a.name) + '</div>' +
        '<div class="profile-popup-email">' + escapeHtml(a.email) + '</div>' +
        '</div>' +
        '<button class="profile-popup-remove" title="Remove this saved account" data-email="' + escapeHtml(a.email) + '">' +
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>' +
        '</button>' +
        '</div>'
    ).join('');
    container.querySelectorAll('.profile-popup-remove').forEach(btn => {
        btn.addEventListener('click', async (ev) => {
            ev.stopPropagation();
            const email = btn.getAttribute('data-email');
            if (!email) return;
            btn.disabled = true;
            try {
                await window.pg.accounts.remove(email);
                state.savedAccounts = state.savedAccounts.filter(a => a.email !== email);
                renderPopupAccounts(currentEmail);
            } catch (e) {
                console.warn('[PhishGuard] could not remove account', email, e);
                btn.disabled = false;
            }
        });
    });
    container.querySelectorAll('.profile-popup-account').forEach(el => {
        el.addEventListener('click', () => {
            closeProfilePopup();
            handleConnectClick();
        });
    });
}

function toggleProfilePopup() {
    const section = document.getElementById('sidebarProfile');
    section.classList.toggle('open');
}

function closeProfilePopup() {
    document.getElementById('sidebarProfile').classList.remove('open');
}

function openConnectModal() {
    document.getElementById('connectModal').style.display = 'flex';
    const savedCid = localStorage.getItem('phishguard_client_id');
    if (savedCid) {
        document.getElementById('clientIdInput').value = savedCid;
    }
    checkAuthStatus();
}

function closeConnectModal() {
    document.getElementById('connectModal').style.display = 'none';
}

async function handleConnectClick() {
    try {
        const r = await fetch('/api/auth/status');
        const data = await r.json();

        if (data.connected) {
            // Disconnect
            await apiFetch('/api/auth/disconnect', { method: 'POST' });
            setStatus('Disconnected — showing mock emails');
            await loadEmails();
            checkAuthStatus();
            state.selectedIdx = null;
            document.getElementById('emailPreview').style.display = 'none';
            document.getElementById('dashboardView').style.display = '';
            return;
        }

        // Not connected — open the connect modal
        openConnectModal();
    } catch (e) {
        setStatus('Connection error: ' + e.message);
    }
}

async function connectOutlook() {
    const clientId = document.getElementById('clientIdInput').value.trim();
    const statusEl = document.getElementById('connectionStatus');

    if (!clientId) {
        if (statusEl) statusEl.textContent = 'Please enter your Client ID.';
        return;
    }

    localStorage.setItem('phishguard_client_id', clientId);
    if (statusEl) statusEl.textContent = 'Opening Microsoft sign-in...';
    setStatus('Opening Microsoft sign-in...');

    try {
        const resp = await apiFetch('/api/auth/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ client_id: clientId }),
        });
        const result = await resp.json();

        if (result.error) {
            if (statusEl) statusEl.textContent = 'Error: ' + result.error;
            setStatus('Error: ' + result.error);
            return;
        }

        // Open Microsoft login in a popup
        window.open(result.auth_url, 'outlookAuth', 'width=600,height=700');

        // Poll for completion
        if (statusEl) statusEl.textContent = 'Waiting for sign-in...';
        let pollDone = false;
        const pollInterval = setInterval(async () => {
            if (pollDone) return;
            try {
                const status = await fetch('/api/auth/status');
                const statusData = await status.json();
                if (statusData.connected) {
                    pollDone = true;
                    clearInterval(pollInterval);
                    if (statusEl) statusEl.textContent = 'Connected as ' + (statusData.user_email || statusData.user_name);
                    setStatus('Connected as ' + (statusData.user_email || statusData.user_name));
                    closeConnectModal();
                    await loadEmails();
                    checkAuthStatus();
                }
            } catch (e) { /* keep polling */ }
        }, 1500);

        setTimeout(() => {
            if (!pollDone) {
                clearInterval(pollInterval);
                pollDone = true;
                if (statusEl && statusEl.textContent === 'Waiting for sign-in...') {
                    statusEl.textContent = 'Timed out. Please try again.';
                }
            }
        }, 120000);

    } catch (e) {
        if (statusEl) statusEl.textContent = 'Connection error: ' + e.message;
        setStatus('Connection error: ' + e.message);
    }
}

async function refreshOutlookEmails() {
    try {
        setStatus('Refreshing emails...', 'scanning');
        const r = await apiFetch('/api/auth/refresh', { method: 'POST' });
        const data = await r.json();
        if (data.error) {
            setStatus('Refresh failed: ' + data.error, 'error');
            return;
        }
        // Respect the current nav. If the user is sitting in Junk, a
        // refresh should re-fetch junk — not blow away the junk list
        // by loading the inbox.
        const inJunk = state.navView === 'junk';
        const res = await fetch(inJunk ? '/api/emails/junk' : '/api/emails');
        if (res.ok) {
            const emailData = await res.json();
            state.emails = emailData.emails || [];
            state.selectedIdx = null;
            renderEmailList();
            document.getElementById('emailPreview').style.display = 'none';
            document.getElementById('dashboardView').style.display = '';
        }
        setStatus('Loaded ' + (data.count || state.emails.length) + ' emails', '');
    } catch (e) {
        setStatus('Refresh error: ' + e.message, 'error');
    }
}

/* ======================== NAV ======================== */

async function setNav(view) {
    /* When changing between inbox-family views (inbox/threats/safe/starred)
       state.emails stays the same — only the filter changes. When entering
       or leaving the junk folder, state.emails has to be swapped between
       the inbox list and the junk list. */
    const prevWasJunk = state.navView === 'junk';
    const newIsJunk = view === 'junk';
    state.navView = view;
    state.selectedIdx = null;
    document.getElementById('emailPreview').style.display = 'none';
    const dashEl = document.getElementById('dashboardView');
    dashEl.style.display = '';
    dashEl.classList.remove('dash-enter');
    void dashEl.offsetWidth;
    dashEl.classList.add('dash-enter');
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-nav') === view);
    });

    /* Keep the topbar title in sync with the active view (it used to
       stay "Inbox" forever). */
    const titleEl = document.querySelector('.topbar-title');
    if (titleEl) {
        const titles = { inbox: 'Inbox', threats: 'Threats', safe: 'Safe',
                         starred: 'Starred', junk: 'Junk' };
        titleEl.textContent = titles[view] || 'Inbox';
    }

    const list = document.getElementById('emailList');
    list.classList.add('fade-out');
    list.classList.remove('fade-in');

    if (newIsJunk) {
        await loadJunkEmails();
    } else if (prevWasJunk) {
        // Coming back from junk — reload inbox so state.emails restores.
        await loadEmails();
    } else {
        // Inbox-family view to another inbox-family view; same data,
        // different filter.
        listAnimateNext = true;
        renderEmailList();
    }

    // Trigger fade-in after a short delay
    setTimeout(() => {
        list.classList.remove('fade-out');
        list.classList.add('fade-in');
    }, 50);
}

async function loadJunkEmails() {
    setStatus('Loading junk folder...', 'scanning');
    try {
        const res = await fetch('/api/emails/junk');
        if (!res.ok) throw new Error('Failed to load junk emails');
        const data = await res.json();
        state.emails = data.emails || [];
        // DO NOT clear scanResults — those are keyed by message_id and
        // legitimately persist across folder views. Clearing them here
        // was wiping the user's scan history every time they clicked
        // Junk.
        state.selectedIdx = null;
        listAnimateNext = true;
        renderEmailList();
        document.getElementById('emailPreview').style.display = 'none';
        document.getElementById('dashboardView').style.display = '';
        setStatus('Junk folder — ' + state.emails.length + ' emails', '');
    } catch (err) {
        console.error('loadJunkEmails error:', err);
        setStatus('Error loading junk folder', 'error');
    }
}

async function moveToJunk(idx) {
    if (idx === null) return;
    const messageId = emailIdFromIdx(idx);
    if (!messageId) {
        setStatus('Could not resolve message', 'error');
        return;
    }
    setStatus('Moving to junk...', 'scanning');
    try {
        const res = await apiFetch('/api/messages/' + encodeURIComponent(messageId) + '/move-to-junk',
                                    { method: 'POST' });
        const data = await res.json();
        if (data.error) {
            setStatus('Failed: ' + data.error, 'error');
            return;
        }
        // Graph rewrites the message id when moving folders. Move the
        // existing scan result to the new id so its score ring still
        // shows up in the junk view (instead of disappearing).
        if (data.new_id && data.old_id && data.new_id !== data.old_id
            && state.scanResults[data.old_id]) {
            const sr = state.scanResults[data.old_id];
            sr.id = data.new_id;
            sr.messageId = data.new_id;
            state.scanResults[data.new_id] = sr;
            delete state.scanResults[data.old_id];
        }
        setStatus('Moved to junk folder', '');
        if (state.navView === 'junk') {
            await loadJunkEmails();
        } else {
            await loadEmails();
        }
    } catch (err) {
        setStatus('Failed to move email', 'error');
    }
}

/* ======================== EVENT LISTENERS ======================== */

document.addEventListener('DOMContentLoaded', async () => {
    loadPreferences();
    randomTip();
    // If Electron preload exposed a launch secret, pick it up so apiFetch
    // can prove the request came from the app the main process spawned.
    try {
        if (window.electron && typeof window.electron.launchSecret === 'string') {
            state.launchSecret = window.electron.launchSecret;
        }
    } catch (e) { /* non-electron context */ }
    // Fetch the CSRF token before any mutation runs.
    await ensureCsrfToken();
    await loadDetectionRules();
    loadEmails();

    /* Search */
    const searchInput = document.getElementById('searchInput');
    const searchClear = document.getElementById('searchClear');

    searchInput.addEventListener('input', () => {
        searchClear.classList.toggle('visible', searchInput.value.length > 0);
        showSearchAutocomplete(searchInput.value.trim());
        renderEmailList();
    });
    searchInput.addEventListener('focus', () => {
        if (searchInput.value.trim()) showSearchAutocomplete(searchInput.value.trim());
    });
    searchInput.addEventListener('blur', () => {
        setTimeout(() => { document.getElementById('searchAutocomplete').style.display = 'none'; }, 150);
    });

    searchClear.addEventListener('click', () => {
        searchInput.value = '';
        searchClear.classList.remove('visible');
        document.getElementById('searchAutocomplete').style.display = 'none';
        renderEmailList();
        searchInput.focus();
    });

    /* Sidebar Nav */
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setNav(btn.getAttribute('data-nav'));
        });
    });

    /* Refresh emails */
    document.getElementById('refreshBtn').addEventListener('click', () => {
        refreshOutlookEmails();
    });

    /* Live scan toggle — auto-scan on open. Persisted. */

    /* Scan / Hide toggle */
    document.getElementById('scanBtn').addEventListener('click', () => {
        if (state.selectedIdx === null) return;
        scanEmail(state.selectedIdx);
    });

    /* Header toggle */
    /* #15 Rich HTML body toggle */
    document.querySelectorAll('.preview-body-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.getAttribute('data-view');
            const bodyEl = document.getElementById('previewBody');
            const iframe = document.getElementById('previewBodyHtml');
            document.getElementById('bodyTogglePlain').classList.toggle('active', view === 'plain');
            document.getElementById('bodyToggleHtml').classList.toggle('active', view === 'html');

            if (view === 'html') {
                bodyEl.style.display = 'none';
                iframe.style.display = 'block';
                const htmlContent = iframe.getAttribute('data-html') || '';
                const isDark = document.documentElement.getAttribute('data-mode') === 'dark';
                /* Sanitize the HTML and hand it over via srcdoc. The iframe has
                   sandbox="" with no allow-same-origin / no allow-scripts, and a
                   strict CSP prevents any remote image or stylesheet from being
                   fetched even if the sanitiser misses something. */
                const safeHtml = sanitizeEmailHtml(htmlContent);
                /* Match the plain-text body: settings slider value + 2.5px
                   (the slider used to have no effect on the rich view). */
                const htmlFontPx = ((state.fontSize || 11) + 2.5) + 'px';
                const baseStyle =
                    'body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; font-size: ' + htmlFontPx + '; line-height: 1.55; padding: 0; margin: 0; ' +
                    (isDark ? 'background: transparent; color: #ADB0CE;' : 'background: transparent; color: #333;') +
                    ' } a { color: #8B93FF; pointer-events: none; text-decoration: underline; }';
                const csp = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'";
                iframe.srcdoc =
                    '<!DOCTYPE html><html><head>' +
                    '<meta http-equiv="Content-Security-Policy" content="' + csp + '">' +
                    '<style>' + baseStyle + '</style></head><body>' + safeHtml + '</body></html>';
            } else {
                bodyEl.style.display = '';
                iframe.style.display = 'none';
            }
        });
    });

    /* #13 Contact card from preview sender name */
    document.getElementById('previewSender').addEventListener('click', () => {
        if (state.selectedIdx === null) return;
        const email = state.emails[state.selectedIdx];
        if (email) showContactCard(email, document.getElementById('previewSender'));
    });

    /* Link preview on hover.
       The body now renders a defanged URL ("hxxps://evil[.]com/...") so a
       hover or stray copy-paste cannot accidentally produce a working link.
       The popup that appears asks the user to deliberately reveal the real
       URL — a click on the reveal button swaps in the original. Popup has
       pointer-events enabled (CSS) so the button is interactable; we use a
       short delay before dismissing on mouseleave so the cursor can travel
       from the link to the popup without the popup vanishing under it. */
    let linkPreviewEl = null;
    let linkPreviewHideTimer = null;
    let linkPreviewCurrentUrl = '';

    function dismissLinkPreview() {
        if (linkPreviewHideTimer) {
            clearTimeout(linkPreviewHideTimer);
            linkPreviewHideTimer = null;
        }
        if (linkPreviewEl) {
            linkPreviewEl.remove();
            linkPreviewEl = null;
        }
        linkPreviewCurrentUrl = '';
    }

    function scheduleLinkPreviewDismiss() {
        if (linkPreviewHideTimer) clearTimeout(linkPreviewHideTimer);
        linkPreviewHideTimer = setTimeout(dismissLinkPreview, 250);
    }

    function cancelLinkPreviewDismiss() {
        if (linkPreviewHideTimer) {
            clearTimeout(linkPreviewHideTimer);
            linkPreviewHideTimer = null;
        }
    }

    /* Resolve a link's verdict. Prefer the authoritative per-link score from
       the scan result (same engine the report uses); fall back to a structural
       check for links that haven't been scanned yet. Never claim "no threats"
       on an unscanned link. */
    function linkVerdictFor(url) {
        try {
            const r = scanResultFor(state.selectedIdx);
            const L = r && r.assessment && r.assessment.dimensions && r.assessment.dimensions.links;
            if (L && L.links) {
                for (let i = 0; i < L.links.length; i++) {
                    if (L.links[i].url === url) {
                        return { scored: true, level: L.links[i].level, reason: L.links[i].reason };
                    }
                }
            }
        } catch (e) { /* fall through to heuristic */ }
        let host = '';
        try { host = new URL(url.indexOf('://') >= 0 ? url : 'http://' + url).hostname.toLowerCase(); } catch (e) {}
        const tld = host.indexOf('.') >= 0 ? host.split('.').pop() : '';
        const configuredTlds = state.detectionRules.risky_tlds || [];
        const RISKY_TLD = configuredTlds.length
            ? Object.fromEntries(configuredTlds.map(t => [String(t).replace(/^\./, ''), 1]))
            : {test:1,tk:1,ml:1,ga:1,cf:1,gq:1,xyz:1,top:1,click:1,zip:1,mov:1,buzz:1,work:1,country:1,link:1,review:1,loan:1,men:1,date:1,fit:1,rest:1};
        if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) return { scored:false, level:'high', reason:'Uses a raw IP address instead of a domain name.' };
        if (url.indexOf('@') >= 0) return { scored:false, level:'high', reason:"Contains an '@' that hides the link's true destination." };
        if (host.indexOf('xn--') >= 0) return { scored:false, level:'high', reason:'Punycode — a possible look-alike of a real domain.' };
        if (RISKY_TLD[tld]) return { scored:false, level:'elevated', reason:'Unfamiliar domain on a high-risk top-level domain (.' + tld + ').' };
        return { scored:false, level:'unknown', reason:'' };
    }

    const _LP_WARN = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    const _LP_OK = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    const _LP_INFO = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';

    function renderLinkPreviewBody(url, revealed) {
        const isHttps = url.startsWith('https');
        const v = linkVerdictFor(url);
        const truncate = (s) => (s.length > 80 ? s.slice(0, 80) + '...' : s);
        const display = revealed ? truncate(url) : truncate(defangUrl(url));
        const urlClass = revealed ? 'link-preview-url' : 'link-preview-url link-preview-url-defanged';

        let cls, icon, label;
        if (v.level === 'high') { cls = 'link-preview-risky'; icon = _LP_WARN; label = 'High-risk destination'; }
        else if (v.level === 'elevated') { cls = 'link-preview-risky'; icon = _LP_WARN; label = 'Suspicious destination'; }
        else if (v.scored) { cls = 'link-preview-safe'; icon = _LP_OK; label = 'No threats identified'; }
        else { cls = 'link-preview-unknown'; icon = _LP_INFO; label = 'Not yet analyzed — run a scan'; }

        const verdict =
            '<div class="link-preview-verdict ' + cls + '">' + icon + ' ' + label +
            (!isHttps ? ' (unencrypted)' : '') + '</div>' +
            (v.reason ? '<div class="link-preview-reason">' + escapeHtml(v.reason) + '</div>' : '');
        const revealBtn = revealed
            ? '<div class="link-preview-hint">Real link revealed — do not click through.</div>'
            : '<button type="button" class="link-preview-reveal-btn" data-action="reveal">Reveal real URL</button>' +
              '<div class="link-preview-hint">This link is defanged for safety.</div>';
        return '<div class="' + urlClass + '">' + escapeHtml(display) + '</div>' + verdict + revealBtn;
    }

    function showLinkPreview(link) {
        const url = (link.getAttribute('data-url') || '').replace(/&amp;/g, '&');
        cancelLinkPreviewDismiss();
        if (linkPreviewEl && linkPreviewCurrentUrl === url) return;
        if (linkPreviewEl) linkPreviewEl.remove();

        linkPreviewCurrentUrl = url;
        linkPreviewEl = document.createElement('div');
        linkPreviewEl.className = 'link-preview-card';
        linkPreviewEl.innerHTML = renderLinkPreviewBody(url, false);
        document.body.appendChild(linkPreviewEl);

        const rect = link.getBoundingClientRect();
        linkPreviewEl.style.left = Math.min(rect.left, window.innerWidth - 340) + 'px';
        linkPreviewEl.style.top = (rect.bottom + 8) + 'px';

        linkPreviewEl.addEventListener('mouseenter', cancelLinkPreviewDismiss);
        linkPreviewEl.addEventListener('mouseleave', scheduleLinkPreviewDismiss);
        linkPreviewEl.addEventListener('click', (ev) => {
            const btn = ev.target.closest('[data-action="reveal"]');
            if (!btn) return;
            ev.preventDefault();
            ev.stopPropagation();
            linkPreviewEl.innerHTML = renderLinkPreviewBody(url, true);
        });
    }

    document.getElementById('previewBody').addEventListener('mouseover', (e) => {
        const link = e.target.closest('.body-link');
        if (!link) return;
        showLinkPreview(link);
    });
    document.getElementById('previewBody').addEventListener('mouseout', (e) => {
        const link = e.target.closest('.body-link');
        if (link) scheduleLinkPreviewDismiss();
    });
    /* Click-outside dismissal so the revealed-URL state can't linger after
       the user moves on. */
    document.addEventListener('mousedown', (e) => {
        if (!linkPreviewEl) return;
        if (linkPreviewEl.contains(e.target)) return;
        if (e.target.closest('.body-link')) return;
        dismissLinkPreview();
    });

    /* Email action buttons */
    document.getElementById('moveToJunkBtn').addEventListener('click', () => {
        if (state.selectedIdx !== null) moveToJunk(state.selectedIdx);
    });

    /* Top bar dark/light toggle */
    document.getElementById('modeToggleTop').addEventListener('change', (e) => {
        setMode(e.target.checked ? 'light' : 'dark');
    });

    /* Modal dark/light toggle */
    document.getElementById('modeToggleModal').addEventListener('change', (e) => {
        setMode(e.target.checked ? 'light' : 'dark');
    });

    /* Settings */
    /* Sidebar collapse toggle */
    const collapseBtn = document.getElementById('sidebarCollapseBtn');
    if (collapseBtn) {
        collapseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            const sidebar = document.getElementById('sidebar');
            const mainArea = document.querySelector('.main-area');
            const statusbar = document.querySelector('.statusbar');
            const collapsed = sidebar.classList.toggle('collapsed');
            if (mainArea) mainArea.classList.toggle('sidebar-collapsed', collapsed);
            if (statusbar) statusbar.classList.toggle('sidebar-collapsed', collapsed);
            localStorage.setItem('pg-sidebar-collapsed', collapsed ? '1' : '0');
        });
    }
    // Restore sidebar state
    if (localStorage.getItem('pg-sidebar-collapsed') === '1') {
        document.getElementById('sidebar').classList.add('collapsed');
        const mainArea = document.querySelector('.main-area');
        const statusbar = document.querySelector('.statusbar');
        if (mainArea) mainArea.classList.add('sidebar-collapsed');
        if (statusbar) statusbar.classList.add('sidebar-collapsed');
    }

    document.getElementById('settingsBtn').addEventListener('click', openSettings);
    document.getElementById('settingsClose').addEventListener('click', closeSettings);

    /* Close modal on overlay click */
    document.getElementById('settingsModal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('settingsModal')) {
            closeSettings();
        }
    });

    const providerRefreshBtn = document.getElementById('providerRefreshBtn');
    if (providerRefreshBtn) {
        providerRefreshBtn.addEventListener('click', refreshProviderSettings);
    }
    document.querySelectorAll('.provider-card[data-provider]').forEach(card => {
        const save = card.querySelector('[data-provider-save]');
        const remove = card.querySelector('[data-provider-remove]');
        const toggle = card.querySelector('[data-provider-enabled]');
        const input = card.querySelector('[data-provider-key]');
        if (save) save.addEventListener('click', () => saveProviderKey(card));
        if (remove) remove.addEventListener('click', () => removeProviderKey(card));
        if (toggle) {
            toggle.addEventListener('change', () => toggleProvider(card, toggle.checked));
        }
        if (input) {
            input.addEventListener('keydown', event => {
                if (event.key === 'Enter') saveProviderKey(card);
            });
        }
    });

    /* Theme swatches */
    document.querySelectorAll('.theme-swatch').forEach(sw => {
        sw.addEventListener('click', () => {
            setTheme(sw.getAttribute('data-theme'));
        });
    });

    /* Font size slider */
    document.getElementById('fontSlider').addEventListener('input', (e) => {
        setFontSize(parseInt(e.target.value, 10));
    });

    const densityToggle = document.getElementById('densityToggle');
    if (densityToggle) {
        densityToggle.addEventListener('change', (e) => {
            setListDensity(e.target.checked);
        });
    }

    const reconnectBannerBtn = document.getElementById('reconnectBannerBtn');
    if (reconnectBannerBtn) {
        reconnectBannerBtn.addEventListener('click', () => {
            const statusReconnect = document.getElementById('pgReconnectBtn');
            if (statusReconnect && statusReconnect.offsetParent !== null) {
                statusReconnect.click();
            } else {
                startExternalMicrosoftSignIn(null).catch(e => {
                    console.error('[PhishGuard] reconnect threw:', e);
                    setStatus('Reconnect failed - see console', 'error');
                });
            }
        });
    }

    /* Connect button in topbar — opens modal or disconnects */
    document.getElementById('connectBtn').addEventListener('click', handleConnectClick);

    /* Connect modal */
    document.getElementById('connectModalClose').addEventListener('click', closeConnectModal);
    document.getElementById('connectModalBtn').addEventListener('click', connectOutlook);
    document.getElementById('connectModal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('connectModal')) closeConnectModal();
    });
    document.getElementById('clientIdInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') connectOutlook();
    });

    /* Profile card toggle */
    document.getElementById('profileCard').addEventListener('click', (e) => {
        e.stopPropagation();
        toggleProfilePopup();
    });
    document.getElementById('logoutBtn').addEventListener('click', async () => {
        closeProfilePopup();
        /* Full sign-out — disconnects Outlook AND signs out of Supabase.
           The status-bar Sign-out button does the same thing. */
        await signOutOfPhishGuard();
    });
    document.getElementById('addAccountBtn').addEventListener('click', async () => {
        closeProfilePopup();
        /* "Add another account" reuses the system-browser OAuth flow.
           Cancelling the browser tab is a no-op — current session
           stays. */
        try {
            await startExternalMicrosoftSignIn(null);
        } catch (e) {
            console.error('[PhishGuard] add-account threw:', e);
            setStatus('Sign-in error — see console', 'error');
        }
    });
    /* Close popup when clicking outside */
    document.addEventListener('click', (e) => {
        const profile = document.getElementById('sidebarProfile');
        if (profile && profile.classList.contains('open') && !profile.contains(e.target)) {
            closeProfilePopup();
        }
    });

    /* Check auth status on load */
    checkAuthStatus();

    /* #6 KEYBOARD SHORTCUTS */
    document.addEventListener('keydown', (e) => {
        /* Don't trigger shortcuts when typing in inputs */
        const tag = (e.target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') {
            if (e.key === 'Escape') {
                closeSettings();
                closeConnectModal();
                closeProfilePopup();
            }
            return;
        }

        /* Shortcuts overlay */
        const shortcutsEl = document.getElementById('shortcutsOverlay');
        if (shortcutsEl.style.display !== 'none') {
            shortcutsEl.style.display = 'none';
            return;
        }

        if (e.key === 'Escape') {
            closeSettings();
            closeConnectModal();
            closeProfilePopup();
            closeContactCard();
            return;
        }

        if (e.key === '?' || (e.shiftKey && e.key === '/')) {
            e.preventDefault();
            shortcutsEl.style.display = 'flex';
            return;
        }

        const key = e.key.toLowerCase();

        /* Arrow navigation */
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            const total = state.emails.length;
            if (total === 0) return;
            let next = state.selectedIdx;
            if (next === null) {
                next = 0;
            } else if (e.key === 'ArrowDown') {
                next = Math.min(total - 1, next + 1);
            } else {
                next = Math.max(0, next - 1);
            }
            selectEmail(next);
            /* Scroll email into view */
            const row = document.querySelector('.email-row[data-idx="' + next + '"]');
            if (row) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            return;
        }

        if (key === 's' && state.selectedIdx !== null && !scanResultFor(state.selectedIdx)) {
            scanEmail(state.selectedIdx);
        } else if (key === 'r') {
            document.getElementById('refreshBtn').click();
        } else if (key === 'f' && state.selectedIdx !== null) {
            toggleStar(state.selectedIdx);
        } else if (key === 'j' && state.selectedIdx !== null) {
            if (document.getElementById('moveToJunkBtn').offsetParent !== null) {
                document.getElementById('moveToJunkBtn').click();
            } else {
                moveToJunk(state.selectedIdx);
            }
        }
    });

    /* #5 Dismiss splash — the fill animation now completes in ~1.3s;
       holding the user for 3.8s every launch was a paper-cut. */
    setTimeout(dismissSplash, 1700);

    /* ============== Supabase auth bring-up ============== */
    initSupabaseAuth();
});

/* ======================== SUPABASE AUTH (PhishGuard account) ======================== */

/* Sign-in modal is shown whenever there is no active Supabase session.
   Bring-up build uses email + password; Microsoft OAuth will replace it
   once the provider is configured on the Supabase dashboard.

   This is separate from the Outlook OAuth flow — Supabase identity is
   "who you are in PhishGuard", Outlook OAuth is "give the app permission
   to read your mailbox". For now the two are unrelated. */
function initSupabaseAuth() {
    if (!window.pg || !window.pg.auth) {
        console.error('[PhishGuard] window.pg.auth missing — supabase-client.js did not initialize.');
        return;
    }

    /* If this renderer instance is the OAuth callback popup, do nothing
       here. Hide the app chrome so we don't briefly flash the inbox /
       sign-in modal. The popup self-closes from supabase-client.js once
       the session lands in localStorage. */
    if (window.pg.isOAuthCallback) {
        document.body.style.background = 'var(--bg)';
        const splash = document.getElementById('splashScreen');
        if (splash) splash.style.display = 'none';
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.style.display = 'none';
        const main = document.querySelector('.main-area');
        if (main) main.style.display = 'none';
        const sb = document.getElementById('statusbar');
        if (sb) sb.style.display = 'none';
        // Tiny placeholder so the window isn't blank.
        const note = document.createElement('div');
        note.style.cssText = 'position:fixed;inset:0;display:flex;align-items:center;'
            + 'justify-content:center;font-family:var(--font-sans);'
            + 'color:var(--text2);font-size:13px';
        note.textContent = 'Completing sign-in…';
        document.body.appendChild(note);
        return;
    }

    const modal = document.getElementById('pgSignInModal');
    const statusEl = document.getElementById('pgSignInStatus');

    function showModal() {
        modal.style.display = 'flex';
    }
    function hideModal() {
        modal.style.display = 'none';
        if (statusEl) statusEl.textContent = '';
    }

    // Sign in with Microsoft - opens the URL in the user's DEFAULT
    // browser so they can leverage an existing Microsoft session (the
    // "click the signed-in account" UX). Flask serves a small callback
    // page that stores a short-lived PKCE code for this renderer to exchange.
    const msBtn = document.getElementById('pgSignInMicrosoftBtn');
    if (msBtn) {
        msBtn.addEventListener('click', async () => {
            statusEl.textContent = 'Starting Microsoft sign-in…';
            // We only need to lock the button until the OAuth URL is
            // actually opened in the browser. Once it's open, we want
            // the user to be able to click Sign-in again (e.g. they
            // closed the tab by mistake) — keeping the button disabled
            // for the full 5-minute poll window would strand them.
            msBtn.disabled = true;
            try {
                await startExternalMicrosoftSignIn(statusEl);
            } catch (e) {
                console.error('[PhishGuard] external sign-in threw:', e);
                statusEl.textContent = 'Sign-in error — see console';
            }
            // Re-enabled in startExternalMicrosoftSignIn itself once the
            // browser has been told to open. Defensive fallback here in
            // case that path didn't run.
            msBtn.disabled = false;
        });
    }

    // Sign-out button in the status bar
    const signOutBtn = document.getElementById('pgSignOutBtn');
    if (signOutBtn) {
        signOutBtn.addEventListener('click', async () => {
            signOutOfPhishGuard();
        });
    }

    // Reconnect-Outlook button — only shown when Supabase session is
    // active but Flask has no Outlook provider token. Provider tokens are
    // not restored from renderer storage, so the user must reconnect Outlook.
    const reconnectBtn = document.getElementById('pgReconnectBtn');
    if (reconnectBtn) {
        reconnectBtn.addEventListener('click', async () => {
            reconnectBtn.disabled = true;
            const prevLabel = reconnectBtn.lastChild.textContent;
            reconnectBtn.lastChild.textContent = ' Opening browser…';
            setStatus('Opening Microsoft sign-in in your browser…', 'scanning');
            try {
                const ok = await startExternalMicrosoftSignIn(null);
                if (ok) {
                    setStatus('Outlook connected', '');
                    await loadEmails();
                    await checkAuthStatus();
                }
            } catch (e) {
                console.warn('[PhishGuard] reconnect failed:', e);
                setStatus('Reconnect failed — see console', 'error');
            } finally {
                reconnectBtn.lastChild.textContent = prevLabel;
                reconnectBtn.disabled = false;
            }
        });
    }

    // React to auth state transitions (sign-in, sign-out, token refresh).
    window.pg.auth.onChange(async (event, session) => {
        console.log('[PhishGuard] auth event:', event,
                    session ? '(' + session.user.email + ')' : '(no session)');
        // TOKEN_REFRESHED on a session we already know about is just the
        // hourly access-token rotation.
        if (event === 'TOKEN_REFRESHED' && session && state.user
            && state.user.id === session.user.id) {
            state.supabaseJwt = session.access_token;
            return;
        }
        if (session) {
            await handleSupabaseSignIn(session);
        } else if (event === 'SIGNED_OUT') {
            state.user = null;
            state.supabaseJwt = null;
            state.starredEmails = {};
            state.savedAccounts = [];
            state.scanResults = {};
            state.stats = { scanned: 0, threats: 0, safe: 0, dangerous: 0, suspicious: 0, safeTier: 0 };
            try { localStorage.removeItem('pg-provider'); } catch (e) {} // legacy cleanup
            renderStats();
            renderEmailList();
            renderPopupAccounts('');
            hideSignedInIndicator();
            setReconnectBannerVisible(false);
            showModal();
        }
    });

    // Initial check — show modal if there's no session on boot. If
    // there IS a session, the SIGNED_IN/INITIAL_SESSION event from
    // onChange handles all the loading paths. We just set the
    // basic state + indicator here so they appear immediately.
    window.pg.auth.currentSession().then(async (session) => {
        if (!session) {
            showModal();
        } else {
            state.user = session.user;
            state.supabaseJwt = session.access_token;
            console.log('[PhishGuard] active session for ' + session.user.email);
            showSignedInIndicator(session.user.email);
        }
    });
}

/* Sign-in side-effects: forward Outlook token to Flask, hide the
   modal, load starred/accounts/scan-history, refresh auth status. This
   runs from both the supabase-js auth event AND directly from the
   external-browser poll handler after a successful setSession — we
   don't rely on supabase-js firing the right event because empirically
   it sometimes fires TOKEN_REFRESHED (or nothing at all) after
   setSession when an in-memory session existed before. Idempotent: if
   it's already run for this user_id in the last 2 seconds, the guard
   below short-circuits. */
async function handleSupabaseSignIn(session) {
    console.log('[PhishGuard] handleSupabaseSignIn: enter',
        session && session.user ? session.user.email : '(no user)');
    if (!session || !session.user) {
        console.warn('[PhishGuard] handleSupabaseSignIn: bailing — no session/user');
        return;
    }
    const now = Date.now();
    if (state._lastSignInUserId === session.user.id
        && state._lastSignInAt
        && (now - state._lastSignInAt) < 2000) {
        console.log('[PhishGuard] handleSupabaseSignIn: skipping duplicate within 2s');
        return;
    }
    state._lastSignInUserId = session.user.id;
    state._lastSignInAt = now;

    state.user = session.user;
    state.supabaseJwt = session.access_token;

    // Synchronous, fast UI updates — these never await on the network,
    // so we do them inline. Anything that hits the network is deferred
    // below into a non-awaited block, so this function returns quickly
    // and doesn't block supabase-js's event-dispatch loop (which uses
    // Promise.all over all onAuthStateChange listeners — if we hang
    // inside one, the whole auth subsystem stalls).
    const modal = document.getElementById('pgSignInModal');
    if (modal) modal.style.display = 'none';
    const signinStatus = document.getElementById('pgSignInStatus');
    if (signinStatus) signinStatus.textContent = '';
    showSignedInIndicator(session.user.email);

    let providerInfo = null;
    if (session.provider_token) {
        providerInfo = {
            provider_token: session.provider_token,
            provider_refresh_token: session.provider_refresh_token || '',
            expires_in: session.expires_in || 3600,
        };
    } else if (state.pendingProviderToken) {
        providerInfo = state.pendingProviderToken;
        state.pendingProviderToken = null;
    }
    console.log('[PhishGuard] handleSupabaseSignIn: providerInfo',
        providerInfo ? 'present' : 'NULL');

    // Fire-and-forget all the network work. If any step hangs, only
    // that step is stuck — the rest of the app stays responsive and
    // any future auth event can still be processed.
    (async () => {
        try {
            console.log('[PhishGuard] post-signin: CSRF refresh');
            await withTimeout(refreshCsrfToken(), 5000, 'refreshCsrfToken');
            if (providerInfo) {
                console.log('[PhishGuard] post-signin: forwarding provider');
                await withTimeout(
                    forwardProviderInfoToFlask(providerInfo, session.user),
                    15000, 'forwardProviderInfoToFlask');
                console.log('[PhishGuard] post-signin: provider forwarded');
            } else {
                setStatus('Click "Connect Outlook" in the bottom-right to load your inbox', '');
            }
        } catch (e) {
            console.warn('[PhishGuard] post-signin provider step failed:', e);
            setStatus('Sign-in step failed: ' + (e && e.message ? e.message : e), 'error');
        }
        // Each loader is independently timed-out + caught so one hung
        // call doesn't strand the others.
        try { await withTimeout(loadStarredFromSupabase(), 8000, 'loadStarred'); }
        catch (e) { console.warn('[PhishGuard] loadStarred failed:', e); }
        try { await withTimeout(loadAccountsFromSupabase(), 8000, 'loadAccounts'); }
        catch (e) { console.warn('[PhishGuard] loadAccounts failed:', e); }
        try { await withTimeout(loadScanHistoryFromSupabase(), 8000, 'loadScanHistory'); }
        catch (e) { console.warn('[PhishGuard] loadScanHistory failed:', e); }
        try { await withTimeout(checkAuthStatus(), 5000, 'checkAuthStatus'); }
        catch (e) { console.warn('[PhishGuard] checkAuthStatus failed:', e); }
        console.log('[PhishGuard] post-signin: all steps attempted');
    })();
}

// Race a promise against a timeout. Rejects with a labeled error if
// the promise doesn't settle in time. Used to keep one stuck network
// call from blocking the entire post-signin chain.
function withTimeout(promise, ms, label) {
    return new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(label + ' timed out after ' + ms + 'ms')), ms);
        Promise.resolve(promise).then(
            (v) => { clearTimeout(t); resolve(v); },
            (e) => { clearTimeout(t); reject(e); },
        );
    });
}

// Surface uncaught renderer errors in the status bar so we don't need
// DevTools open to know something blew up after sign-in.
window.addEventListener('error', function (ev) {
    console.error('[PhishGuard] window.error:', ev.message, ev.filename, ev.lineno);
    try {
        setStatus('JS error: ' + ev.message, 'error');
    } catch (e) {}
});
window.addEventListener('unhandledrejection', function (ev) {
    const r = ev.reason;
    const msg = (r && r.message) ? r.message : String(r);
    console.error('[PhishGuard] unhandledrejection:', msg, r);
    try {
        setStatus('Async error: ' + msg, 'error');
    } catch (e) {}
});

function showSignedInIndicator(email) {
    const wrap = document.getElementById('pgSignedIn');
    const span = document.getElementById('pgSignedInEmail');
    const btn = document.getElementById('pgSignOutBtn');
    if (wrap && span) {
        span.textContent = email || '';
        wrap.style.display = 'flex';
    }
    if (btn) btn.style.display = 'flex';
}

function hideSignedInIndicator() {
    const wrap = document.getElementById('pgSignedIn');
    const btn = document.getElementById('pgSignOutBtn');
    const reconnect = document.getElementById('pgReconnectBtn');
    if (wrap) wrap.style.display = 'none';
    if (btn) btn.style.display = 'none';
    if (reconnect) reconnect.style.display = 'none';
}

/* External-browser Microsoft sign-in.
   Flow:
     1. Generate a random nonce; register it with Flask.
     2. Build a Supabase OAuth URL whose redirectTo points back at
        Flask's /auth/external-callback with the nonce embedded.
     3. Ask main to open that URL in the user's default browser
        (window.electron.openExternal).
     4. The user signs in in their browser (where they're already
        signed into Microsoft - that's the whole point). Microsoft
        redirects to Supabase, then Supabase redirects to Flask's
        callback page with a short-lived PKCE code.
     5. Meanwhile this renderer polls /api/auth/external-poll until
        the code lands, then exchanges it with supabase.auth.
     6. The standard onAuthStateChange listener handles the rest of
        the sign-in flow (load starred, forward provider token, etc.). */
async function startExternalMicrosoftSignIn(statusEl) {
    if (!window.pg || !window.pg.auth || !window.pg.auth.signInWithMicrosoft) {
        if (statusEl) statusEl.textContent = 'Supabase client not ready';
        return false;
    }
    const nonce = pgGenerateNonce();
    // Register the nonce so /auth/external-callback will accept it.
    let start;
    try {
        start = await apiFetch('/api/auth/external-start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nonce }),
        });
    } catch (e) {
        if (statusEl) statusEl.textContent = 'Could not register sign-in nonce';
        throw e;
    }
    if (!start.ok) {
        if (statusEl) statusEl.textContent = 'Sign-in start refused (' + start.status + ')';
        return false;
    }

    const redirectTo = window.location.origin
        + '/auth/external-callback?nonce=' + encodeURIComponent(nonce);
    const { data, error } = await window.pg.auth.signInWithMicrosoft({ redirectTo });
    if (error || !data || !data.url) {
        if (statusEl) statusEl.textContent = (error && error.message) || 'Could not build sign-in URL';
        return false;
    }

    // Open in the system default browser. The user's browser has their
    // Microsoft cookies, so they get the "click your signed-in account"
    // experience without a credentials prompt.
    let opened = { ok: false };
    if (window.electron && window.electron.openExternal) {
        opened = await window.electron.openExternal(data.url);
    }
    if (!opened || !opened.ok) {
        if (statusEl) statusEl.textContent =
            'Could not open browser' + (opened && opened.reason ? ': ' + opened.reason : '');
        return false;
    }

    if (statusEl) statusEl.textContent =
        'Complete sign-in in your browser. Waiting for confirmation…';

    // Browser is open — let the user click Sign-in again if needed
    // (they might close the tab and want to retry). The poll continues
    // in the background.
    const msBtn = document.getElementById('pgSignInMicrosoftBtn');
    if (msBtn) msBtn.disabled = false;

    return pgPollExternalSession(nonce, statusEl);
}

function pgGenerateNonce() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID().replace(/-/g, '');
    }
    return (Math.random().toString(36).slice(2) + Date.now().toString(36)
            + Math.random().toString(36).slice(2));
}

async function pgPollExternalSession(nonce, statusEl) {
    const MAX_POLLS = 150; // ~5 minutes at 2s each
    const INTERVAL_MS = 2000;
    console.log('[PhishGuard] external poll starting for nonce', nonce.slice(0, 8));
    for (let i = 0; i < MAX_POLLS; i++) {
        await new Promise((r) => setTimeout(r, INTERVAL_MS));
        let resp;
        try {
            resp = await fetch('/api/auth/external-poll?nonce='
                + encodeURIComponent(nonce));
        } catch (e) {
            continue; // transient — keep polling
        }
        const data = await resp.json().catch(() => ({}));
        if (data.status === 'pending') continue;
        if (data.status === 'expired') {
            if (statusEl) statusEl.textContent =
                'Sign-in nonce expired. Try again.';
            return false;
        }
        if (data.status === 'ready' && data.access_token) {
            try {
                const { data: sessData, error } =
                    await window.pg.supabase.auth.setSession({
                        access_token: data.access_token,
                        refresh_token: data.refresh_token || '',
                    });
                if (error) {
                    state.pendingProviderToken = null;
                    console.warn('[PhishGuard] external token install failed:', error);
                    if (statusEl) statusEl.textContent = 'Session install failed';
                    return false;
                }
                const newSession = sessData && sessData.session;
                if (!newSession) {
                    state.pendingProviderToken = null;
                    if (statusEl) statusEl.textContent = 'No session returned';
                    return false;
                }
                if (data.provider_token) {
                    state.pendingProviderToken = {
                        provider_token: data.provider_token,
                        provider_refresh_token: data.provider_refresh_token || '',
                        expires_in: data.expires_in || newSession.expires_in || 3600,
                    };
                }
                console.log('[PhishGuard] external token install OK; running handleSupabaseSignIn directly');
                await handleSupabaseSignIn(newSession);
                return true;
            } catch (e) {
                state.pendingProviderToken = null;
                console.error('[PhishGuard] external token install threw:', e);
                if (statusEl) statusEl.textContent = 'Session install error';
                return false;
            }
        }
        if (data.status === 'ready' && data.code) {
            try {
                const { data: sessData, error } =
                    await window.pg.supabase.auth.exchangeCodeForSession(data.code);
                if (error) {
                    state.pendingProviderToken = null;
                    console.warn('[PhishGuard] external code exchange failed:', error);
                    if (statusEl) statusEl.textContent = 'Session exchange failed';
                    return false;
                }
                const newSession = sessData && sessData.session;
                if (!newSession) {
                    state.pendingProviderToken = null;
                    if (statusEl) statusEl.textContent = 'No session returned';
                    return false;
                }
                // Some supabase-js versions expose provider tokens at the
                // response root rather than on the session. Keep them in
                // memory only long enough for handleSupabaseSignIn to forward.
                if (!newSession.provider_token && sessData.provider_token) {
                    state.pendingProviderToken = {
                        provider_token: sessData.provider_token,
                        provider_refresh_token: sessData.provider_refresh_token || '',
                        expires_in: newSession.expires_in || 3600,
                    };
                }
                // Don't rely on supabase-js to fire SIGNED_IN - empirically
                // it fires TOKEN_REFRESHED (or nothing at all) when the
                // in-memory client still has a session from before. Drive
                // the sign-in side-effects directly.
                console.log('[PhishGuard] code exchange OK; running handleSupabaseSignIn directly');
                await handleSupabaseSignIn(newSession);
                return true;
            } catch (e) {
                state.pendingProviderToken = null;
                console.error('[PhishGuard] code exchange threw:', e);
                if (statusEl) statusEl.textContent = 'Session exchange error';
                return false;
            }
        }
    }
    if (statusEl) statusEl.textContent =
        'Sign-in timed out. Close the browser tab and click Sign in again.';
    return false;
}

/* Sign out of PhishGuard. We clear the local Supabase session storage
   synchronously (no await) so a quick "sign-out then sign-in again"
   sequence can't race: an in-flight supabase.auth.signOut() finishing
   AFTER setSession would otherwise fire SIGNED_OUT and clobber the
   newly installed session, leaving the user stranded on the login
   modal even though the browser said "signed in". Outlook disconnect
   on Flask is still done over the network, but that's fire-and-forget
   and doesn't touch Supabase state. */
function signOutOfPhishGuard() {
    console.log('[PhishGuard] sign-out clicked');
    setStatus('Signing out…', '');

    // 1. Reset the renderer state immediately.
    state.user = null;
    state.supabaseJwt = null;
    state.starredEmails = {};
    state.savedAccounts = [];
    state.scanResults = {};
    state.stats = { scanned: 0, threats: 0, safe: 0, dangerous: 0, suspicious: 0, safeTier: 0 };
    state.emails = [];
    state.pendingProviderToken = null;
    // Clear the idempotency guard so the next sign-in's handler runs
    // even if it's the same user_id as before.
    state._lastSignInUserId = null;
    state._lastSignInAt = 0;
    setReconnectBannerVisible(false);

    // 2. Clear localStorage entries: legacy pg-provider plus every
    //    Supabase auth-token key (`sb-<projectRef>-auth-token`). Doing
    //    this directly — instead of calling supabase.auth.signOut() —
    //    avoids the race described in the docstring.
    try { localStorage.removeItem('pg-provider'); } catch (e) {} // legacy cleanup
    try {
        for (let i = localStorage.length - 1; i >= 0; i--) {
            const key = localStorage.key(i);
            if (key && key.startsWith('sb-') && key.endsWith('-auth-token')) {
                localStorage.removeItem(key);
            }
        }
    } catch (e) { /* localStorage might be disabled — ignore */ }

    renderStats();
    renderEmailList();
    renderPopupAccounts('');
    hideSignedInIndicator();
    const modal = document.getElementById('pgSignInModal');
    if (modal) modal.style.display = 'flex';
    const msBtn = document.getElementById('pgSignInMicrosoftBtn');
    if (msBtn) msBtn.disabled = false;
    const signinStatusEl = document.getElementById('pgSignInStatus');
    if (signinStatusEl) signinStatusEl.textContent = '';
    setStatus('Signed out', '');

    // 3. Fire-and-forget the server-side Outlook disconnect. Failure
    //    here doesn't block the user from re-signing-in.
    (async () => {
        try {
            const r = await fetch('/api/auth/status');
            const data = await r.json();
            if (data.connected) {
                try {
                    await apiFetch('/api/auth/disconnect', { method: 'POST' });
                    console.log('[PhishGuard] Outlook session disconnected');
                } catch (e) {
                    console.warn('[PhishGuard] Outlook disconnect failed:', e);
                }
            }
        } catch (e) {
            console.warn('[PhishGuard] status check failed:', e);
        }
    })();
}

async function loadStarredFromSupabase() {
    try {
        const list = await window.pg.starred.list();
        state.starredEmails = {};
        for (const id of list) state.starredEmails[id] = true;
        renderEmailList();
        console.log('[PhishGuard] loaded ' + list.length + ' starred email(s) from Supabase');
    } catch (e) {
        console.warn('[PhishGuard] could not load starred from Supabase:', e);
    }
}

async function loadAccountsFromSupabase() {
    try {
        const list = await window.pg.accounts.list();
        state.savedAccounts = list;
        console.log('[PhishGuard] loaded ' + list.length + ' Outlook account(s) from Supabase');
    } catch (e) {
        console.warn('[PhishGuard] could not load accounts from Supabase:', e);
    }
}

/* Hydrate state.scanResults from the scan_history table so previously-
   scanned emails come back marked across sessions. The table is
   append-only (multiple rows per message_id over time), so we sort
   newest-first and only keep the most recent verdict per message.
   The detailed url_analysis / header_result aren't persisted to
   scan_history — those stay as one-shot in-session data — so restored
   entries have null for those fields. The renderer's list view only
   needs prediction + confidence, so the score rings show correctly. */
async function loadScanHistoryFromSupabase() {
    if (!window.pg || !window.pg.supabase) return;
    try {
        const { data, error } = await window.pg.supabase
            .from('scan_history')
            .select('message_id, prediction, confidence, signals, scanned_at')
            .order('scanned_at', { ascending: false })
            .limit(1000);
        if (error) throw error;
        const rawCount = (data || []).length;
        const restored = {};
        for (const row of data || []) {
            if (restored[row.message_id]) continue; // older duplicate
            const sigs = row.signals || {};
            // The full assessment is persisted in signals (since this update),
            // so restored scans show the complete report without rescanning.
            // Older rows have no assessment → fall back to verdict-only.
            const assessment = sigs.assessment || null;
            const riskScore = (typeof sigs.risk_score === 'number')
                ? sigs.risk_score
                : (assessment && typeof assessment.overall === 'number' ? assessment.overall : undefined);
            restored[row.message_id] = {
                id: row.message_id,
                messageId: row.message_id,
                prediction: row.prediction,
                confidence: row.confidence,
                risk_score: riskScore,
                scanned_at: row.scanned_at || null,
                assessment: assessment,
                url_analysis: null,
                header_result: sigs.header_result || null,
                threat_intel: {
                    confirmed: Boolean(sigs.confirmed),
                    signals: Array.isArray(sigs.intel_signals) ? sigs.intel_signals : [],
                    checks_run: Array.isArray(sigs.checks_run) ? sigs.checks_run : [],
                },
                restored: true,
            };
        }
        // Merge — in-session results win over restored ones.
        let merged = 0;
        for (const id in restored) {
            if (!state.scanResults[id]) {
                state.scanResults[id] = restored[id];
                merged++;
            }
        }
        updateStatsFromResults();
        renderEmailList();
        const uniq = Object.keys(restored).length;
        console.log('[PhishGuard] scan_history: ' + rawCount + ' raw rows, '
                    + uniq + ' unique messages, ' + merged + ' merged into state');
        // Surface the result so the user can see whether anything came
        // back — failure to restore is otherwise silent.
        if (uniq > 0) {
            setStatus('Restored ' + uniq + ' previous scan' + (uniq === 1 ? '' : 's'), '');
        } else if (rawCount === 0) {
            setStatus('No previous scans found for this account', '');
        }
    } catch (e) {
        console.warn('[PhishGuard] could not load scan history:', e);
        setStatus('Could not load scan history — see console', 'error');
    }
}

/* Forward Outlook provider tokens to Flask so it can call Graph on the
   user's behalf. Tokens are never persisted in renderer storage; they are
   held only in memory long enough to hand them to Flask SessionState.
   Reloads /api/emails after the handoff so the renderer picks up the
   real Outlook inbox. */
async function forwardProviderInfoToFlask(info, user) {
    if (!info || !info.provider_token) return;
    try {
        const meta = (user && user.user_metadata) || {};
        const name = meta.full_name || meta.name || (user && user.email) || '';
        await apiFetch('/api/auth/supabase-provider', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider_token: info.provider_token,
                provider_refresh_token: info.provider_refresh_token || '',
                expires_in: info.expires_in || 3600,
                user: { name, email: (user && user.email) || '' },
            }),
        });
        console.log('[PhishGuard] forwarded Outlook token to Flask');
        await loadEmails();
    } catch (e) {
        console.warn('[PhishGuard] could not forward provider token:', e);
    }
}
