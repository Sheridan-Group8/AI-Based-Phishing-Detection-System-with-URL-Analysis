/* ================================================================
   PhishGuard Web Dashboard — Complete SPA JavaScript
   ================================================================ */

/* ======================== STATE ======================== */
const state = {
    emails: [],
    selectedIdx: null,
    scanResults: {},
    stats: { scanned: 0, threats: 0, safe: 0 },
    navView: 'inbox',
    theme: 'cupertino',
    mode: 'dark',
    fontSize: 11,
    inlineVisible: false,
};

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
    'Tip: Select an email from the list and click "Scan for Phishing" to analyze it.',
    'Tip: Hover over sender avatars to check the full email address.',
    'Tip: Look for SPF, DKIM, and DMARC passes to verify sender authenticity.',
    'Tip: Be cautious of emails that create a sense of urgency.',
    'Tip: Never click links in suspicious emails. Verify the URL first.',
    'Tip: Check for mismatched sender names and email addresses.',
    'Tip: Legitimate companies rarely ask for passwords via email.',
    'Tip: Watch for spelling and grammar mistakes in professional emails.'
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

function getUrlList(result) {
    /* Normalize url_analysis from backend into an array of {url, features, risk} objects */
    const ua = result.url_analysis;
    if (!ua) return [];
    if (Array.isArray(ua)) return ua;  /* already an array */
    const urls = ua.urls_found || [];
    const features = ua.features || {};
    const prob = ua.url_model_prob;
    return urls.map(url => {
        const feats = features[url] || {};
        /* Count how many risk features are flagged */
        const riskFlags = ['has_ip_address', 'suspicious_tld', 'is_shortened',
            'has_brand_in_subdomain', 'has_suspicious_keyword', 'has_port',
            'double_slash_redirect', 'has_at_symbol'];
        let riskyCount = 0;
        for (const f of riskFlags) {
            if (feats[f]) riskyCount++;
        }
        const isHttps = feats.is_https || false;
        let risk = 'low';
        if (prob !== undefined && prob > 0.7) risk = 'high';
        else if (riskyCount >= 2) risk = 'high';
        else if (riskyCount === 1 || !isHttps) risk = 'medium';
        return { url, features: feats, risk, is_phishing: risk === 'high' };
    });
}

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
    try {
        const res = await fetch('/api/emails');
        if (!res.ok) throw new Error('Failed to load emails');
        const data = await res.json();
        state.emails = data.emails || [];
        state.scanResults = {};
        state.selectedIdx = null;
        updateStatsFromResults();
        renderEmailList();
        document.getElementById('emailPreview').style.display = 'none';
        document.getElementById('dashboardView').style.display = '';
        setStatus('Ready — ' + state.emails.length + ' emails loaded', '');
    } catch (err) {
        console.error('loadEmails error:', err);
        setStatus('Error loading emails', 'error');
    }
}

async function scanEmail(idx) {
    setStatus('Scanning email...', 'scanning');
    try {
        const res = await fetch('/api/scan/' + idx, { method: 'POST' });
        const data = await res.json();
        if (data.error) {
            console.error('Scan error:', data.error);
            setStatus('Scan failed: ' + data.error, 'error');
            return;
        }
        state.scanResults[idx] = data;

        updateStatsFromResults();
        renderEmailList();
        if (state.selectedIdx === idx) {
            showScanResults(idx);
        }
        setStatus('Scan complete', '');
    } catch (err) {
        console.error('scanEmail error:', err);
        setStatus('Scan failed', 'error');
    }
}

async function scanAll() {
    setStatus('Scanning all emails...', 'scanning');
    try {
        const res = await fetch('/api/scan-all', { method: 'POST' });
        if (!res.ok) throw new Error('Scan all failed');
        const data = await res.json();

        if (data.results) {
            for (const key in data.results) {
                state.scanResults[key] = data.results[key];
            }
        }

        updateStatsFromResults();
        renderEmailList();
        if (state.selectedIdx !== null && state.scanResults[state.selectedIdx]) {
            showScanResults(state.selectedIdx);
        }
        setStatus('All emails scanned', '');
    } catch (err) {
        console.error('scanAll error:', err);
        setStatus('Scan all failed', 'error');
    }
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
    for (const key in state.scanResults) {
        scanned++;
        const r = state.scanResults[key];
        if (r.prediction === 1) {
            threats++;
        } else {
            safe++;
        }
    }
    state.stats = { scanned, threats, safe };
    renderStats();
}

function renderStats() {
    document.getElementById('statScannedNum').textContent = state.stats.scanned;
    document.getElementById('statThreatsNum').textContent = state.stats.threats;
    document.getElementById('statSafeNum').textContent = state.stats.safe;

    document.getElementById('dashScanned').textContent = state.stats.scanned;
    document.getElementById('dashThreats').textContent = state.stats.threats;
    document.getElementById('dashSafe').textContent = state.stats.safe;
}

/* ======================== EMAIL LIST RENDERING ======================== */

function renderEmailList() {
    const container = document.getElementById('emailList');
    const searchVal = document.getElementById('searchInput').value.toLowerCase().trim();

    let filtered = state.emails.map((email, idx) => ({ email, idx }));

    if (searchVal) {
        filtered = filtered.filter(({ email }) => {
            const subject = (email.subject || '').toLowerCase();
            const sender = (email.sender_name || email.sender || '').toLowerCase();
            return subject.includes(searchVal) || sender.includes(searchVal);
        });
    }

    if (state.navView === 'threats') {
        filtered = filtered.filter(({ idx }) => {
            const r = state.scanResults[idx];
            return r && (r.prediction === 1);
        });
    } else if (state.navView === 'safe') {
        filtered = filtered.filter(({ idx }) => {
            const r = state.scanResults[idx];
            return r && (r.prediction === 0);
        });
    }

    if (filtered.length === 0) {
        container.innerHTML = '<div style="padding:30px 16px;text-align:center;color:var(--text3);font-size:12px;">' +
            (searchVal ? 'No emails match your search.' :
             state.navView === 'threats' ? 'No threats detected.' :
             state.navView === 'safe' ? 'No safe emails yet.' :
             'No emails loaded.') +
            '</div>';
        return;
    }

    let html = '';
    for (const { email, idx } of filtered) {
        const senderName = email.sender_name || email.sender || 'Unknown';
        const senderAddr = email.sender || '';
        const subject = email.subject || '(no subject)';
        const preview = truncate(email.bodyPreview || '', 42);
        const date = formatDate(email.date);
        const isUnread = email.isRead === false;
        const isSelected = state.selectedIdx === idx;
        const initials = getInitials(senderName);
        const avatarColor = getAvatarColor(senderAddr);
        const scanResult = state.scanResults[idx];

        let badgeHtml = '';
        if (scanResult) {
            const isPhishing = scanResult.prediction === 1;
            badgeHtml = '<div class="email-row-badges">' +
                '<span class="scan-badge ' + (isPhishing ? 'scan-badge-danger' : 'scan-badge-safe') + '">' +
                (isPhishing ? 'Threat' : 'Safe') +
                '</span></div>';
        }

        html += '<div class="email-row' +
            (isUnread ? ' unread' : '') +
            (isSelected ? ' selected' : '') +
            '" data-idx="' + idx + '">' +
            '<div class="email-avatar" style="background:' + avatarColor + ';">' + escapeHtml(initials) + '</div>' +
            '<div class="email-row-content">' +
            '<div class="email-row-top">' +
            '<span class="email-row-sender">' + escapeHtml(truncate(senderName, 24)) + '</span>' +
            '<span class="email-row-date">' + escapeHtml(date) + '</span>' +
            '</div>' +
            '<div class="email-row-subject">' + escapeHtml(truncate(subject, 40)) + '</div>' +
            '<div class="email-row-preview">' + escapeHtml(preview) + '</div>' +
            badgeHtml +
            '</div>' +
            '</div>';
    }

    container.innerHTML = html;

    container.querySelectorAll('.email-row').forEach(row => {
        row.addEventListener('click', () => {
            const idx = parseInt(row.getAttribute('data-idx'), 10);
            selectEmail(idx);
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
    document.getElementById('emailPreview').style.display = 'flex';

    document.getElementById('previewSubject').textContent = email.subject || '(no subject)';

    const senderName = email.sender_name || email.sender || 'Unknown';
    const senderAddr = email.sender || '';
    const initials = getInitials(senderName);
    const avatarColor = getAvatarColor(senderAddr);

    const avatarEl = document.getElementById('previewAvatar');
    avatarEl.textContent = initials;
    avatarEl.style.background = avatarColor;

    document.getElementById('previewSender').textContent = senderName;
    document.getElementById('previewEmail').textContent = senderAddr;
    document.getElementById('previewDate').textContent = formatDate(email.date);

    renderAttachments(email);
    renderReputation(email);
    renderHeaders(email);

    const bodyText = email.body || '';
    const bodyEl = document.getElementById('previewBody');
    bodyEl.innerHTML = escapeHtml(bodyText).replace(/\n/g, '<br>');

    const scanBtn = document.getElementById('scanBtn');
    const scanResults = document.getElementById('scanResults');
    scanResults.style.display = 'none';
    state.inlineVisible = false;

    if (state.scanResults[idx]) {
        scanBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/></svg> Show Report';
    } else {
        scanBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/></svg> Scan for Phishing';
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
    let html = '';
    for (const att of attachments) {
        const name = typeof att === 'string' ? att : (att.name || att.filename || 'file');
        const risky = isRiskyExtension(name);
        html += '<span class="attachment-badge' + (risky ? ' risky' : '') + '">' +
            '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>' +
            (risky ? '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>' : '') +
            escapeHtml(name) +
            '</span>';
    }
    list.innerHTML = html;
}

async function renderReputation(email) {
    const panel = document.getElementById('reputationPanel');
    const senderAddr = email.sender || '';
    const domain = extractDomain(senderAddr);

    if (!domain) {
        panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';
    document.getElementById('repDomain').textContent = domain;
    document.getElementById('repGaugeValue').textContent = '--';
    document.getElementById('repVerdict').textContent = 'Checking...';
    document.getElementById('repBadge').textContent = '';
    document.getElementById('repBadge').className = 'rep-badge';

    setGaugeArc(0, 'var(--border)');

    const repData = await getReputation(domain);
    if (repData && repData.score !== undefined) {
        const score = Math.round(repData.score);
        document.getElementById('repGaugeValue').textContent = score;

        let color, verdict, badgeClass, badgeText;
        if (score >= 70) {
            color = 'var(--success)';
            verdict = 'This domain has a good reputation.';
            badgeClass = 'rep-badge rep-badge-trusted';
            badgeText = 'Trusted';
        } else if (score >= 40) {
            color = 'var(--warning)';
            verdict = 'This domain has a mixed reputation.';
            badgeClass = 'rep-badge rep-badge-caution';
            badgeText = 'Caution';
        } else {
            color = 'var(--danger)';
            verdict = 'This domain has a poor reputation.';
            badgeClass = 'rep-badge rep-badge-suspicious';
            badgeText = 'Suspicious';
        }

        setGaugeArc(score, color);
        document.getElementById('repVerdict').textContent = verdict;
        const badgeEl = document.getElementById('repBadge');
        badgeEl.className = badgeClass;
        badgeEl.textContent = badgeText;
    } else {
        document.getElementById('repGaugeValue').textContent = '?';
        document.getElementById('repVerdict').textContent = 'Reputation data unavailable.';
        const badgeEl = document.getElementById('repBadge');
        badgeEl.className = 'rep-badge rep-badge-caution';
        badgeEl.textContent = 'Unknown';
    }
}

function setGaugeArc(percent, color) {
    const arc = document.getElementById('repGaugeArc');
    const pct = Math.max(0, Math.min(100, percent));
    arc.style.background = 'conic-gradient(' +
        color + ' 0% ' + pct + '%, ' +
        'var(--border) ' + pct + '% 100%)';
}

function renderHeaders(email) {
    const viewer = document.getElementById('headerViewer');
    const details = document.getElementById('headerDetails');
    const chevron = document.getElementById('headerChevron');
    const headers = email.headers || null;

    if (!headers) {
        viewer.style.display = 'none';
        return;
    }

    viewer.style.display = 'block';
    details.style.display = 'none';
    chevron.classList.remove('open');

    setHeaderValue('headerSPFVal', headers.spf || 'none');
    setHeaderValue('headerDKIMVal', headers.dkim || 'none');
    setHeaderValue('headerDMARCVal', headers.dmarc || 'none');
}

function setHeaderValue(elementId, value) {
    const el = document.getElementById(elementId);
    const val = (value || 'none').toLowerCase();
    el.textContent = val.charAt(0).toUpperCase() + val.slice(1);
    el.className = 'header-value';
    if (val === 'pass') {
        el.classList.add('header-value-pass');
    } else if (val === 'fail') {
        el.classList.add('header-value-fail');
    } else {
        el.classList.add('header-value-none');
    }
}

/* ======================== SCAN RESULTS DISPLAY ======================== */

function showScanResults(idx) {
    const result = state.scanResults[idx];
    if (!result) return;

    const email = state.emails[idx];
    const isPhishing = result.prediction === 1;
    const confidence = result.confidence !== undefined ? Math.round(result.confidence * 100) : null;
    const bodyText = (email.body || '').toLowerCase();

    renderRiskOverview(result, email);
    renderVerdictCard(isPhishing, confidence);
    renderWhyFlagged(isPhishing, result, email);
    renderLinkSafety(result);
    renderSenderVerification(email, result);
    renderRecommendations(isPhishing, result, email);

    document.getElementById('scanResults').style.display = 'block';
    state.inlineVisible = true;

    const scanBtn = document.getElementById('scanBtn');
    scanBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18"/><path d="M6 6l12 12"/></svg> Hide Report';
}

function renderRiskOverview(result, email) {
    const bodyText = (email.body || '').toLowerCase();
    const isPhishing = result.prediction === 1;

    let contentScore = 0;
    let linksScore = 0;
    let senderScore = 0;

    let urgencyCount = 0;
    for (const word of URGENCY_WORDS) {
        if (bodyText.includes(word)) urgencyCount++;
    }
    let moneyCount = 0;
    for (const word of MONEY_WORDS) {
        if (bodyText.includes(word)) moneyCount++;
    }
    let threatCount = 0;
    for (const word of THREAT_WORDS) {
        if (bodyText.includes(word)) threatCount++;
    }

    contentScore = Math.min(100, (urgencyCount * 20) + (moneyCount * 15) + (threatCount * 20));

    const urls = getUrlList(result);
    if (urls.length > 0) {
        let riskyUrls = 0;
        for (const u of urls) {
            if (u.risk === 'high' || u.risk === 'dangerous' || u.prediction === 'phishing' || u.is_phishing) {
                riskyUrls++;
            }
        }
        linksScore = urls.length > 0 ? Math.round((riskyUrls / urls.length) * 100) : 0;
    } else {
        linksScore = isPhishing ? 30 : 5;
    }

    const headers = email.headers || {};
    const spf = (headers.spf || 'none').toLowerCase();
    const dkim = (headers.dkim || 'none').toLowerCase();
    const dmarc = (headers.dmarc || 'none').toLowerCase();
    let authFails = 0;
    if (spf === 'fail') authFails++;
    if (dkim === 'fail') authFails++;
    if (dmarc === 'fail') authFails++;
    let authNone = 0;
    if (spf === 'none') authNone++;
    if (dkim === 'none') authNone++;
    if (dmarc === 'none') authNone++;
    senderScore = Math.min(100, (authFails * 33) + (authNone * 10));
    if (isPhishing && senderScore < 30) senderScore = 30;

    setRiskBar('riskContent', 'riskContentPct', contentScore);
    setRiskBar('riskLinks', 'riskLinksPct', linksScore);
    setRiskBar('riskSender', 'riskSenderPct', senderScore);
}

function setRiskBar(barId, pctId, value) {
    const bar = document.getElementById(barId);
    const pct = document.getElementById(pctId);
    const v = Math.max(0, Math.min(100, Math.round(value)));
    bar.style.width = v + '%';
    bar.className = 'risk-bar-fill';
    if (v <= 33) bar.classList.add('low');
    else if (v <= 66) bar.classList.add('medium');
    else bar.classList.add('high');
    pct.textContent = v + '%';
}

function renderVerdictCard(isPhishing, confidence) {
    const card = document.getElementById('verdictCard');
    card.className = 'verdict-card ' + (isPhishing ? 'verdict-card-danger' : 'verdict-card-safe');

    document.getElementById('verdictIcon').textContent = isPhishing ? '\u26A0\uFE0F' : '\u2705';
    document.getElementById('verdictText').textContent = isPhishing ? 'Phishing Detected' : 'Email Looks Safe';

    const confEl = document.getElementById('verdictConfidence');
    if (confidence !== null) {
        confEl.textContent = confidence + '% confidence';
    } else {
        confEl.textContent = '';
    }
}

function renderWhyFlagged(isPhishing, result, email) {
    const titleEl = document.getElementById('whyFlaggedTitle');
    const listEl = document.getElementById('reasonList');
    const bodyText = (email.body || '').toLowerCase();

    titleEl.textContent = isPhishing ? 'Why Was This Flagged?' : 'Why Does This Look Safe?';

    const reasons = [];

    if (isPhishing) {
        let foundUrgency = [];
        for (const word of URGENCY_WORDS) {
            if (bodyText.includes(word)) foundUrgency.push(word);
        }
        if (foundUrgency.length > 0) {
            reasons.push('Contains urgency language: "' + foundUrgency.slice(0, 3).join('", "') + '"');
        }

        let foundMoney = [];
        for (const word of MONEY_WORDS) {
            if (bodyText.includes(word)) foundMoney.push(word);
        }
        if (foundMoney.length > 0) {
            reasons.push('References financial terms: "' + foundMoney.slice(0, 3).join('", "') + '"');
        }

        let foundThreat = [];
        for (const word of THREAT_WORDS) {
            if (bodyText.includes(word)) foundThreat.push(word);
        }
        if (foundThreat.length > 0) {
            reasons.push('Contains threatening language: "' + foundThreat.slice(0, 2).join('", "') + '"');
        }

        const urls = getUrlList(result);
        let riskyCount = 0;
        for (const u of urls) {
            if (u.risk === 'high' || u.risk === 'dangerous' || u.prediction === 'phishing' || u.is_phishing) {
                riskyCount++;
            }
        }
        if (riskyCount > 0) {
            reasons.push('Contains ' + riskyCount + ' suspicious link' + (riskyCount > 1 ? 's' : ''));
        }

        const headers = email.headers || {};
        const spf = (headers.spf || 'none').toLowerCase();
        const dkim = (headers.dkim || 'none').toLowerCase();
        const dmarc = (headers.dmarc || 'none').toLowerCase();
        let failedAuths = [];
        if (spf === 'fail') failedAuths.push('SPF');
        if (dkim === 'fail') failedAuths.push('DKIM');
        if (dmarc === 'fail') failedAuths.push('DMARC');
        if (failedAuths.length > 0) {
            reasons.push('Failed email authentication: ' + failedAuths.join(', '));
        }

        const attachments = email.attachments || [];
        let riskyAttachments = 0;
        for (const att of attachments) {
            const name = typeof att === 'string' ? att : (att.name || att.filename || '');
            if (isRiskyExtension(name)) riskyAttachments++;
        }
        if (riskyAttachments > 0) {
            reasons.push('Contains ' + riskyAttachments + ' potentially dangerous attachment' + (riskyAttachments > 1 ? 's' : ''));
        }

        if (reasons.length === 0) {
            reasons.push('The content patterns match known phishing characteristics.');
        }
    } else {
        const headers = email.headers || {};
        const spf = (headers.spf || 'none').toLowerCase();
        const dkim = (headers.dkim || 'none').toLowerCase();
        const dmarc = (headers.dmarc || 'none').toLowerCase();
        let passedAuths = [];
        if (spf === 'pass') passedAuths.push('SPF');
        if (dkim === 'pass') passedAuths.push('DKIM');
        if (dmarc === 'pass') passedAuths.push('DMARC');
        if (passedAuths.length > 0) {
            reasons.push('Email authentication passed: ' + passedAuths.join(', '));
        }

        let urgencyFound = false;
        for (const word of URGENCY_WORDS) {
            if (bodyText.includes(word)) { urgencyFound = true; break; }
        }
        if (!urgencyFound) {
            reasons.push('No urgency or pressure language detected.');
        }

        const urls = getUrlList(result);
        let allSafe = true;
        for (const u of urls) {
            if (u.risk === 'high' || u.risk === 'dangerous' || u.prediction === 'phishing' || u.is_phishing) {
                allSafe = false;
                break;
            }
        }
        if (urls.length > 0 && allSafe) {
            reasons.push('All ' + urls.length + ' link' + (urls.length > 1 ? 's appear' : ' appears') + ' to be safe.');
        } else if (urls.length === 0) {
            reasons.push('No suspicious links found in the email.');
        }

        reasons.push('Content does not match known phishing patterns.');
    }

    listEl.innerHTML = reasons.map(r => '<li>' + escapeHtml(r) + '</li>').join('');
}

function renderLinkSafety(result) {
    const section = document.getElementById('linkSafety');
    const container = document.getElementById('linkCards');
    const urls = getUrlList(result);

    if (urls.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    let html = '';
    for (const u of urls) {
        const url = u.url || u.link || '';
        let riskLevel, riskClass, desc;

        if (u.risk === 'high' || u.risk === 'dangerous' || u.prediction === 'phishing' || u.is_phishing) {
            riskLevel = 'Dangerous';
            riskClass = 'dangerous';
            desc = u.reason || 'This link shows characteristics commonly associated with phishing or malware distribution.';
        } else if (u.risk === 'medium' || u.risk === 'suspicious') {
            riskLevel = 'Suspicious';
            riskClass = 'suspicious';
            desc = u.reason || 'This link has some unusual characteristics. Proceed with caution.';
        } else {
            riskLevel = 'Appears Safe';
            riskClass = 'safe';
            desc = u.reason || 'This link does not show obvious signs of malicious intent.';
        }

        html += '<div class="link-card">' +
            '<div class="link-card-url">' + escapeHtml(url) + '</div>' +
            '<div class="link-card-risk ' + riskClass + '">' + riskLevel + '</div>' +
            '<div class="link-card-desc">' + escapeHtml(desc) + '</div>' +
            '</div>';
    }
    container.innerHTML = html;
}

function renderSenderVerification(email, result) {
    const container = document.getElementById('verificationRows');
    const headers = email.headers || {};

    const checks = [
        {
            key: 'spf',
            name: 'SPF (Sender Policy Framework)',
            desc: 'Verifies the sending server is authorized'
        },
        {
            key: 'dkim',
            name: 'DKIM (DomainKeys Identified Mail)',
            desc: 'Verifies the email was not altered in transit'
        },
        {
            key: 'dmarc',
            name: 'DMARC (Domain-based Message Authentication)',
            desc: 'Ensures SPF and DKIM alignment'
        }
    ];

    let html = '';
    for (const check of checks) {
        const val = (headers[check.key] || 'none').toLowerCase();
        let statusClass, statusText;
        if (val === 'pass') {
            statusClass = 'verification-status verification-status-pass';
            statusText = 'Pass';
        } else if (val === 'fail') {
            statusClass = 'verification-status verification-status-fail';
            statusText = 'Fail';
        } else {
            statusClass = 'verification-status verification-status-none';
            statusText = 'None';
        }

        html += '<div class="verification-row">' +
            '<div>' +
            '<div class="verification-name">' + escapeHtml(check.name) + '</div>' +
            '<div class="verification-desc">' + escapeHtml(check.desc) + '</div>' +
            '</div>' +
            '<span class="' + statusClass + '">' + statusText + '</span>' +
            '</div>';
    }

    container.innerHTML = html;
}

function renderRecommendations(isPhishing, result, email) {
    const list = document.getElementById('recoList');
    const recommendations = [];

    if (isPhishing) {
        recommendations.push('Do not click any links or download any attachments from this email.');
        recommendations.push('Do not reply to this email or provide any personal information.');
        recommendations.push('Report this email as phishing to your IT department or email provider.');

        const attachments = email.attachments || [];
        if (attachments.length > 0) {
            recommendations.push('Delete any downloaded attachments from this sender immediately.');
        }

        recommendations.push('If you already clicked a link, change your passwords and monitor your accounts.');
    } else {
        recommendations.push('This email appears safe, but always remain cautious with unexpected requests.');
        recommendations.push('Verify the sender if the email asks for sensitive information.');
        recommendations.push('Keep your email security software up to date.');
    }

    list.innerHTML = recommendations.map((r, i) =>
        '<li data-num="' + (i + 1) + '">' + escapeHtml(r) + '</li>'
    ).join('');
}

/* ======================== THEME SYSTEM ======================== */

function setTheme(name) {
    state.theme = name;
    document.documentElement.setAttribute('data-theme', name);
    localStorage.setItem('pg-theme', name);

    document.querySelectorAll('.theme-swatch').forEach(sw => {
        sw.classList.toggle('active', sw.getAttribute('data-theme') === name);
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

function loadPreferences() {
    const savedTheme = localStorage.getItem('pg-theme');
    const savedMode = localStorage.getItem('pg-mode');
    const savedFontSize = localStorage.getItem('pg-fontSize');

    if (savedTheme) setTheme(savedTheme);
    else setTheme('cupertino');

    if (savedMode) setMode(savedMode);
    else setMode('dark');

    if (savedFontSize) setFontSize(parseInt(savedFontSize, 10));
    else setFontSize(11);
}

/* ======================== SETTINGS MODAL ======================== */

function openSettings() {
    document.getElementById('settingsModal').style.display = 'flex';
    checkAuthStatus();
}

function closeSettings() {
    document.getElementById('settingsModal').style.display = 'none';
}

/* ======================== OUTLOOK CONNECTION ======================== */

async function checkAuthStatus() {
    try {
        const r = await fetch('/api/auth/status');
        const data = await r.json();
        const connectBtn = document.getElementById('connectBtn');
        const userLabel = document.getElementById('topbarUser');
        const statusEl = document.getElementById('connectionStatus');

        if (data.connected) {
            connectBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg> Disconnect';
            userLabel.textContent = data.user_name || 'Connected';
            if (statusEl) statusEl.textContent = 'Connected as ' + (data.user_email || data.user_name);
        } else {
            connectBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/></svg> Connect';
            userLabel.textContent = '';
            if (statusEl) statusEl.textContent = '';
        }
    } catch (e) { /* ignore */ }
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
            await fetch('/api/auth/disconnect', { method: 'POST' });
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
        const resp = await fetch('/api/auth/connect', {
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
        const pollInterval = setInterval(async () => {
            try {
                const status = await fetch('/api/auth/status');
                const statusData = await status.json();
                if (statusData.connected) {
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
            clearInterval(pollInterval);
            if (statusEl && statusEl.textContent === 'Waiting for sign-in...') {
                statusEl.textContent = 'Timed out. Please try again.';
            }
        }, 120000);

    } catch (e) {
        if (statusEl) statusEl.textContent = 'Connection error: ' + e.message;
        setStatus('Connection error: ' + e.message);
    }
}

async function refreshOutlookEmails() {
    try {
        setStatus('Refreshing emails...');
        const r = await fetch('/api/auth/refresh', { method: 'POST' });
        const data = await r.json();
        if (data.error) {
            setStatus('Refresh failed: ' + data.error);
        } else {
            await loadEmails();
            setStatus('Loaded ' + data.count + ' emails');
        }
    } catch (e) {
        setStatus('Refresh error: ' + e.message);
    }
}

/* ======================== NAV ======================== */

function setNav(view) {
    state.navView = view;
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-nav') === view);
    });
    renderEmailList();
}

/* ======================== EVENT LISTENERS ======================== */

document.addEventListener('DOMContentLoaded', () => {
    loadPreferences();
    randomTip();
    loadEmails();

    /* Search */
    const searchInput = document.getElementById('searchInput');
    const searchClear = document.getElementById('searchClear');

    searchInput.addEventListener('input', () => {
        searchClear.classList.toggle('visible', searchInput.value.length > 0);
        renderEmailList();
    });

    searchClear.addEventListener('click', () => {
        searchInput.value = '';
        searchClear.classList.remove('visible');
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

    /* Scan All */
    document.getElementById('scanAllBtn').addEventListener('click', () => {
        scanAll();
    });

    /* Scan / Hide toggle */
    document.getElementById('scanBtn').addEventListener('click', () => {
        if (state.selectedIdx === null) return;

        if (state.inlineVisible) {
            document.getElementById('scanResults').style.display = 'none';
            state.inlineVisible = false;
            const scanBtn = document.getElementById('scanBtn');
            scanBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/></svg> Scan for Phishing';
        } else {
            if (state.scanResults[state.selectedIdx]) {
                showScanResults(state.selectedIdx);
            } else {
                scanEmail(state.selectedIdx);
            }
        }
    });

    /* Header toggle */
    document.getElementById('headerToggle').addEventListener('click', () => {
        const details = document.getElementById('headerDetails');
        const chevron = document.getElementById('headerChevron');
        const isOpen = details.style.display !== 'none';
        details.style.display = isOpen ? 'none' : 'block';
        chevron.classList.toggle('open', !isOpen);
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
    document.getElementById('settingsBtn').addEventListener('click', openSettings);
    document.getElementById('settingsClose').addEventListener('click', closeSettings);

    /* Close modal on overlay click */
    document.getElementById('settingsModal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('settingsModal')) {
            closeSettings();
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

    /* Check auth status on load */
    checkAuthStatus();

    /* Keyboard: Escape to close modals */
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeSettings();
            closeConnectModal();
        }
    });
});
