"""
PhishGuard Web Dashboard — Flask Backend
Serves the SPA and provides REST API for phishing email analysis.
"""

import base64
import hashlib
import html
import re
import secrets
import socket
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, jsonify, redirect, request, send_from_directory, session

from phishing_detector import PhishingDetector

try:
    import requests as http_requests
except ImportError:
    http_requests = None

# ─────────────────────────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = secrets.token_hex(32)

# ─────────────────────────────────────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────────────────────────────────────
detector = PhishingDetector()
MODEL_PATH = Path(__file__).parent / "phishing_model.pkl"
try:
    detector.load_model(str(MODEL_PATH))
    print("PhishGuard model loaded successfully.")
except Exception as e:
    print(f"WARNING: Could not load model: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  Server-side scan results cache
# ─────────────────────────────────────────────────────────────────────────────
scan_results = {}

# ─────────────────────────────────────────────────────────────────────────────
#  Domain Reputation System
# ─────────────────────────────────────────────────────────────────────────────
_TRUSTED_DOMAINS = frozenset({
    # Major email providers
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "msn.com", "yahoo.com", "ymail.com", "aol.com", "icloud.com", "me.com",
    "mac.com", "protonmail.com", "proton.me", "zoho.com", "mail.com",
    "gmx.com", "gmx.net", "fastmail.com", "tutanota.com", "hey.com",
    # Tech companies
    "google.com", "microsoft.com", "apple.com", "amazon.com", "meta.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "github.com", "gitlab.com", "atlassian.com", "slack.com", "zoom.us",
    "dropbox.com", "salesforce.com", "adobe.com", "oracle.com", "ibm.com",
    "intel.com", "cisco.com", "vmware.com", "shopify.com", "stripe.com",
    "twilio.com", "cloudflare.com", "digitalocean.com", "heroku.com",
    "notion.so", "figma.com", "canva.com", "spotify.com", "netflix.com",
    # Financial / payments
    "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "citibank.com", "americanexpress.com", "discover.com", "capitalone.com",
    "fidelity.com", "schwab.com", "venmo.com", "squareup.com",
    # Services / social
    "uber.com", "lyft.com", "airbnb.com", "doordash.com", "grubhub.com",
    "reddit.com", "pinterest.com", "tiktok.com", "snapchat.com",
    "whatsapp.com", "telegram.org", "signal.org", "discord.com",
    # Enterprise / productivity
    "jira.com", "trello.com", "asana.com", "monday.com",
    "hubspot.com", "zendesk.com", "intercom.com", "mailchimp.com",
    "sendgrid.net", "amazonaws.com", "azure.com",
    # Education / government
    "edu", "gov", "mil",
})

_SUSPICIOUS_TLDS = frozenset({
    ".xyz", ".top", ".buzz", ".click", ".club", ".info", ".win", ".bid",
    ".loan", ".racing", ".review", ".stream", ".gq", ".cf", ".tk", ".ml",
    ".ga", ".pw", ".work", ".party", ".date", ".trade", ".webcam",
    ".science", ".accountant", ".cricket", ".faith", ".zip", ".mov",
})

_FREEMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "ymail.com", "aol.com", "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me", "zoho.com", "mail.com", "gmx.com",
    "gmx.net", "fastmail.com", "tutanota.com",
})

# Cache for WHOIS / DNS lookups (domain -> result dict)
_domain_rep_cache = {}


def _get_domain_age(domain):
    """Try to get domain age in days via python-whois. Returns None if unavailable."""
    try:
        import whois
        from datetime import datetime, timezone
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - created).days
            return max(0, age)
    except Exception:
        pass
    return None


def _check_dns(domain):
    """Check if domain resolves via DNS. Returns True/False/None."""
    try:
        socket.getaddrinfo(domain, None, socket.AF_INET, socket.SOCK_STREAM)
        return True
    except socket.gaierror:
        return False
    except Exception:
        return None


def _check_phishtank(domain):
    """Check if a domain is in PhishTank's database. Returns True/False/None."""
    if not http_requests:
        return None
    try:
        url = f"http://{domain}/"
        resp = http_requests.post(
            "https://checkurl.phishtank.com/checkurl/",
            data={"url": url, "format": "json"},
            headers={"User-Agent": "phishtank/PhishGuard"},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", {})
            return results.get("in_database", False) and results.get("valid", False)
    except Exception:
        pass
    return None


def _check_domain_reputation(domain):
    """
    Compute a worldwide reputation score for a domain (0.0-1.0).
    Uses multiple heuristics without requiring API keys:
      - Trusted domain list match
      - TLD risk analysis
      - Domain age via WHOIS (if available)
      - Structural analysis (impersonation patterns)
      - DNS existence check
    Results are cached to avoid repeated lookups.
    """
    if not domain:
        return {"score": 0.5, "signals": [], "category": "unknown"}

    domain = domain.lower().strip()
    if domain in _domain_rep_cache:
        return _domain_rep_cache[domain]

    score = 0.5
    signals = []

    # 1. Trusted domain list
    is_trusted = (
        domain in _TRUSTED_DOMAINS
        or any(domain.endswith("." + td) for td in _TRUSTED_DOMAINS)
        or domain.split(".")[-1] in _TRUSTED_DOMAINS  # .edu, .gov, .mil
    )
    if is_trusted:
        score += 0.35
        signals.append(("trusted", "Globally recognized domain"))

    is_freemail = domain in _FREEMAIL_DOMAINS
    if is_freemail:
        signals.append(("freemail", "Free email provider — anyone can register"))
        score -= 0.05  # slight reduction since freemail can be abused

    # 2. TLD risk analysis
    tld = "." + domain.split(".")[-1] if "." in domain else ""
    if tld in _SUSPICIOUS_TLDS:
        score -= 0.25
        signals.append(("bad_tld", f"Uses high-risk TLD ({tld})"))
    elif tld in (".com", ".org", ".net", ".co"):
        score += 0.05
        signals.append(("good_tld", "Common, established TLD"))

    # 3. Structural analysis
    name_part = domain.split(".")[0] if "." in domain else domain

    # Excessive hyphens -> impersonation / DGA
    if name_part.count("-") >= 2:
        score -= 0.15
        signals.append(("hyphens", "Multiple hyphens — common in fake domains"))

    # Numbers mixed with letters in the name (e.g., g00gle, paypa1)
    has_digits = bool(re.search(r'\d', name_part))
    has_letters = bool(re.search(r'[a-z]', name_part))
    if has_digits and has_letters and len(name_part) > 6:
        score -= 0.15
        signals.append(("mixedchars", "Mix of letters and numbers — possible impersonation"))

    # Very long domain name
    if len(name_part) > 20:
        score -= 0.1
        signals.append(("long_name", "Unusually long domain name"))

    # Brand impersonation patterns
    _brands = ["paypal", "apple", "google", "microsoft", "amazon", "chase",
               "wellsfargo", "bankofamerica", "netflix", "facebook", "instagram"]
    for brand in _brands:
        if brand in name_part and domain not in _TRUSTED_DOMAINS:
            score -= 0.25
            signals.append(("impersonation", f"Contains '{brand}' but isn't the real domain"))
            break

    # 4. Domain age via WHOIS
    domain_age_days = _get_domain_age(domain)
    if domain_age_days is not None:
        if domain_age_days < 30:
            score -= 0.25
            signals.append(("new_domain", f"Domain is only {domain_age_days} days old"))
        elif domain_age_days < 180:
            score -= 0.1
            signals.append(("young_domain", f"Domain is {domain_age_days} days old"))
        elif domain_age_days > 365 * 3:
            score += 0.1
            years = domain_age_days // 365
            signals.append(("established", f"Domain has been active for {years}+ years"))

    # 5. DNS existence check
    dns_exists = _check_dns(domain)
    if dns_exists is False:
        score -= 0.3
        signals.append(("no_dns", "Domain does not resolve — may not exist"))
    elif dns_exists is True and not is_trusted:
        signals.append(("dns_ok", "Domain resolves in DNS"))

    # 6. PhishTank database check
    phishtank_hit = _check_phishtank(domain)
    if phishtank_hit is True:
        score -= 0.5
        signals.append(("phishtank", "Listed in PhishTank as a known phishing site"))
    elif phishtank_hit is False:
        signals.append(("phishtank_clean", "Not found in PhishTank database"))

    # Clamp
    score = max(0.0, min(1.0, score))

    # Categorize
    if score >= 0.7:
        category = "trusted"
    elif score >= 0.4:
        category = "neutral"
    else:
        category = "suspicious"

    result = {"score": score, "signals": signals, "category": category}
    _domain_rep_cache[domain] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  HTML to Text
# ─────────────────────────────────────────────────────────────────────────────
class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and decode entities, skipping non-visible content."""
    _SKIP_TAGS = {"style", "script", "head", "title"}
    _BLOCK_TAGS = {
        "p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3", "h4", "h5",
        "h6", "blockquote", "pre", "table", "thead", "tbody", "tfoot",
        "section", "article", "header", "footer", "nav", "ul", "ol",
    }

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag_l in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth == 0 and tag_l in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def handle_entityref(self, name):
        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&#{name};"))

    def get_text(self):
        return "".join(self._parts)


def _html_to_text(raw_html):
    """Safely convert HTML to clean plain text that reads like a normal email."""
    try:
        parser = _HTMLTextExtractor()
        parser.feed(raw_html)
        text = parser.get_text()
    except Exception:
        text = re.sub(r'<[^>]+>', ' ', raw_html)
    # Clean up each line: collapse inline whitespace, strip
    lines = [re.sub(r'[^\S\n]+', ' ', ln).strip() for ln in text.splitlines()]
    # Remove empty lines, then rejoin — consecutive non-empty lines become one paragraph
    # Only keep a blank line where there were 2+ blank lines (real paragraph break)
    cleaned = []
    blank_count = 0
    for ln in lines:
        if ln == '':
            blank_count += 1
        else:
            if blank_count >= 2 and cleaned:
                cleaned.append('')  # one paragraph break
            elif blank_count == 1 and cleaned:
                cleaned.append('')  # preserve single blank line as paragraph break
            blank_count = 0
            cleaned.append(ln)
    return '\n'.join(cleaned).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Mock Emails (all 24 test emails from the desktop app)
# ─────────────────────────────────────────────────────────────────────────────
MOCK_EMAILS = [
    # ── Legitimate emails ────────────────────────────────────────────────────
    {
        "subject": "Q1 2026 Revenue Report \u2014 Final Numbers",
        "from": {"emailAddress": {"name": "Sarah Chen", "address": "sarah.chen@company.com"}},
        "receivedDateTime": "2026-03-08T09:14:00Z",
        "isRead": True,
        "bodyPreview": "Hi team, Attached is the finalized Q1 revenue report. We exceeded our target by 12% across all regions.",
        "body": {"content": "Hi team,\n\nAttached is the finalized Q1 revenue report. We exceeded our target by 12% across all regions. Key highlights:\n\n- North America: $4.2M (+15%)\n- EMEA: $2.8M (+8%)\n- APAC: $1.9M (+14%)\n\nThe board presentation is scheduled for Thursday at 2pm EST. Please review your section and flag any discrepancies by EOD Wednesday.\n\nBest,\nSarah"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
        "hasAttachments": True,
        "attachments": [{"name": "Q1_Revenue_Report.xlsx", "contentType": "application/vnd.ms-excel", "size": 189000}],
    },
    {
        "subject": "Re: Team lunch Friday?",
        "from": {"emailAddress": {"name": "Mike Rodriguez", "address": "mike.r@company.com"}},
        "receivedDateTime": "2026-03-08T08:42:00Z",
        "isRead": False,
        "bodyPreview": "Count me in! Should we try that new Thai place on 5th? I heard their pad thai is amazing.",
        "body": {"content": "Count me in! Should we try that new Thai place on 5th? I heard their pad thai is amazing. I can make a reservation for 12:30 if that works for everyone.\n\nAlso, reminder that it's Jake's birthday next week \u2014 maybe we can combine?\n\n- Mike"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Your Amazon order has shipped",
        "from": {"emailAddress": {"name": "Amazon", "address": "shipment-tracking@amazon.com"}},
        "receivedDateTime": "2026-03-07T22:15:00Z",
        "isRead": True,
        "bodyPreview": "Your order #114-3941872-6284210 has shipped and is on its way. Estimated delivery: March 10, 2026.",
        "body": {"content": "Your order #114-3941872-6284210 has shipped.\n\nItems:\n- USB-C Hub, 7-in-1 ($34.99)\n- Wireless Mouse ($24.99)\n\nEstimated delivery: March 10, 2026\nCarrier: UPS\nTracking: 1Z999AA10123456784\n\nThank you for shopping with Amazon."},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Meeting notes: Project Phoenix kickoff",
        "from": {"emailAddress": {"name": "Lisa Park", "address": "lisa.park@company.com"}},
        "receivedDateTime": "2026-03-07T17:30:00Z",
        "isRead": True,
        "bodyPreview": "Hi all, Here are the notes from today's Project Phoenix kickoff. Action items are highlighted in bold.",
        "body": {"content": "Hi all,\n\nHere are the notes from today's Project Phoenix kickoff meeting:\n\n1. Timeline: MVP by June 30, full launch September 15\n2. Tech stack: Python backend, React frontend, PostgreSQL\n3. Team leads: Backend (James), Frontend (Priya), QA (Tom)\n\nAction items:\n- James: Set up CI/CD pipeline by Friday\n- Priya: Share wireframes by Monday\n- Tom: Draft test plan by next Wednesday\n\nNext standup: Tuesday 10am\n\nThanks,\nLisa"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
        "hasAttachments": True,
        "attachments": [{"name": "Phoenix_Kickoff_Notes.pdf", "contentType": "application/pdf", "size": 92000}],
    },
    {
        "subject": "Your monthly bank statement is ready",
        "from": {"emailAddress": {"name": "Chase Bank", "address": "no-reply@chase.com"}},
        "receivedDateTime": "2026-03-07T14:00:00Z",
        "isRead": True,
        "bodyPreview": "Your February 2026 statement is now available. Log in to chase.com to view your account details.",
        "body": {"content": "Your February 2026 statement is now available.\n\nAccount ending in 4829\nStatement period: Feb 1 - Feb 28, 2026\n\nTo view your statement, log in to your account at chase.com.\n\nIf you have questions, call us at 1-800-935-9935.\n\nThank you for banking with Chase."},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Invitation: Design review \u2014 March 12",
        "from": {"emailAddress": {"name": "Google Calendar", "address": "calendar-notification@google.com"}},
        "receivedDateTime": "2026-03-07T11:20:00Z",
        "isRead": False,
        "bodyPreview": "Priya Sharma has invited you to Design review on Wednesday, March 12 at 3:00pm EST.",
        "body": {"content": "Design review\nWednesday, March 12, 2026 \u00b7 3:00pm \u2013 4:00pm EST\n\nOrganizer: Priya Sharma (priya@company.com)\nLocation: Conference Room B / Zoom link attached\n\nAgenda:\n- Review updated wireframes\n- Discuss color palette and typography\n- Finalize component library structure\n\nYes \u00b7 No \u00b7 Maybe"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Re: Vacation request approved",
        "from": {"emailAddress": {"name": "HR Department", "address": "hr@company.com"}},
        "receivedDateTime": "2026-03-06T16:45:00Z",
        "isRead": True,
        "bodyPreview": "Your vacation request for March 24-28 has been approved. Enjoy your time off!",
        "body": {"content": "Hi,\n\nYour vacation request has been approved.\n\nDates: March 24 - March 28, 2026 (5 business days)\nRemaining PTO balance: 12 days\n\nPlease ensure your out-of-office reply is set up and that you've delegated any urgent tasks before your leave.\n\nEnjoy your time off!\n\nBest regards,\nHR Department"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    # ── Phishing emails ──────────────────────────────────────────────────────
    {
        "subject": "URGENT: Your account has been compromised!",
        "from": {"emailAddress": {"name": "Microsoft Security", "address": "security-alert@m1cr0soft-support.com"}},
        "receivedDateTime": "2026-03-08T07:33:00Z",
        "isRead": False,
        "bodyPreview": "We detected suspicious activity on your Microsoft account. Your account will be LOCKED in 24 hours unless you verify immediately.",
        "body": {"content": "Dear Valued Customer,\n\nWe detected suspicious activity on your Microsoft account. Unauthorized login attempts were made from an unrecognized device in Russia.\n\nYour account will be PERMANENTLY LOCKED within 24 hours unless you verify your identity immediately.\n\nClick here to verify your account: http://m1cr0soft-security-verify.com/login?user=target\n\nIf you did not make these changes, please secure your account NOW to prevent data loss.\n\nMicrosoft Security Team\nThis is an automated message \u2014 do not reply."},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=fail dkim=fail dmarc=fail"}],
    },
    {
        "subject": "You have a pending PayPaI refund of $847.50",
        "from": {"emailAddress": {"name": "PayPal Support", "address": "refund-processing@paypa1-service.net"}},
        "receivedDateTime": "2026-03-07T19:55:00Z",
        "isRead": False,
        "bodyPreview": "A refund of $847.50 is pending on your account. Please confirm your banking details to receive the refund within 24 hours.",
        "body": {"content": "Dear Customer,\n\nA refund of $847.50 USD has been issued to your PayPal account from a recent transaction dispute.\n\nTo process this refund, we need you to confirm your banking information:\n\nClick here to confirm: http://paypa1-refund-center.com/confirm?id=8847261\n\nIf you do not confirm within 48 hours, the refund will be cancelled and the funds returned to the merchant.\n\nTransaction ID: TXN-9928441\nAmount: $847.50 USD\nStatus: Pending verification\n\nPayPal Customer Service"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=softfail dkim=fail dmarc=fail"}],
    },
    {
        "subject": "IT Department: Password expires today",
        "from": {"emailAddress": {"name": "IT Help Desk", "address": "admin@it-helpdesk-portal.xyz"}},
        "receivedDateTime": "2026-03-07T13:10:00Z",
        "isRead": False,
        "bodyPreview": "Your corporate password expires today. Click below to update your password and avoid losing access to all company systems.",
        "body": {"content": "Dear Employee,\n\nYour corporate email password will expire TODAY at 11:59 PM.\n\nIf you do not update your password immediately, you will lose access to:\n- Email and Calendar\n- SharePoint and Teams\n- VPN and internal systems\n\nUpdate your password now: http://192.168.1.100:8080/password-reset/corporate\n\nThis is an automated message from the IT Help Desk. Do not reply to this email.\n\nIT Support Team"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=none dkim=none dmarc=none"}],
    },
    {
        "subject": "Congratulations! You've won a $500 Apple Gift Card",
        "from": {"emailAddress": {"name": "Apple Rewards", "address": "rewards@apple-giftcard-promo.com"}},
        "receivedDateTime": "2026-03-06T20:00:00Z",
        "isRead": True,
        "bodyPreview": "You've been selected as the winner of our monthly sweepstakes! Claim your $500 Apple Gift Card before it expires.",
        "body": {"content": "\ud83c\udf89 CONGRATULATIONS! \ud83c\udf89\n\nYou have been randomly selected as the WINNER of our March 2026 customer appreciation sweepstakes!\n\nPrize: $500 Apple Gift Card\nClaim by: March 10, 2026\n\nTo claim your prize, simply click the link below and enter your shipping details:\n\nhttp://apple-rewards-winner.com/claim?winner=true&prize=500\n\nThis offer is exclusive and non-transferable. Only 1 winner per household.\n\nApple Promotions Team\n\n*This promotion is not affiliated with Apple Inc."},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=fail dkim=none dmarc=fail"}],
    },
    {
        "subject": "Invoice #INV-29481 attached \u2014 payment overdue",
        "from": {"emailAddress": {"name": "Accounts Receivable", "address": "billing@secure-invoice-portal.ru"}},
        "receivedDateTime": "2026-03-06T09:30:00Z",
        "isRead": True,
        "bodyPreview": "Please find the attached invoice for services rendered. Payment is overdue by 15 days. Immediate action required.",
        "body": {"content": "Dear Sir/Madam,\n\nPlease find attached invoice #INV-29481 for services rendered in February 2026.\n\nAmount Due: $3,240.00\nDue Date: February 20, 2026 (OVERDUE)\nLate Fee: $162.00\n\nPlease process payment immediately to avoid further penalties. If you have already made this payment, please disregard this notice.\n\nView and pay invoice: http://secure-invoice-portal.ru/pay?inv=29481&auth=x7k2m\n\nFor questions, contact our billing department.\n\nRegards,\nAccounts Receivable Department"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=fail dkim=fail dmarc=fail"}],
        "hasAttachments": True,
        "attachments": [{"name": "Invoice-29481.pdf", "contentType": "application/pdf", "size": 245000}],
    },
    {
        "subject": "Shared document: 'Salary_Review_2026.xlsx'",
        "from": {"emailAddress": {"name": "OneDrive", "address": "notification@onedrlve-share.com"}},
        "receivedDateTime": "2026-03-05T15:22:00Z",
        "isRead": False,
        "bodyPreview": "Your manager has shared a confidential document with you. Click to view the salary review spreadsheet.",
        "body": {"content": "Your manager has shared a document with you via OneDrive.\n\nDocument: Salary_Review_2026.xlsx\nShared by: David Thompson (Manager)\nAccess: View only\n\nThis document contains confidential salary information for your department's annual review.\n\nView document: http://onedrlve-share.com/d/s!salary_review_2026?auth=open\n\nYou must sign in with your corporate credentials to access this file.\n\nMicrosoft OneDrive\nThis is an automated notification."},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=fail dkim=none dmarc=fail"}],
        "hasAttachments": True,
        "attachments": [{"name": "Salary_Review_2026.xlsx", "contentType": "application/vnd.ms-excel", "size": 54000}],
    },
    # ── More legitimate emails ───────────────────────────────────────────────
    {
        "subject": "Weekly engineering digest \u2014 March 3-7",
        "from": {"emailAddress": {"name": "Engineering Bot", "address": "eng-digest@company.com"}},
        "receivedDateTime": "2026-03-07T18:00:00Z",
        "isRead": True,
        "bodyPreview": "This week: 47 PRs merged, 12 bugs fixed, 3 new features shipped. Sprint velocity is up 8% from last week.",
        "body": {"content": "Engineering Weekly Digest\nMarch 3 - March 7, 2026\n\nHighlights:\n- 47 pull requests merged\n- 12 bugs fixed (4 critical)\n- 3 new features shipped to production\n- Sprint velocity: 82 points (+8%)\n\nTop contributors:\n1. James W. \u2014 9 PRs\n2. Priya S. \u2014 7 PRs\n3. Tom K. \u2014 6 PRs\n\nUpcoming:\n- Database migration scheduled for Saturday 2am EST\n- New CI pipeline goes live Monday\n\n\u2014 Engineering Bot"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Your Uber receipt",
        "from": {"emailAddress": {"name": "Uber Receipts", "address": "uber.us@uber.com"}},
        "receivedDateTime": "2026-03-06T23:40:00Z",
        "isRead": True,
        "bodyPreview": "Thanks for riding with Uber. Your trip on March 6 cost $18.43. View your full receipt for details.",
        "body": {"content": "Thanks for riding, and thanks for tipping!\n\nTrip on March 6, 2026\nUberX \u00b7 7:12 PM - 7:34 PM\n\nPickup: 425 Market St, San Francisco\nDropoff: 1 Ferry Building, San Francisco\n\nTrip fare: $14.50\nService fee: $2.43\nTip: $3.00\nTotal: $18.43\n\nCharged to Visa ending in 4829\n\nRate your driver: \u2605\u2605\u2605\u2605\u2605"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "GitHub: [company/phoenix] PR #247 merged",
        "from": {"emailAddress": {"name": "GitHub", "address": "notifications@github.com"}},
        "receivedDateTime": "2026-03-06T14:15:00Z",
        "isRead": True,
        "bodyPreview": "Pull request #247 'Add rate limiting to API endpoints' was merged by james-w into main.",
        "body": {"content": "Pull request #247 merged\n\nAdd rate limiting to API endpoints\nby james-w \u00b7 merged into main\n\n+142 -23 across 6 files\n\nReviewers: priya-s (approved), tom-k (approved)\n\nChanges:\n- Added token bucket rate limiter middleware\n- 100 req/min for authenticated users\n- 20 req/min for anonymous users\n- Added rate limit headers to responses\n\nView on GitHub: github.com/company/phoenix/pull/247"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Your Slack workspace weekly summary",
        "from": {"emailAddress": {"name": "Slack", "address": "feedback@slack.com"}},
        "receivedDateTime": "2026-03-07T08:00:00Z",
        "isRead": True,
        "bodyPreview": "Here's what happened in your workspace this week. 234 messages across 12 channels.",
        "body": {"content": "Your weekly Slack summary\n\nThis week in your workspace:\n- 234 messages sent across 12 channels\n- 3 new channels created\n- Most active: #engineering (89 messages)\n\nTop threads:\n1. Database migration plan \u2014 24 replies\n2. New office snack suggestions \u2014 18 replies\n3. Code review best practices \u2014 15 replies\n\nView in Slack: https://app.slack.com/workspace/summary\nManage notifications: https://slack.com/account/notifications\n\n\u2014 Slack"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Your Jira sprint is ending tomorrow",
        "from": {"emailAddress": {"name": "Jira", "address": "jira@company.atlassian.net"}},
        "receivedDateTime": "2026-03-06T09:00:00Z",
        "isRead": False,
        "bodyPreview": "Sprint 'Phoenix v1.2' ends March 7. You have 3 open tickets remaining.",
        "body": {"content": "Sprint ending soon\n\nSprint: Phoenix v1.2\nEnds: March 7, 2026\n\nYour open tickets:\n- PHOE-341: Fix login timeout issue (In Progress)\n- PHOE-355: Add export to CSV button (To Do)\n- PHOE-362: Update API docs (To Do)\n\nView sprint board: https://company.atlassian.net/jira/board/42\nView your tickets: https://company.atlassian.net/jira/my-work\n\nIf you need to carry tickets over, please update the status before EOD.\n\n\u2014 Jira Automation"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "New sign-in to your Google Account",
        "from": {"emailAddress": {"name": "Google", "address": "no-reply@accounts.google.com"}},
        "receivedDateTime": "2026-03-05T21:30:00Z",
        "isRead": True,
        "bodyPreview": "Your Google Account was just signed in to from a new MacBook Pro in San Francisco, CA.",
        "body": {"content": "New sign-in to your Google Account\n\nYour account was just signed in to from a new device.\n\nDevice: MacBook Pro\nLocation: San Francisco, CA, United States\nTime: March 5, 2026, 9:30 PM PST\nBrowser: Chrome\n\nIf this was you, you can disregard this message.\n\nIf this wasn't you, your account may be compromised. Secure your account:\nhttps://myaccount.google.com/security\n\nYou can also review your recent activity:\nhttps://myaccount.google.com/notifications\n\n\u2014 The Google Accounts team"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    {
        "subject": "Your npm package has a new vulnerability",
        "from": {"emailAddress": {"name": "GitHub Security", "address": "noreply@github.com"}},
        "receivedDateTime": "2026-03-05T12:45:00Z",
        "isRead": True,
        "bodyPreview": "Dependabot found a high severity vulnerability in lodash (CVE-2026-12345) used by company/phoenix.",
        "body": {"content": "Dependabot alert: high severity vulnerability\n\nRepository: company/phoenix\nPackage: lodash (npm)\nVulnerability: CVE-2026-12345\nSeverity: High\nAffected versions: < 4.17.22\nPatched version: 4.17.22\n\nDependabot has created a pull request to fix this:\nhttps://github.com/company/phoenix/pull/251\n\nReview the alert:\nhttps://github.com/company/phoenix/security/dependabot/14\n\nLearn more about Dependabot alerts:\nhttps://docs.github.com/en/code-security/dependabot\n\n\u2014 GitHub Security"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}],
    },
    # ── More phishing emails ─────────────────────────────────────────────────
    {
        "subject": "Action Required: Verify your Apple ID",
        "from": {"emailAddress": {"name": "Apple Support", "address": "support@apple-id-verification.net"}},
        "receivedDateTime": "2026-03-08T06:15:00Z",
        "isRead": False,
        "bodyPreview": "Your Apple ID has been disabled for security reasons. Verify your identity to restore access.",
        "body": {"content": "Dear Apple Customer,\n\nYour Apple ID has been temporarily disabled due to unusual activity detected on your account.\n\nTo restore access, please verify your identity within 48 hours:\n\nhttp://apple-id-verification.net/restore?session=xk29m&apple_id=verify\n\nIf you do not verify, your account will be permanently deleted and all purchases, photos, and iCloud data will be lost.\n\nFor your security, do not share this link with anyone.\n\nAlternatively, copy this URL into your browser:\nhttp://192.168.44.120:9090/appleid/confirm\n\nApple Support\nOne Apple Park Way, Cupertino, CA 95014"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=fail dkim=fail dmarc=fail"}],
    },
    {
        "subject": "DHL: Your package could not be delivered",
        "from": {"emailAddress": {"name": "DHL Express", "address": "tracking@dhl-delivery-notice.com"}},
        "receivedDateTime": "2026-03-07T10:20:00Z",
        "isRead": False,
        "bodyPreview": "We attempted to deliver your package but no one was available. Schedule a redelivery or it will be returned.",
        "body": {"content": "DHL Express Delivery Notice\n\nDear Customer,\n\nWe attempted to deliver your package today but were unable to complete delivery.\n\nTracking Number: DHL-7729104856\nStatus: Delivery Failed \u2014 Addressee Unavailable\n\nTo schedule a redelivery, click below:\nhttp://dhl-delivery-notice.com/reschedule?pkg=7729104856&token=a8x2k\n\nAlternatively, you can pay the $2.99 redelivery fee online:\nhttp://bit.ly/3xFk9Qm\n\nIf no action is taken within 5 days, your package will be returned to the sender.\n\nDHL Customer Service\nhttp://dhl-delivery-notice.com/support"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=softfail dkim=none dmarc=fail"}],
    },
    {
        "subject": "Netflix: Billing information update required",
        "from": {"emailAddress": {"name": "Netflix", "address": "billing@netflix-account-update.com"}},
        "receivedDateTime": "2026-03-06T18:30:00Z",
        "isRead": False,
        "bodyPreview": "We were unable to process your last payment. Update your billing info to avoid service interruption.",
        "body": {"content": "Hi,\n\nWe're having trouble with your current billing information. We were unable to process your payment for the Premium plan ($22.99/mo).\n\nUpdate your payment method to continue enjoying Netflix:\n\nhttp://netflix-account-update.com/billing?uid=9182736&plan=premium\n\nIf your payment details are not updated within 72 hours, your account will be suspended and your viewing history, profiles, and preferences may be deleted.\n\nCurrent plan: Premium (4K + HDR)\nAmount due: $22.99\nDue date: March 9, 2026\n\nUpdate now: http://netflix.com.account-billing.support.xyz/update\n\nNetflix Billing Team"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=fail dkim=fail dmarc=fail"}],
    },
    {
        "subject": "WeTransfer: Someone sent you files",
        "from": {"emailAddress": {"name": "WeTransfer", "address": "noreply@wetransfer-download.info"}},
        "receivedDateTime": "2026-03-05T14:50:00Z",
        "isRead": False,
        "bodyPreview": "You received 3 files (12.4 MB) from accounting@company.com. Download before they expire on March 8.",
        "body": {"content": "You have received files via WeTransfer\n\nFrom: accounting@company.com\nFiles: 3 (12.4 MB total)\n- Q1_Financial_Report.xlsx\n- Expense_Summary.pdf\n- Payroll_March.docm\n\nDownload link (expires March 8):\nhttp://wetransfer-download.info/d/x8k2m9n4?files=financial\n\nMessage from sender:\n\"Hi, please review the attached financial documents and confirm the numbers by Friday.\"\n\nTo report abuse: http://wetransfer-download.info/report\n\n\u00a9 2026 WeTransfer"},
        "internetMessageHeaders": [{"name": "Authentication-Results", "value": "spf=none dkim=fail dmarc=fail"}],
        "hasAttachments": True,
        "attachments": [
            {"name": "Q1_Financial_Report.xlsx", "contentType": "application/vnd.ms-excel", "size": 340000},
            {"name": "Expense_Summary.pdf", "contentType": "application/pdf", "size": 128000},
            {"name": "Payroll_March.docm", "contentType": "application/vnd.ms-word.document.macroEnabled", "size": 67000},
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: run a scan on an email by index
# ─────────────────────────────────────────────────────────────────────────────
def _scan_email(idx):
    """Run the phishing detector on an email by its index. Returns result dict."""
    if idx < 0 or idx >= len(MOCK_EMAILS):
        return None

    email = MOCK_EMAILS[idx]
    body_raw = email.get("body", {}).get("content", "")
    body_text = _html_to_text(body_raw) if "<" in body_raw and ">" in body_raw else body_raw
    subject = email.get("subject", "")
    full_text = f"Subject: {subject}\n\n{body_text}"

    headers = email.get("internetMessageHeaders", [])

    try:
        prediction, confidence, url_analysis, header_result = detector.predict(
            full_text, headers=headers
        )
    except Exception as e:
        return {
            "error": str(e),
            "idx": idx,
        }

    # Make url_analysis JSON-serializable (numpy types)
    if url_analysis:
        clean_features = {}
        for url, feats in url_analysis.get("features", {}).items():
            clean_features[url] = {k: int(v) if hasattr(v, 'item') else v for k, v in feats.items()}
        url_analysis["features"] = clean_features
        if "url_model_prob" in url_analysis:
            url_analysis["url_model_prob"] = float(url_analysis["url_model_prob"])

    # Make header_result JSON-serializable
    if header_result:
        header_result = {
            k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in header_result.items()
        }

    result = {
        "idx": idx,
        "prediction": int(prediction),
        "confidence": float(confidence),
        "url_analysis": url_analysis,
        "header_result": header_result,
    }

    scan_results[idx] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the SPA."""
    return send_from_directory("templates", "index.html")


@app.route("/api/reputation/<domain>", methods=["GET"])
def get_reputation(domain):
    """Return worldwide domain reputation."""
    rep = _check_domain_reputation(domain)
    # Convert signal tuples to serializable dicts
    serializable_signals = [
        {"type": sig[0], "message": sig[1]} for sig in rep.get("signals", [])
    ]
    return jsonify({
        "domain": domain,
        "score": rep["score"],
        "signals": serializable_signals,
        "category": rep["category"],
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Microsoft Graph API Client (OAuth2 Auth Code + PKCE)
# ─────────────────────────────────────────────────────────────────────────────
AUTHORITY = "https://login.microsoftonline.com/common"
GRAPH_URL = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.Read", "User.Read"]
REDIRECT_URI = "http://localhost:5050/auth/callback"

# Server-side state for the authenticated session
_graph_token = {"access_token": None, "expiry": None}
_graph_user = {"name": "", "email": ""}  # user info — stored server-side, not in session
_live_emails = []  # emails fetched from real Outlook
_pending_oauth = {}  # state -> {client_id, verifier, expires} — avoids session cookie issues in popup
_OAUTH_TIMEOUT = timedelta(minutes=10)  # pending auth requests expire after 10 min
_live_mode = False  # True when connected to real Outlook


def _generate_pkce():
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _graph_request(endpoint, params=None):
    if not _graph_token["access_token"]:
        raise ValueError("Not authenticated")
    if _graph_token["expiry"] and datetime.now() >= _graph_token["expiry"]:
        _graph_token["access_token"] = None
        _graph_token["expiry"] = None
        raise ValueError("Token expired")
    headers = {
        "Authorization": f"Bearer {_graph_token['access_token']}",
        "Content-Type": "application/json",
    }
    r = http_requests.get(f"{GRAPH_URL}/{endpoint}", headers=headers, params=params)
    if r.status_code == 401:
        _graph_token["access_token"] = None
        _graph_token["expiry"] = None
        raise ValueError("Token expired")
    r.raise_for_status()
    return r.json()


def _fetch_all_emails():
    """Fetch all inbox emails via pagination."""
    all_emails = []
    params = {
        "$top": 250,
        "$select": "id,subject,from,receivedDateTime,bodyPreview,body,isRead,internetMessageHeaders,hasAttachments",
        "$orderby": "receivedDateTime desc",
    }
    result = _graph_request("me/mailFolders/inbox/messages", params)
    all_emails.extend(result.get("value", []))

    # Follow @odata.nextLink until no more pages
    next_link = result.get("@odata.nextLink")
    while next_link:
        headers = {
            "Authorization": f"Bearer {_graph_token['access_token']}",
            "Content-Type": "application/json",
        }
        r = http_requests.get(next_link, headers=headers)
        if r.status_code == 401:
            _graph_token["access_token"] = None
            _graph_token["expiry"] = None
            raise ValueError("Token expired")
        r.raise_for_status()
        data = r.json()
        all_emails.extend(data.get("value", []))
        next_link = data.get("@odata.nextLink")

    return all_emails


# ── OAuth Routes ─────────────────────────────────────────────────────────────

@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Return current connection status."""
    connected = bool(_graph_token["access_token"] and _live_mode)
    return jsonify({
        "connected": connected,
        "live_mode": _live_mode,
        "user_name": _graph_user["name"],
        "user_email": _graph_user["email"],
    })


@app.route("/api/auth/connect", methods=["POST"])
def auth_connect():
    """Start OAuth flow — returns the Microsoft login URL."""
    global _live_mode
    if http_requests is None:
        return jsonify({"error": "requests library not installed"}), 500

    data = request.get_json(silent=True) or {}
    client_id = data.get("client_id", "").strip()
    if not client_id:
        return jsonify({"error": "Client ID required"}), 400

    # Store client_id and PKCE server-side (not in session — popup may lose cookie)
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    # Clean up expired pending requests
    now = datetime.now()
    expired = [k for k, v in _pending_oauth.items() if now >= v["expires"]]
    for k in expired:
        del _pending_oauth[k]

    _pending_oauth[state] = {
        "client_id": client_id,
        "verifier": verifier,
        "expires": now + _OAUTH_TIMEOUT,
    }

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "response_mode": "query",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{AUTHORITY}/oauth2/v2.0/authorize?{urlencode(params)}"
    return jsonify({"auth_url": auth_url})


@app.route("/auth/callback")
def auth_callback():
    """Handle Microsoft OAuth redirect callback."""
    global _graph_token, _live_emails, _live_mode

    error = request.args.get("error")
    if error:
        desc = request.args.get("error_description", error)
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Sign-in Failed</h2><p>{desc}</p>
        <p>You can close this tab.</p></body></html>"""

    received_state = request.args.get("state")
    pending = _pending_oauth.pop(received_state, None)
    if not pending or datetime.now() >= pending["expires"]:
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>Invalid or expired state parameter.</p>
        <p>You can close this tab.</p></body></html>"""

    code = request.args.get("code")
    if not code:
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>No authorization code received.</p>
        <p>You can close this tab.</p></body></html>"""

    # Exchange code for token
    client_id = pending["client_id"]
    verifier = pending["verifier"]
    token_url = f"{AUTHORITY}/oauth2/v2.0/token"
    token_data = {
        "client_id": client_id,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
        "scope": " ".join(SCOPES),
    }

    try:
        r = http_requests.post(token_url, data=token_data)
        if r.status_code != 200:
            err = r.json()
            desc = err.get("error_description", err.get("error", "Token exchange failed"))
            short = desc.split("\r\n")[0] if "\r\n" in desc else desc
            return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
            <h2>Sign-in Failed</h2><p>{short}</p>
            <p>You can close this tab.</p></body></html>"""

        tokens = r.json()
        access_token = tokens.get("access_token")
        expires_in = int(tokens.get("expires_in", 3600))

        # Temporarily set token so _graph_request works, but don't expose
        # to the poll yet (poll checks _graph_token to decide "connected")
        _graph_token["access_token"] = access_token
        _graph_token["expiry"] = datetime.now() + timedelta(seconds=expires_in)

        # Get user info
        try:
            user_info = _graph_request("me")
            _graph_user["name"] = user_info.get("displayName", "User")
            _graph_user["email"] = user_info.get("mail") or user_info.get("userPrincipalName", "")
        except Exception:
            _graph_user["name"] = "Connected"
            _graph_user["email"] = ""

        # Fetch emails before marking live mode — prevents race where
        # the main window poll sees "connected" but emails aren't loaded yet
        try:
            _live_emails = _fetch_all_emails()
        except Exception as e:
            _live_emails = []

        # Now mark live — the poll will see "connected" and loadEmails()
        # will return the real emails
        _live_mode = True

        # Clear any previous scan results
        scan_results.clear()

    except Exception as e:
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>{str(e)}</p>
        <p>You can close this tab.</p></body></html>"""

    return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
    <h2>Connected to Outlook</h2>
    <p>Signed in as {_graph_user['name'] or 'User'}</p>
    <p>You can close this tab and return to PhishGuard.</p>
    <script>window.close();</script></body></html>"""


@app.route("/api/auth/disconnect", methods=["POST"])
def auth_disconnect():
    """Disconnect from Outlook and switch back to mock emails."""
    global _graph_token, _live_emails, _live_mode
    _graph_token = {"access_token": None, "expiry": None}
    _graph_user["name"] = ""
    _graph_user["email"] = ""
    _live_emails = []
    _live_mode = False
    scan_results.clear()
    _pending_oauth.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/refresh", methods=["POST"])
def auth_refresh_emails():
    """Re-fetch emails from Outlook."""
    global _live_emails
    if not _live_mode or not _graph_token["access_token"]:
        return jsonify({"error": "Not connected"}), 401
    try:
        _live_emails = _fetch_all_emails()
        scan_results.clear()
        return jsonify({"count": len(_live_emails)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Patch email endpoints to support live mode ───────────────────────────────

def _get_email_list():
    """Return the active email list (live emails when connected, empty otherwise)."""
    return _live_emails if _live_mode else []


def _parse_auth_headers(internet_headers):
    """Parse internetMessageHeaders list into {spf, dkim, dmarc}."""
    result = {"spf": "none", "dkim": "none", "dmarc": "none"}
    if not internet_headers:
        return result
    for h in internet_headers:
        if h.get("name", "").lower() == "authentication-results":
            val = h.get("value", "").lower()
            for key in ("spf", "dkim", "dmarc"):
                if f"{key}=pass" in val:
                    result[key] = "pass"
                elif f"{key}=fail" in val or f"{key}=softfail" in val:
                    result[key] = "fail"
            break
    return result


def _normalize_email(email, idx):
    """Convert Graph API / mock email to flat format the frontend expects."""
    # Extract sender info
    sender = email.get("from", {})
    if "emailAddress" in sender:
        sender_name = sender["emailAddress"].get("name", "")
        sender_addr = sender["emailAddress"].get("address", "")
    else:
        sender_name = sender.get("name", "")
        sender_addr = sender.get("address", "")

    # Extract body text
    body_raw = email.get("body", {})
    if isinstance(body_raw, dict):
        body_content = body_raw.get("content", "")
    else:
        body_content = str(body_raw)
    body_text = _html_to_text(body_content) if body_content and "<" in body_content and ">" in body_content else (body_content or "")
    body_text = body_text.strip()

    # Parse auth headers
    headers = _parse_auth_headers(email.get("internetMessageHeaders"))

    return {
        "idx": idx,
        "subject": email.get("subject", "(No Subject)"),
        "sender_name": sender_name,
        "sender": sender_addr,
        "date": email.get("receivedDateTime", ""),
        "isRead": bool(email.get("isRead", True)),
        "bodyPreview": email.get("bodyPreview", ""),
        "body": body_text,
        "hasAttachments": email.get("hasAttachments", False),
        "attachments": email.get("attachments", []),
        "headers": headers,
        "scanned": idx in scan_results,
        "scanResult": scan_results.get(idx),
    }


@app.route("/api/emails", methods=["GET"])
def get_emails_v2():
    emails = _get_email_list()
    emails_summary = [_normalize_email(email, i) for i, email in enumerate(emails)]
    return jsonify({"emails": emails_summary})


@app.route("/api/emails/<int:idx>", methods=["GET"])
def get_email_v2(idx):
    emails = _get_email_list()
    if idx < 0 or idx >= len(emails):
        return jsonify({"error": "Email not found"}), 404
    return jsonify(_normalize_email(emails[idx], idx))


def _to_native(obj):
    """Recursively convert numpy/non-JSON types to native Python types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _make_serializable(url_analysis, header_result):
    """Convert all scan result data to JSON-serializable Python types."""
    return _to_native(url_analysis), _to_native(header_result)


@app.route("/api/scan/<int:idx>", methods=["POST"])
def scan_email_v2(idx):
    emails = _get_email_list()
    if idx < 0 or idx >= len(emails):
        return jsonify({"error": "Email not found"}), 404
    if not detector.is_trained:
        return jsonify({"error": "Model not loaded"}), 503

    email = emails[idx]
    subject = email.get("subject", "") or ""
    body_raw = email.get("body")
    if isinstance(body_raw, dict):
        body = body_raw.get("content", "") or ""
    elif isinstance(body_raw, str):
        body = body_raw
    else:
        body = email.get("bodyPreview", "") or ""
    body_t = _html_to_text(body) if body else ""
    full = f"Subject: {subject}\n\n{body_t}"

    try:
        prediction, confidence, url_analysis, header_result = detector.predict(
            full, headers=email.get("internetMessageHeaders"))
        url_analysis, header_result = _make_serializable(url_analysis, header_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    result = {
        "idx": idx,
        "prediction": _to_native(prediction),
        "confidence": _to_native(confidence),
        "url_analysis": url_analysis,
        "header_result": header_result,
    }
    scan_results[idx] = result
    return jsonify(result)


@app.route("/api/scan-all", methods=["POST"])
def scan_all_v2():
    emails = _get_email_list()
    if not detector.is_trained:
        return jsonify({"error": "Model not loaded"}), 503
    results = []
    for i, email in enumerate(emails):
        subject = email.get("subject", "") or ""
        body_raw = email.get("body")
        if isinstance(body_raw, dict):
            body = body_raw.get("content", "") or ""
        elif isinstance(body_raw, str):
            body = body_raw
        else:
            body = email.get("bodyPreview", "") or ""
        body_t = _html_to_text(body) if body else ""
        full = f"Subject: {subject}\n\n{body_t}"
        try:
            prediction, confidence, url_analysis, header_result = detector.predict(
                full, headers=email.get("internetMessageHeaders"))
            url_analysis, header_result = _make_serializable(url_analysis, header_result)
            result = {
                "idx": i,
                "prediction": int(prediction),
                "confidence": float(confidence),
                "url_analysis": url_analysis,
                "header_result": header_result,
            }
            scan_results[i] = result
            results.append(result)
        except Exception:
            pass
    results_dict = {str(r["idx"]): r for r in results}
    return jsonify({"results": results_dict})


@app.route("/api/stats", methods=["GET"])
def get_stats_v2():
    scanned = len(scan_results)
    threats = sum(1 for r in scan_results.values() if r.get("prediction") == 1)
    safe = scanned - threats
    return jsonify({
        "scanned": scanned,
        "threats": threats,
        "safe": safe,
        "total": len(_get_email_list()),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5050)
