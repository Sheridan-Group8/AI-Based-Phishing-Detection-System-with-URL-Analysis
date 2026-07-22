# PhishGuard

**AI-Based Phishing Detection System**  
**Sheridan College Capstone Project - Group 8**  
**Current status:** Week 10 development

## Team Members

- Aarambha Adhikari
- Arvind Balusu
- Aasiya Chauhan
- Muhammad Yahya Khan

## Project Overview

PhishGuard is a desktop phishing-detection application for Microsoft Outlook email. It combines calibrated local AI models, URL risk analysis, sender authentication checks, attachment analysis, configurable detection rules, and threat-intelligence signals into one unified scan assessment.

The application is built as an Electron desktop app with a Flask backend. Outlook email access is handled through Microsoft Graph, while scan history and user-linked data can be synchronized through Supabase when the user is signed in.

## Problem Statement

Phishing emails continue to bypass traditional spam filters by using convincing language, spoofed senders, malicious links, and risky attachments. Users need a tool that can inspect message content, hidden links, sender authenticity, and external threat signals before they interact with suspicious email.

## Current Solution

PhishGuard analyzes Outlook messages and produces a risk-based verdict for each scan. The current system evaluates:

- Email text and subject content
- Visible and hidden HTML links from `href`, `src`, and `action` attributes
- URL structure, reputation, and ONNX URL-model scoring
- RDAP-based domain-age checks for suspicious sender and link domains
- Sender authentication using SPF, DKIM, DMARC, domain alignment, Reply-To mismatch, and provider signals
- Sender reputation and originating sender IP reputation
- Attachments using file-type rules, hash lookups, and optional VirusTotal deep scanning
- Sender DNA behavior, including writing style, timing, greeting style, and signature patterns
- Exact URL matches from threat-intelligence feeds such as OpenPhish
- Brand impersonation signals, including deceptive link text, look-alike domains, and display-name spoofing

Scan results are returned as a unified assessment with:

- `risk_score`
- overall verdict
- dimension bars for content, links, sender, authentication, and files
- threat cards
- recommended user actions
- scan-history metadata

## Key Features

- Electron desktop dashboard
- Flask API backend
- Microsoft Outlook email retrieval through Microsoft Graph
- Local ONNX email classification model
- Local ONNX URL classification model
- Calibrated shipped ONNX thresholds with precision-focused metadata
- SHA-256 integrity checks for shipped ONNX model artifacts
- Rich link scoring pipeline using structural analysis, threat intel, and URL-model scoring
- Threat intelligence support for Google Safe Browsing, VirusTotal URLs, urlscan.io, URLhaus, AbuseIPDB, and OpenPhish
- Config-backed detection rules in `PHISHGUARD/config/detection_rules.json`
- Renderer-accessible detection-rule endpoint at `/api/detection-rules`
- RDAP domain-age checks without the old optional `whois` dependency
- Brand look-alike, homoglyph, deceptive-link, and display-name spoofing detection
- VirusTotal attachment hash lookup, upload, and polling support
- Microsoft profile photo retrieval through Graph
- Sender DNA profiling and comparison
- Local scan history metadata
- Optional Supabase-backed scan history, reports, and sender profiles
- Reconnect-to-Outlook banner when signed in but Outlook is disconnected
- Dashboard insights for threat breakdown, common threats, and last scan timing
- Compact email-list preference for denser triage
- CSRF, origin, request-size, launch-secret, and local data-directory hardening

## Technology Stack

- Electron
- HTML, CSS, JavaScript
- Python Flask
- Microsoft Graph API
- Supabase
- ONNX Runtime
- Tokenizers
- VirusTotal API
- Google Safe Browsing API
- urlscan.io API
- URLhaus API
- AbuseIPDB API
- OpenPhish community feed
- RDAP.org domain data

## Current Development Status - Week 10

Completed in the current version:

- Desktop UI integrated with Electron
- Flask backend integrated with the renderer
- Microsoft Graph sign-in and Outlook email retrieval
- Local ONNX text model and ONNX URL model loading
- ONNX threshold recalibration:
  - Text model threshold updated to `0.89`
  - URL model threshold updated to `0.92`
  - Calibration metadata records eval files, precision target, selected policy, precision, recall, false positives, and false negatives
- ONNX model SHA-256 verification before model loading
- Config-driven detection rules for brand domains, risky TLDs, credential-lure phrases, URL shorteners, and trusted domains
- `/api/detection-rules` endpoint for frontend detection-rule awareness
- Unified scan assessment with `risk_score` and dimension-level analysis
- Hidden HTML URL extraction during scans
- Threat-intel link scoring pipeline
- OpenPhish feed ingestion with exact-URL matching
- RDAP domain-age checks replacing the old optional WHOIS lookup path
- Stronger brand impersonation detection for deceptive links, look-alike domains, display-name spoofing, and suspicious sender domains
- Full sender authentication analysis
- Sender DNA profiling
- Attachment risk scoring and VirusTotal deep scan support
- Microsoft profile photo support
- Scan history metadata persistence
- Dashboard insights including threat breakdown, common threats, and last scan text
- Reconnect-to-Outlook banner for signed-in users whose Outlook inbox is disconnected
- Compact email-list setting
- Repository ignore rules scoped so new root-level files outside `PHISHGUARD/` are not tracked by default
- Security hardening around request origins, CSRF, launch secrets, writable data directories, and request limits

## Getting the Project Running

### 1. Install Prerequisites

Install these before cloning or running the app:

- Git
- Git LFS
- Python 3.11 or newer
- Node.js and npm

### 2. Clone the Repository

```bash
git clone <repository-url>
cd REPO
```

### 3. Pull Large Model Files

The ONNX model files are stored with Git LFS.

```bash
git lfs install
git lfs pull
```

### 4. Install Python Dependencies

From the `PHISHGUARD` folder, install the required Python packages:

```bash
cd PHISHGUARD
pip install -r requirements.txt
```

### 5. Install Desktop App Dependencies

From the `PHISHGUARD` folder:

```bash
npm install
```

### 6. Optional API Keys

The app can run without these keys, but the related threat-intelligence features will be limited. Create `PHISHGUARD/.env` if you want to enable them:

```text
VIRUSTOTAL_API_KEY=your_key_here
GOOGLE_SAFE_BROWSING_KEY=your_key_here
URLSCAN_API_KEY=your_key_here
ABUSEIPDB_KEY=your_key_here
```

OpenPhish community-feed checks and RDAP domain-age checks do not require an API key.

### 7. Start the App

From the `PHISHGUARD` folder:

```bash
npm start
```

Electron starts the Flask backend automatically, opens the PhishGuard desktop interface, and loads the local ONNX models. After the app opens, sign in with Microsoft to retrieve Outlook mail and begin scanning messages.

## Build Commands

From the `PHISHGUARD` folder:

```bash
npm run build
npm run build:win
npm run build:mac
npm run build:linux
```

For Windows, `npm run build:win` creates a one-click NSIS installer at:

```text
PHISHGUARD/dist/PhishGuard Setup 2.0.0.exe
```

The Windows installer bundles the Flask backend with PyInstaller, including the
Python runtime dependencies and the local quantized ONNX models. End users do
not need to install Python separately.

## Project Status

PhishGuard is in active Week 10 capstone development. The current build is focused on calibrated local AI scanning, model integrity, configurable detection rules, threat-intelligence-backed risk assessment, sender profiling, attachment analysis, dashboard usability, and secure Outlook workflow.
