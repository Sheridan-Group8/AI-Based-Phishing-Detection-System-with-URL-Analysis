# PhishGuard

**AI-Based Phishing Detection System**  
**Sheridan College Capstone Project - Group 8**  
**Current status:** Week 8 development

## Team Members

- Aarambha Adhikari
- Arvind Balusu
- Aasiya Chauhan
- Muhammad Yahya Khan

## Project Overview

PhishGuard is a desktop phishing-detection application for Microsoft Outlook email. It combines a local AI model, URL risk analysis, sender authentication checks, attachment analysis, and threat-intelligence signals into one unified scan assessment.

The application is built as an Electron desktop app with a Flask backend. Outlook email access is handled through Microsoft Graph, while scan history and user-linked data can be synchronized through Supabase when the user is signed in.

## Problem Statement

Phishing emails continue to bypass traditional spam filters by using convincing language, spoofed senders, malicious links, and risky attachments. Users need a tool that can inspect message content, hidden links, sender authenticity, and external threat signals before they interact with suspicious email.

## Current Solution

PhishGuard analyzes Outlook messages and produces a risk-based verdict for each scan. The current system evaluates:

- Email text and subject content
- Visible and hidden HTML links from `href`, `src`, and `action` attributes
- URL structure, reputation, and ONNX URL-model scoring
- Sender authentication using SPF, DKIM, DMARC, domain alignment, Reply-To mismatch, and provider signals
- Sender reputation and originating sender IP reputation
- Attachments using file-type rules, hash lookups, and optional VirusTotal deep scanning
- Sender DNA behavior, including writing style, timing, greeting style, and signature patterns

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
- Rich link scoring pipeline using structural analysis, threat intel, and URL-model scoring
- Threat intelligence support for Google Safe Browsing, VirusTotal URLs, urlscan.io, URLhaus, and AbuseIPDB
- VirusTotal attachment hash lookup, upload, and polling support
- Microsoft profile photo retrieval through Graph
- Sender DNA profiling and comparison
- Local scan history metadata
- Optional Supabase-backed scan history, reports, and sender profiles
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

## Current Development Status - Week 8

Completed in the current version:

- Desktop UI integrated with Electron
- Flask backend integrated with the renderer
- Microsoft Graph sign-in and Outlook email retrieval
- Local ONNX text model and ONNX URL model loading
- Unified scan assessment with `risk_score` and dimension-level analysis
- Hidden HTML URL extraction during scans
- Threat-intel link scoring pipeline
- Full sender authentication analysis
- Sender DNA profiling
- Attachment risk scoring and VirusTotal deep scan support
- Microsoft profile photo support
- Scan history metadata persistence
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

## Project Status

PhishGuard is in active Week 8 capstone development. The current build is focused on the integrated desktop experience, local AI scanning, threat-intelligence-backed risk assessment, sender profiling, attachment analysis, and secure Outlook workflow.
