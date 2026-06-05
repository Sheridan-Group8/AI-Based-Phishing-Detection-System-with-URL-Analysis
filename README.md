# AI-Based Phishing Detection System with Dual-Model Analysis

**Capstone Project – Sheridan College**
**Group 8**
**Date:** 06 June 2026 (Week 5 in Phase 2)

---

##  Team Members

* Aarambha Adhikari
* Arvind Balusu
* Aasiya Chauhan
* Muhammad Yahya Khan

---

# Project Overview

PhishGuard is an AI-powered desktop cybersecurity application designed to detect phishing emails from Microsoft Outlook. The system combines machine learning, URL intelligence, email authentication analysis, and Microsoft Graph integration to provide comprehensive phishing detection and risk assessment.

The platform analyzes email content, embedded URLs, sender authenticity, and email structure to identify malicious messages that may bypass traditional spam filters.

PhishGuard is currently in Phase 2 development, evolving from a proof-of-concept into a full-featured desktop application built with Electron, Flask, Supabase, and advanced machine learning models.

---

# Problem Statement

Phishing attacks remain one of the most common and effective cyber threats. Attackers frequently impersonate trusted organizations and use deceptive emails, malicious links, and spoofed sender identities to steal credentials and sensitive information.

Traditional email filtering solutions often rely on signature-based detection and may fail to identify sophisticated phishing campaigns. Users therefore require additional protection that analyzes email content, URLs, authentication records, and behavioral indicators before interacting with suspicious messages.

---

# Proposed Solution

PhishGuard provides a multi-layered phishing detection framework that:

* Detects phishing emails using machine learning models
* Analyzes embedded URLs for phishing indicators
* Identifies brand impersonation attempts
* Detects suspicious domains, shortened URLs, and IP-based links
* Validates sender authenticity using SPF, DKIM, and DMARC
* Evaluates structural and behavioral characteristics of emails
* Stores scan history and security events in Supabase
* Integrates directly with Microsoft Outlook using Microsoft Graph API
* Presents results through an intuitive desktop dashboard

---

# Detection Architecture

## Text Analysis Model

The primary email classifier uses:

* TF-IDF Vectorization (10,000 features)
* Unigrams and Bigrams
* Multinomial Naive Bayes Classification
* Structural Email Feature Analysis

### Structural Features

The system evaluates:

* Capitalization ratio
* Exclamation density
* Question mark density
* Urgency language
* Financial/reward terminology
* Threat-related wording
* Email length
* Line count
* Special character density
* Numeric content ratio
* HTML presence
* Link-to-text ratio

---

## URL Intelligence Model

A dedicated URL classifier independently analyzes URLs extracted from emails.

### URL Risk Indicators

The system detects:

* Suspicious top-level domains (.xyz, .top, .click, etc.)
* URL shorteners
* Brand impersonation
* IP-address-based URLs
* Suspicious keywords
* Excessive subdomains
* Non-standard ports
* Redirect manipulation techniques
* Long and obfuscated URLs

### URL Classification

* Logistic Regression Model
* Trained on 108,000+ phishing and legitimate URLs
* Separate from the email content classifier
* Combined with text classification during final scoring

---

## Email Authentication Analysis

PhishGuard validates sender authenticity using:

### SPF (Sender Policy Framework)

Verifies whether the sender is authorized to send email on behalf of the domain.

### DKIM (DomainKeys Identified Mail)

Validates cryptographic signatures attached to emails.

### DMARC (Domain-based Message Authentication, Reporting & Conformance)

Ensures alignment between SPF and DKIM validation results.

Authentication results are incorporated into the final phishing confidence score.

---

# Dual-Model Scoring System

PhishGuard uses a weighted scoring approach:

Final Risk Score =

* 55% Text Model Score
* 45% URL Model Score

Authentication results from SPF, DKIM, and DMARC can increase or decrease the final risk assessment.

This layered approach improves detection accuracy and reduces false positives compared to relying solely on email text analysis.

---

# Training Datasets

The phishing detection models are trained using multiple publicly available datasets.

### Email Datasets

* Kaggle Phishing Email Dataset
* Hugging Face Phishing Email Dataset
* CybersecTony PhishingEmailDetection v2.0

### URL Datasets

* CybersecTony PhishingEmailDetection v2.0 URL Dataset
* Shawhin Phishing Site Classification Dataset

Combined URL training data contains more than 108,000 labeled URLs.

---

# Technology Stack

## Frontend

* Electron
* HTML
* CSS
* JavaScript

## Backend

* Flask (Python)

## Database

* Supabase

## Microsoft Integration

* Microsoft Graph API
* Outlook Mail Access
* Microsoft Authentication

## Machine Learning

* Scikit-Learn
* TF-IDF Vectorization
* Multinomial Naive Bayes
* Logistic Regression
* NumPy
* Pandas

---

# Current Development Status (Week 5 – Phase 2)

### Completed

✓ Electron desktop application integration

✓ Flask backend integration

✓ Supabase database implementation

✓ Microsoft Graph API integration

✓ Outlook email retrieval

✓ Dual-model phishing detection architecture

✓ Advanced URL intelligence engine

✓ SPF/DKIM/DMARC validation module

✓ Structural email analysis module

✓ Persistent scan history storage

✓ Enhanced phishing confidence scoring

---

### In Progress

🔄 Dashboard UI/UX enhancements

🔄 Model performance improvements


---

# Database Connection Note

Database synchronization and Outlook access are handled through the Microsoft sign-in flow.

When PhishGuard launches:

1. Sign in with your Microsoft account.
2. Outlook emails are retrieved through Microsoft Graph.
3. Scan history is synchronized with Supabase.
4. Phishing analysis results become available in the dashboard.

---

# Future Enhancements

* Attachment malware analysis
* Sender reputation scoring
* Threat intelligence integration
* Advanced reporting and analytics


---

# Project Status

**Status:** Active Development (On Track)

This project is being developed for academic purposes as part of the Sheridan College Capstone Project.
