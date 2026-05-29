# PhishGuard – AI-Based Phishing Detection System (POC)

## Overview

PhishGuard is an AI-powered phishing detection platform focused on Outlook email security. The system connects to Microsoft Outlook using Microsoft Graph API, analyzes emails and URLs using machine learning and security checks, and displays phishing detection results through a web dashboard.

This Proof of Concept (POC) demonstrates phishing detection, URL analysis, sender verification, and email security reporting in a single application.

---

# Features

* Outlook email integration using Microsoft Graph API
* AI-based phishing email detection
* URL and sender analysis
* Email header inspection (SPF, DKIM, DMARC)
* Real-time phishing detection dashboard
* Machine learning-based classification
* Local scan history and logging

---

# Technology Stack

## Frontend

* HTML
* CSS
* JavaScript

## Backend

* Python
* Flask

## Machine Learning

* scikit-learn
* TF-IDF
* Naive Bayes
* Logistic Regression

## APIs & Security

* Microsoft Graph API
* OAuth 2.0
* URL reputation and lexical analysis

---

# Project Structure

```bash
PhishGuard/
│
├── app.py
├── phishing_detector.py
├── phishing_model.pkl
├── static/
├── templates/
└── README.md
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/Sheridan-Group8/AI-Based-Phishing-Detection-System-with-URL-Analysis.git
```

## Navigate to the Project Folder

```bash
cd AI-Based-Phishing-Detection-System-with-URL-Analysis
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the Flask server:

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

# Current POC Capabilities

* Microsoft Graph API integration
* Basic phishing detection model
* URL analysis
* Email header analysis
* Dashboard interface
* Local logging and scan tracking

---

# Future Improvements

* Electron desktop application
* Supabase/PostgreSQL integration
* Improved ML accuracy
* Advanced sender profiling
* Better reporting and analytics
* Windows installer packaging

---

# Team Members

Sheridan Group 8

### Team A

* Aarambha
* Yahya

### Team B

* Aasiya
* Arvind

---

# Disclaimer

This project is developed for educational and cybersecurity research purposes only.
