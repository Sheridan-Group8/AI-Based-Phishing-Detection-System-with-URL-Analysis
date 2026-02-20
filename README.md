# AI-Based Phishing Detection System with URL Analysis

## 📌 Capstone Project – Sheridan College  
**Group 8**  
**Date:** January 19, 2026  

### 👥 Team Members
- Aarambha Adhikari  
- Arvind Balusu  
- Aasiya Chauhan  
- Muhammad Yahya Khan  

---

## 🔐 Project Overview

Phishing remains one of the most damaging cybersecurity threats worldwide. Attackers use deceptive emails and malicious URLs to trick users into revealing sensitive information such as:

- login credentials  
- financial information  
- personal data  

Modern phishing attacks are increasingly sophisticated. Attackers now leverage AI to craft realistic and context-aware messages, making traditional rule-based filters ineffective.

This project proposes an **AI-based phishing detection system** that combines:

✅ Email content analysis  
✅ URL intelligence analysis  

to improve detection accuracy and reduce false positives.

---

## ❗ Problem Statement

Traditional phishing detection methods rely on:

- keyword filtering  
- rule-based detection  
- static blacklists  

These approaches fail against:

- AI-generated phishing emails  
- social engineering tactics  
- newly registered malicious domains  

An intelligent detection system is required to analyze **context, language patterns, and URL characteristics**.

---

## 💡 Proposed Solution

The system combines machine learning and security intelligence to detect phishing attempts through two components:

---

### 📧 1. Email Phishing Detection

Machine learning models will analyze email content to classify messages as **legitimate** or **phishing**.

#### 📊 Datasets
- Public phishing email datasets  
- Enron Email Dataset (legitimate corporate communication)

#### 🤖 Models
- **Naive Bayes** — baseline classification  
- **BERT / DistilBERT** — contextual language understanding  
  - urgency language detection  
  - impersonation patterns  
  - social engineering tactics  

#### 🔗 Integration
- Microsoft Graph API for secure retrieval of Microsoft 365 emails

---

### 🌐 2. AI-Based URL Analysis

Phishing emails often include malicious links. The system evaluates URLs using structural and domain intelligence.

#### 🔎 URL Structure Analysis
- URL length and complexity  
- use of special characters and hyphens  
- multiple subdomains  
- suspicious keywords  

#### 🌍 Domain & Infrastructure Analysis
- domain age and registration history  
- DNS and SSL certificate details  
- hosting provider & reputation indicators  

#### 🧪 Future Enhancement (Optional)
- Execute suspicious URLs in a **sandbox environment** to observe behavior safely.

---

## 🎯 Project Impact

The combined analysis approach improves detection accuracy and reduces false positives.

### Benefits to Organizations
- Detect phishing emails more effectively  
- Prevent credential theft and financial fraud  
- Enhance cloud email security  
- Protect users from social engineering attacks  

---

## ⚙️ Feasibility

This project is highly feasible due to:

- publicly available labeled datasets  
- mature machine learning frameworks  
- accessible APIs for email and domain analysis  
- extensive research and documentation  

---

## 🛠️ Technologies & Tools

**Languages & Frameworks**
- Python
- Scikit-learn
- TensorFlow / PyTorch
- Hugging Face Transformers

**Security & Analysis**
- Microsoft Graph API
- WHOIS & DNS lookup tools
- SSL certificate inspection

**Development Environment**
- GitHub
- VS Code / GitHub Codespaces

---

## 🚀 Future Improvements

- real-time email monitoring  
- browser extension for phishing warnings  
- sandbox detonation for malicious URLs  
- integration with Microsoft Defender & SIEM systems  
- model performance optimization & threat intelligence feeds  

---

## 📚 References

- Cleanfox — AI Phishing Detection  
  https://blog.cleanfox.io/ai-phishing-detection-how-artificial-intelligence-is-changing-email-security/

- H2O.ai — BERT  
  https://h2o.ai/wiki/bert/

- H2O.ai — Naive Bayes  
  https://h2o.ai/wiki/naive-bayes/

- Intezer — URL Analysis for Phishing  
  https://intezer.com/blog/url-analysis-phishing-part-1/

- Insights2TechInfo — Naive Bayes in Phishing Detection  
  https://insights2techinfo.com/unveiling-the-power-of-naive-bayes-in-phishing-detection/

---

## 📎 Repository

GitHub Repository:  
https://github.com/Sheridan-Group8/AI-Based-Phishing-Detection-System-with-URL-Analysis

---

## 📜 License

This project is developed for academic purposes at Sheridan College.
