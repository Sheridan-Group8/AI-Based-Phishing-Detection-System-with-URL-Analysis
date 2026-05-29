# AI-Based Phishing Detection System with URL Analysis

## Capstone Project – Sheridan College  
**Group 8**  
**Date:** 21 May, 2026  

### 👥 Team Members
- Aarambha Adhikari  
- Arvind Balusu  
- Aasiya Chauhan  
- Muhammad Yahya Khan  

---

## Project Overview
PhishGuard is a desktop-based cybersecurity application that detects phishing emails from Microsoft Outlook. It integrates Microsoft Graph API, machine learning models, URL analysis, and email header validation to classify emails and highlight potential security threats.

The project is currently in Phase 2 development, transitioning from a proof-of-concept system to a full desktop application with persistent database storage and improved detection accuracy.


## Problem Statement

Phishing emails are one of the most common cybersecurity threats, often bypassing basic email filters and targeting users through deceptive links and spoofed sender identities. Existing email clients provide limited deep analysis of URLs, sender behavior, and email structure, making users vulnerable to attacks.


## Proposed Solution

PhishGuard addresses this issue by providing an intelligent phishing detection system that:

Analyzes email content using machine learning models
Inspects URLs using lexical and domain-based features
Validates email authenticity using SPF, DKIM, and DMARC checks
Stores scan history and sender data in a structured database
Presents results through a clear desktop dashboard for user action

## Current Status (Week 4 – Phase 2)
Database schema designed and implemented in Supabase
Tables created for scan history, sender data, and logs
Supabase database connection added to the PhishGuard dashboard
Existing dashboard integrated into a basic Electron desktop window
Local Electron startup tested with Flask backend running on localhost
System architecture updated to Electron + Flask + Supabase
UI/UX flow continues to be improved for the desktop dashboard

## Database Connection Note
For now, database connection is handled from inside the app. Open PhishGuard, go to Settings, and connect to the database from there.

This process is temporary. A future update will simplify the database connection flow so users do not need to manually connect through Settings.

## In Progress
- ML/AI model accuracy improvements
- Login and authentication flow refinement
- URL and domain analysis enhancements
- User interface and dashboard usability improvements

## Status: In Progress (On Track)




## This project is developed for academic purposes at Sheridan College.
