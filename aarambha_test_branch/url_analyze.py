"""
URL Phishing Detector using Random Forest
With PhishTank Reputation Scoring
"""

import re
import pandas as pd
import numpy as np
import whois
import tldextract

from datetime import datetime
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix


# =============================================
# URL STRUCTURAL ANALYZER
# =============================================

class URLAnalyzer:

    SUSPICIOUS_TLDS = {
        ".xyz",".top",".club",".work",".click",".link",".gq",".ml",
        ".cf",".tk",".ga",".pw",".cc",".buzz",".info",".biz",".ru"
    }

    SUSPICIOUS_KEYWORDS = {
        "login","signin","verify","secure","account","update",
        "confirm","password","credential","authenticate",
        "suspend","locked","urgent","expire","billing","payment"
    }

    def analyze(self,url):

        features = {
            "url_length":len(url),
            "num_dots":url.count("."),
            "num_digits":sum(c.isdigit() for c in url),
            "num_hyphens":url.count("-"),
            "has_at_symbol":1 if "@" in url else 0,
            "has_ip":0,
            "suspicious_tld":0,
            "keyword":0
        }

        if not url.startswith(("http://","https://")):
            url="http://"+url

        parsed=urlparse(url)
        domain=parsed.netloc.lower()

        if re.search(r"\d+\.\d+\.\d+\.\d+",domain):
            features["has_ip"]=1

        for tld in self.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                features["suspicious_tld"]=1

        for word in self.SUSPICIOUS_KEYWORDS:
            if word in url.lower():
                features["keyword"]=1

        return np.array(list(features.values()))


# =============================================
# DOMAIN REPUTATION (PHISHTANK)
# =============================================

class DomainReputationAnalyzer:

    def __init__(self, phishtank_file="phishtank.csv"):

        self.bad_urls=set()
        self.load_phishtank(phishtank_file)

    def load_phishtank(self,file):

        try:
            df=pd.read_csv(file)

            if "url" in df.columns:
                self.bad_urls=set(df["url"].str.lower())
            elif "phish_url" in df.columns:
                self.bad_urls=set(df["phish_url"].str.lower())

            print("Loaded",len(self.bad_urls),"PhishTank URLs")

        except Exception as e:
            print("PhishTank not loaded:",e)

    def extract_domain(self,url):

        try:
            ext=tldextract.extract(url)
            return f"{ext.domain}.{ext.suffix}"
        except:
            return None

    def domain_age(self,domain):

        try:
            w=whois.whois(domain)
            creation=w.creation_date

            if isinstance(creation,list):
                creation=creation[0]

            if not creation:
                return None

            return (datetime.now()-creation).days

        except:
            return None

    def score(self,url):

        result={
            "listed_phishtank":False,
            "domain_age":None,
            "score":0
        }

        url=url.lower()

        # PhishTank blacklist
        if url in self.bad_urls:
            result["listed_phishtank"]=True
            result["score"]-=0.9
            return result

        domain=self.extract_domain(url)

        age=self.domain_age(domain)

        result["domain_age"]=age

        if age:

            if age<30:
                result["score"]-=0.4
            elif age<90:
                result["score"]-=0.2
            else:
                result["score"]+=0.1

        return result


# =============================================
# MAIN DETECTOR
# =============================================

class URLPhishingDetector:

    def __init__(self):

        self.vectorizer=TfidfVectorizer(
            analyzer="char",
            ngram_range=(2,5),
            max_features=5000
        )

        self.model=RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )

        self.analyzer=URLAnalyzer()
        self.reputation=DomainReputationAnalyzer()

        self.trained=False


    def preprocess(self,url):

        url=url.lower()

        url=re.sub(r"https?://"," ",url)
        url=re.sub(r"[^a-z0-9]"," ",url)

        return url


    def extract_features(self,urls,fit=False):

        text=[self.preprocess(u) for u in urls]

        if fit:
            text_features=self.vectorizer.fit_transform(text)
        else:
            text_features=self.vectorizer.transform(text)

        url_features=[self.analyzer.analyze(u) for u in urls]

        url_features=csr_matrix(np.array(url_features))

        X=hstack([text_features,url_features])

        return X


    def train(self,urls,labels):

        X=self.extract_features(urls,fit=True)

        self.model.fit(X,labels)

        self.trained=True

        print("Model trained successfully")
        print("Feature dimension:",X.shape)


    def predict(self,url):

        if not self.trained:
            raise ValueError("Model not trained")

        X=self.extract_features([url],fit=False)

        prediction=self.model.predict(X)[0]

        prob=self.model.predict_proba(X)[0]

        confidence=max(prob)

        # reputation adjustment
        rep=self.reputation.score(url)

        adjusted_conf=confidence+(-rep["score"]*0.3)

        adjusted_conf=max(0,min(1,adjusted_conf))

        return prediction,adjusted_conf,rep


# =============================================
# MAIN PROGRAM
# =============================================

def main():

    print("\nURL Phishing Detector (Random Forest)")
    print("="*45)

    detector=URLPhishingDetector()

    urls=[
        "paypal-secure-login.xyz/verify",
        "bank-account-update.top/login",
        "secure-amazon-confirm.click",
        "microsoft-support-login.ru/reset",
        "google.com",
        "github.com/login",
        "wikipedia.org/wiki/python",
        "amazon.com/product"
    ]

    labels=[1,1,1,1,0,0,0,0]

    detector.train(urls,labels)

    while True:

        url=input("\nEnter URL (or quit): ").strip()

        if url.lower() in ["quit","exit"]:
            break

        pred,conf,rep=detector.predict(url)

        if pred==1:
            print(f"⚠ PHISHING ({conf:.2%} confidence)")
        else:
            print(f"✓ SAFE ({conf:.2%} confidence)")

        print("Reputation:",rep)


if __name__=="__main__":
    main()
