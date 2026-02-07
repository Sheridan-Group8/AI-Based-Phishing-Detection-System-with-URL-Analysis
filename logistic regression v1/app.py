import pickle

# Load trained model
with open("model/email_classifier.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

print("Email Phishing Detection System")
print("--------------------------------")

email_text = input("Enter email text: ")

email_tfidf = vectorizer.transform([email_text])

prediction = model.predict(email_tfidf)[0]

if prediction == 1:
    print("⚠️  PHISHING EMAIL DETECTED")
else:
    print("✅ LEGITIMATE EMAIL")
