# predict_email.py

import joblib

# Load the trained model and vectorizer
svm_model = joblib.load("svm_phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Phishing Email Detection")
print("Type 'exit' to quit\n")

while True:
    # Ask user to input email text
    email_text = input("Enter email text: ").strip()
    
    if email_text.lower() == "exit":
        print("Goodbye!")
        break
    
    # Remove punctuation (optional if you want consistency with training)
    import string
    email_text_clean = email_text.translate(str.maketrans("", "", string.punctuation))
    
    # Transform the text using the saved vectorizer
    X_input = vectorizer.transform([email_text_clean])
    
    # Predict using the trained model
    prediction = svm_model.predict(X_input)[0]
    
    # Display result
    if prediction == 1:
        print("This is likely a PHISHING email.\n")
    else:
        print("This is likely a LEGITIMATE email.\n")
