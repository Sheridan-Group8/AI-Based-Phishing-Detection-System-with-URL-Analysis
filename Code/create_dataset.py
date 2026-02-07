import pandas as pd

data = {
    "email_text": [
        "Dear user your account has been suspended Click the link below to verify your information",
        "Urgent action required Your bank account has been compromised Verify immediately",
        "Congratulations You have won a 1000 gift card Click here to claim now",
        "Your password will expire today Confirm your credentials to avoid service interruption",
        "We detected unusual login activity Please update your security details",
        "Invoice attached for your recent purchase Please review and confirm payment",
        "Hi team the meeting is scheduled for Monday at 10 AM Let me know if you can attend",
        "Your order has been shipped and will arrive in 3 to 5 business days",
        "Can we reschedule our call to tomorrow afternoon",
        "Please find the project report attached Feedback is welcome",
        "Reminder submit your timesheet before Friday",
        "Thank you for your payment This email confirms your transaction"
    ],
    "label": [1,1,1,1,1,0,0,0,0,0,0,0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("phishing_emails_clean_no_punct.csv", index=False)

print("CSV created successfully!")