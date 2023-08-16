import re
import smtplib
import email
from email.parser import Parser
from email.header import decode_header
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set up your email credentials
email_address = 'parthonabar10203@gmail.com'
email_password = 'P.n2021@4918'

# Load the trained NLP model
vectorizer = CountVectorizer()
nlp_model = MultinomialNB()
# Train the model with your dataset containing harmful and non-harmful emails
# X_train: list of preprocessed email texts
# y_train: corresponding labels (1 for harmful, 0 for non-harmful)
X_train = [
    "This is a harmful email about password reset.",
    "Please be aware of the phishing attempt in this email.",
    "We have detected fraudulent activity in your account.",
    "This email is a scam, do not respond to it."
]
y_train = [1, 1, 1, 1]
X_train_counts = vectorizer.fit_transform(X_train)
nlp_model.fit(X_train_counts, y_train)

# Function to extract email text from message
def extract_text_from_email(msg):
    if msg.is_multipart():
        return extract_text_from_email(msg.get_payload(0))
    else:
        return msg.get_payload(decode=True).decode('utf-8')  # Decode the bytes-like object to string

# Function to preprocess and tokenize text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Function to report email and IP address
def report_email(sender_ip):
    subject = 'Suspicious Email Detected'
    body = f'Suspicious email received from IP: {sender_ip}'
    
    # Send the report via email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.starttls()
        smtp.login(email_address, email_password)
        msg = f'Subject: {subject}\n\n{body}'
        smtp.sendmail(email_address, 'writeto.partho23@gmail.com', msg)
    
    print('Email reported:', email_content)  # Print a message indicating that the email has been reported

# Function to process the received email
def process_email(email_content):
    msg = Parser().parsestr(email_content)
    text = extract_text_from_email(msg)
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    preprocessed_text_counts = vectorizer.transform([preprocessed_text])

    # Predict the label using the NLP model
    predicted_label = nlp_model.predict(preprocessed_text_counts)

    print('Email:', email_content)  # Print the email content for debugging

    if predicted_label == 1:
        print('Label: Harmful')
        # Get sender IP address from email headers
        if 'Received' in msg:
            received_headers = decode_header(msg['Received'])
            if received_headers is not None:
                received_headers = [header for header in received_headers if header[0] is not None]
                for received_header in received_headers:
                    if isinstance(received_header[0], bytes):
                        received_header = received_header[0].decode('utf-8', errors='replace')
                    if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', received_header):
                        sender_ip = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', received_header).group()
                        print('Sender IP:', sender_ip)  # Print the extracted sender's IP address
                        report_email(sender_ip)
                        break
    else:
        print('Label: Non-Harmful')

# Example usage
received_email = """
Received: from mail.example.com (mail.example.com [127.0.0.1])
    by smtp.example.com (Postfix) with ESMTP id ABCDEFG
    for <recipient@example.com>; Mon, 1 Jul 2023 12:34:56 -0400 (EDT)
From: sender@example.com
Subject: Important: Password Reset
Date: Mon, 1 Jul 2023 12:34:56 -0400 (EDT)

Dear user,

Please reset your password by clicking on the following link: example.com/reset
"""

process_email(received_email)