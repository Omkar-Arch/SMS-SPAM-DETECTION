import os

print("Current Directory:", os.getcwd())


import streamlit as st
import pickle
import nltk
import string

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Page Config

st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="📧",
    layout="centered"
)


# Load Models

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(
    open(os.path.join(BASE_DIR, "model.pkl"), "rb")
)

cv = pickle.load(
    open(os.path.join(BASE_DIR, "count_vectorizer.pkl"), "rb")
)

tfidf = pickle.load(
    open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb")
)

# NLP Setup

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []

    for word in message:
        if word.isalnum():
            y.append(word)

    filtered = []

    for word in y:
        if word not in stop_words:
            filtered.append(ps.stem(word))

    return " ".join(filtered)

# Header

st.title("📧 Spam SMS Detector")

st.markdown(
    """
    Detect whether a message is **Spam** or **Not Spam**
    using Machine Learning (Naive Bayes + TF-IDF).
    """
)


# Input Box

message = st.text_area(
    "Enter Message",
    height=180,
    placeholder="Type or paste your message here..."
)


# Prediction

if st.button("Predict", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        # preprocessing
        transformed = transform_message(message)

        # vectorization
        vector = cv.transform([transformed])
        vector = tfidf.transform(vector)

        # prediction
        prediction = model.predict(vector)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")


# Footer

st.divider()

st.caption(
    "Built using Python, NLTK, Scikit-Learn and Streamlit"
)