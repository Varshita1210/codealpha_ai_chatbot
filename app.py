from flask import Flask, render_template, request, jsonify
from faq_data import faq_data

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

questions = list(faq_data.keys())
answers = list(faq_data.values())

stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return " ".join(filtered_tokens)

# Preprocess all FAQ questions
processed_questions = [preprocess(q) for q in questions]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():

    user_input = request.form["message"]

    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarity.argmax()

    best_score = similarity[0][best_match_index]

    if best_score > 0.3:
        response = answers[best_match_index]
    else:
        response = "Sorry, I could not understand your question."

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)