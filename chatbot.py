from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLP resources
nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)

# -----------------------------
# Load intents
# -----------------------------
with open("intents.json", "r") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# -----------------------------
# NLP Vectorization & Training
# -----------------------------
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(patterns)

model = LogisticRegression(max_iter=1000)
model.fit(X, tags)

# -----------------------------
# NLP Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# -----------------------------
# Chatbot Response Logic
# -----------------------------
def get_bot_response(user_input):
    user_input = clean_text(user_input)

    input_vector = vectorizer.transform([user_input])
    probabilities = model.predict_proba(input_vector)
    confidence = max(probabilities[0])

    if confidence < 0.25:
        return "Sorry, I didn't understand that. Can you please rephrase?"

    tag = model.predict(input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    reply = get_bot_response(user_message)
    return jsonify({"reply": reply})

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
