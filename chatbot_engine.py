import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("punkt")
nltk.download("punkt_tab")

# Load intents
with open("intents.json", "r") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# NLP Vectorization
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(patterns)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, tags)

def get_bot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    probabilities = model.predict_proba(input_vector)
    confidence = max(probabilities[0])

    if confidence < 0.4:
        return "Sorry, I didn't understand that. Can you rephrase?"

    tag = model.predict(input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
