from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
import os
import pickle
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from fuzzywuzzy import process  # For approximate string matching

# ✅ Ensure necessary NLTK components are downloaded
nltk.download("punkt")
nltk.download("stopwords")

# ✅ Define stopwords
stop_words = set(stopwords.words("english"))

# ✅ Get the absolute path to project and chatbot directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")
CSV_PATH = os.path.join(CHATBOT_DIR, "chat_data.csv")
MODEL_PATH = os.path.join(CHATBOT_DIR, "chatbot_model.pkl")
VECTORIZER_PATH = os.path.join(CHATBOT_DIR, "vectorizer.pkl")

# ✅ Load chatbot dataset for keyword-based fallback
if os.path.exists(CSV_PATH):
    try:
        df = pd.read_csv(CSV_PATH)
        responses = {row["user_input"].lower(): row["bot_response"] for _, row in df.iterrows()}
    except Exception as e:
        print(f"❌ Error loading chat_data.csv: {e}")
        responses = {}
else:
    print("❌ chat_data.csv not found! Using an empty fallback response set.")
    responses = {}

# ✅ Load the trained model & vectorizer (if available)
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print(f"✅ Loading ML model from: {MODEL_PATH}")
    print(f"✅ Loading vectorizer from: {VECTORIZER_PATH}")
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
else:
    print("❌ Model or vectorizer not found! Using keyword-based fallback.")
    model = None
    vectorizer = None


# ✅ **LIVE SEARCH SUGGESTION**
@csrf_exempt
def get_suggestions(request):
    query = request.GET.get("query", "").lower()
    if not query:
        return JsonResponse([], safe=False)  # Ensure safe=False when returning a list

    try:
        df = pd.read_csv("path/to/chat_data.csv")  # Adjust the path
        suggestions = df[df["message"].str.lower().str.contains(query, na=False)]["message"].tolist()
        return JsonResponse(suggestions[:5], safe=False)  # Return only top 5 results
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ✅ **Mathematical Calculation Function**
def evaluate_math_expression(expression):
    try:
        expression = re.sub(r"[^0-9+\-*/().]", "", expression)  # Remove unwanted characters
        result = eval(expression, {"__builtins__": {}})
        return f"The answer is {result}"
    except Exception:
        return "Sorry, I couldn't compute that."


# ✅ **Text Preprocessing Function**
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)


# ✅ **Sentiment Analysis Function**
def analyze_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        return "positive"
    elif sentiment_score < 0:
        return "negative"
    return "neutral"


# ✅ **Named Entity Recognition (NER)**
def extract_entities(text):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    entities = {}
    for word, tag in pos_tags:
        if tag == "NNP":
            entities["PERSON"] = word
    return entities


# ✅ **Find Response (ML + Fuzzy Matching + Math Evaluation)**
def get_chatbot_response(user_input):
    user_input = user_input.lower()

    # ✅ Check if input is a math expression
    if re.search(r"^[0-9+\-*/(). ]+$", user_input):
        return evaluate_math_expression(user_input)

    # ✅ 1. Exact Match from `chat_data.csv`
    if user_input in responses:
        return responses[user_input]

    # ✅ 2. Use ML Model (if available)
    if model and vectorizer:
        cleaned_text = preprocess_text(user_input)
        X_input = vectorizer.transform([cleaned_text])
        return model.predict(X_input)[0]

    # ✅ 3. Use Fuzzy Matching for best match
    best_match, score = process.extractOne(user_input, responses.keys())
    if score > 60:
        return responses[best_match]

    return "I'm not sure, but I can try to help!"


# ✅ **Chatbot Response API**
@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()

            if not user_message:
                return JsonResponse({"response": "Please type something to chat!"})

            # ✅ Get chatbot response
            bot_reply = get_chatbot_response(user_message)

            # ✅ Sentiment Analysis
            sentiment = analyze_sentiment(user_message)

            # ✅ Named Entity Recognition (NER)
            entities = extract_entities(user_message)

            # ✅ Modify response based on sentiment
            if sentiment == "negative":
                bot_reply = f"I'm here for you. {bot_reply}"
            elif sentiment == "positive":
                bot_reply = f"Great to hear! {bot_reply}"

            # ✅ Personalize response if NER detects a name
            if "PERSON" in entities:
                bot_reply = f"Hey {entities['PERSON']}, {bot_reply}"

            print(f"✅ Bot Response: {bot_reply}")
            return JsonResponse({"response": bot_reply})

        except Exception as e:
            print(f"❌ Error: {e}")
            return JsonResponse({"response": "Sorry, I encountered an error!"})

    return JsonResponse({"error": "Invalid request"}, status=400)


# ✅ **Render Chatbot UI**
def chatbot_home(request):
    return render(request, "chatbot/index.html")
