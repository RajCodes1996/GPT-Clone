from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
import os
import pickle
import nltk
import pandas as pd
from fuzzywuzzy import process
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Ensure necessary NLTK components are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Define stop words
stop_words = set(stopwords.words('english'))

# Get the absolute path to model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ✅ Load the trained model & vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print(f"✅ Loading model from: {MODEL_PATH}")
    print(f"✅ Loading vectorizer from: {VECTORIZER_PATH}")
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
else:
    print("❌ Model or vectorizer not found! Please run `python train_model.py`")
    model = None
    vectorizer = None

# ✅ Load chatbot dataset for live search suggestions
DATA_PATH = os.path.join(BASE_DIR, "chat_data.csv")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    responses = [row["user_input"].lower() for _, row in df.iterrows()]
else:
    print("❌ chat_data.csv not found! Please ensure the dataset is present.")
    responses = []

# ✅ **Preprocessing Function**
def preprocess_text(text):
    """Tokenization, Stopwords Removal, and Lemmatization"""
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    lemmatized_text = " ".join([TextBlob(word).words.lemmatize()[0] for word in words])
    return lemmatized_text

# ✅ **Sentiment Analysis Function**
def analyze_sentiment(text):
    """Perform Sentiment Analysis using TextBlob"""
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        return "positive"
    elif sentiment_score < 0:
        return "negative"
    else:
        return "neutral"

# ✅ **Named Entity Recognition (NER)**
def extract_entities(text):
    """Extract Named Entities using NLTK"""
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    entities = {}
    for word, tag in pos_tags:
        if tag == "NNP":  
            entities["PERSON"] = word
    return entities

# ✅ **Chatbot Response Function**
@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()

            if not user_message:
                return JsonResponse({"response": "Please type something to chat!"})

            # Check if the model is loaded
            if model is None or vectorizer is None:
                return JsonResponse({"response": "Error: Model not loaded. Please train the chatbot using `python train_model.py`"})

            # ✅ Preprocess user message
            cleaned_text = preprocess_text(user_message)

            # ✅ Transform user input using vectorizer
            X_input = vectorizer.transform([cleaned_text])

            # ✅ Predict chatbot response
            bot_reply = model.predict(X_input)[0]

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

# ✅ **Live Search Suggestions Function**
def get_suggestions(request):
    query = request.GET.get("query", "").lower()
    if not query or not responses:
        return JsonResponse({"suggestions": []})

    # Get top 5 closest matches using fuzzy matching
    matches = process.extract(query, responses, limit=5)
    suggestions = [match[0] for match in matches if match[1] > 50]  # Only include matches above 50%

    return JsonResponse({"suggestions": suggestions})

# ✅ **Render Chatbot UI**
def chatbot_home(request):
    return render(request, "chatbot/index.html")
