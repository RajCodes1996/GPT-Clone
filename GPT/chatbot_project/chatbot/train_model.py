import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "chat_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ✅ Check if chat_data.csv exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")

# ✅ Load chatbot training data
try:
    df = pd.read_csv(DATA_PATH)
    print("✅ chat_data.csv loaded successfully!")
except Exception as e:
    raise ValueError(f"❌ Error loading chat_data.csv: {e}")

# ✅ Ensure required columns exist
if "user_input" not in df.columns or "bot_response" not in df.columns:
    raise ValueError("❌ chat_data.csv must have 'user_input' and 'bot_response' columns.")

# ✅ Remove any empty rows
df = df.dropna(subset=["user_input", "bot_response"])

# ✅ Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["user_input"])
y = df["bot_response"]

# ✅ Train the ML model
model = LogisticRegression()
model.fit(X, y)

# ✅ Save the trained model & vectorizer
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)
with open(VECTORIZER_PATH, "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Chatbot model trained & saved successfully!")
