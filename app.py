import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🕵️",
    layout="centered"
)
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #38bdf8;
    text-align: center;
}
.stTextArea textarea {
    border-radius: 12px;
    padding: 12px;
}
.stButton button {
    background-color: #38bdf8;
    color: black;
    border-radius: 12px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Fake Review Detection", layout="wide")

st.title("🕵️ Fake Review Detection App")

st.write("CSV file load pannitu table display panrom 👇")

# CSV file path
file_path = "data/fake reviews dataset.csv"

try:
    df = pd.read_csv(file_path)
    st.success("CSV file successfully loaded!")
    st.dataframe(df)
except Exception as e:
    st.error(f"File load aagala 😢 : {e}")
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

st.subheader("🧹 Data Cleaning in Progress...")

df['clean_text'] = df['text_'].apply(clean_text)

st.success("Text cleaning completed ✅")

st.subheader("✨ Cleaned Text Preview")
st.dataframe(df[['text_', 'clean_text']].head(10))
from sklearn.feature_extraction.text import TfidfVectorizer

st.subheader("🔢 Feature Extraction (TF-IDF)")

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])

st.success("Text successfully converted into numerical features!")

st.write("TF-IDF Matrix Shape:", X.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.subheader("🤖 Model Training - Logistic Regression")

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.success("Model trained successfully!")
st.write("🎯 Model Accuracy:", round(acc * 100, 2), "%")
# Save model & vectorizer
joblib.dump(model, "model/fake_review_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

st.success("Model & Vectorizer saved successfully in model/ folder 💾")

st.markdown("<h2 style='text-align:center;'>🕵️ Fake Review Detection</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Paste a product review below to check whether it is Genuine or Fake</p>", unsafe_allow_html=True)

review = st.text_area("✍️ Enter Review Here", height=150, placeholder="Type or paste your review here...")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict_btn = st.button("🔍 Detect Review")

if predict_btn:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        clean_input = clean_text(review)
        vec_input = vectorizer.transform([clean_input])
        prediction = model.predict(vec_input)[0]

        if prediction == "CG":
            st.success("✅ Genuine Review Detected")
        else:
            st.error("🚨 Fake Review Detected")

