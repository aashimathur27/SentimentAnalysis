import streamlit as st
import joblib
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

st.title("📝 Amazon Product Review Sentiment Analysis")

st.subheader("🔍 Enter a Product Review")
user_input = st.text_area("Type your review here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() != "":
        clean_text = preprocess_text(user_input)
        vec_text = vectorizer.transform([clean_text])
        prediction = model.predict(vec_text)[0]
        proba = model.predict_proba(vec_text)[0]

        st.success(f"**Predicted Sentiment:** {prediction}")

        proba_df = pd.DataFrame({
            'Sentiment': model.classes_,
            'Probability': proba
        })
        fig = px.pie(proba_df, names='Sentiment', values='Probability', title='Prediction Probabilities')
        st.plotly_chart(fig)
    else:
        st.warning("Please enter a review before clicking Analyze.")

st.subheader("📄 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with a 'reviews' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'reviews' not in df.columns:
            st.error("The CSV must contain a 'reviews' column.")
        else:
            max_rows = st.slider("Select number of reviews to analyze", min_value=100, max_value=len(df), step=100, value=1000)
            df = df.head(max_rows)
            df['cleaned'] = df['reviews'].astype(str).apply(preprocess_text)
            vec_reviews = vectorizer.transform(df['cleaned'])
            df['Predicted Sentiment'] = model.predict(vec_reviews)
            
            st.write("### 🧾 Results:")
            st.dataframe(df[['reviews', 'Predicted Sentiment']])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "sentiment_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
