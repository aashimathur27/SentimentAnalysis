# 🧠 Sentiment Analysis Web App using Streamlit

This is a machine learning project that performs **Sentiment Analysis** on Amazon product reviews using **Natural Language Processing (NLP)** techniques. 
It is deployed as a user-friendly **Streamlit web application** with real-time predictions and CSV upload support.

---

## 🚀 Features

- 🔍 Predict sentiment of product reviews (Positive / Negative / Neutral)
- 📈 Shows prediction probability pie chart
- 📂 Upload CSV file for **batch sentiment analysis**
- 📊 Displays a sentiment summary dashboard for batch uploads
- 💬 Built using **scikit-learn**, **NLTK**, **pandas**, and **Streamlit**

---

## 📁 Dataset

We use Amazon product reviews with `review` and `rating` columns. Ratings are mapped to:
- **1–2 → Negative**
- **3 → Neutral**
- **4–5 → Positive**

---

## 🧰 Tech Stack

- Python 🐍
- Streamlit 📺
- scikit-learn 🧠
- NLTK 📚
- Pandas & NumPy
- Plotly (for interactive charts)

---

## 🏗️ Project Structure

sentiment-analysis/
├── app.py # Streamlit web app
├── train_model.py # Training script to generate model
├── sentiment_model.pkl # Trained classification model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Dependencies
└── Amazon-Product-Reviews.csv # Sample dataset

---

 ▶️ How to Run the App

1. Install Dependencies
```bash
pip install -r requirements.txt
2. Train the Model
python train_model.py
3. Run the Streamlit App
streamlit run app.py

✨ Future Improvements
Add more detailed sentiment categories (e.g., Very Positive)
Use deep learning (e.g., LSTM/BERT) for better accuracy
Allow visualizations like word clouds for uploaded reviews

📌 Sample Usage
Input:
"The product was absolutely wonderful and easy to use!"
Output:
Sentiment: Positive ✅
🤝 Contributing
Feel free to fork the repo and submit a pull request with suggestions or improvements.
