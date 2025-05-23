import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import nltk
import re
from transformers import pipeline

nltk.download('stopwords')

st.set_page_config(page_title="Customer Reviews Analysis", layout="centered")
st.title("Customer Reviews - NLP Analysis")

uploaded_file = st.file_uploader("Upload your CSV file with reviews", type=["csv"])
if uploaded_file:
    # Mostrar contenido del archivo para depuración
    content = uploaded_file.read().decode('utf-8')
    st.text_area("File preview (first 1000 chars):", content[:1000], height=150)
    uploaded_file.seek(0)

    try:
        # Intentar cargar con punto y coma, si falla, intentar con coma
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        if 'opinion' not in df.columns:
            st.error("The file must have a column named 'opinion'.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    st.write("Preview of reviews:")
    st.dataframe(df.head())

    # Limpieza de texto (solo inglés)
    lang = 'english'
    text = " ".join(df['opinion'].astype(str))
    stop_words = set(stopwords.words(lang))
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]

    # Nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(" ".join(filtered_words))
    st.subheader("Word Cloud")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Top 10 palabras más frecuentes
    counter = Counter(filtered_words)
    common_words = counter.most_common(10)
    words_, counts = zip(*common_words)
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.barh(words_, counts, color='#5DADE2')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width()-0.2, bar.get_y()+bar.get_height()/2, str(count), va='center', ha='right', color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title("Top 10 Most Frequent Words", fontsize=14, weight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    # Clasificación de sentimiento (solo inglés, modelo multilingüe)
    st.subheader("Sentiment Classification")
    sentiment_pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    def map_sentiment(label):
        if label in ["Very Positive", "Positive"]:
            return "positive"
        elif label == "Neutral":
            return "neutral"
        elif label in ["Very Negative", "Negative"]:
            return "negative"
        else:
            return "neutral"
    def safe_sentiment(text):
        try:
            result = sentiment_pipe(text)
            if result and len(result) > 0:
                return map_sentiment(result[0]['label'])
            else:
                return "neutral"
        except Exception:
            return "neutral"
    df['sentiment'] = df['opinion'].apply(safe_sentiment)
    st.dataframe(df[['opinion', 'sentiment']])

    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.barh(sentiment_counts.index, sentiment_counts.values, color=['#48C9B0', '#F4D03F', '#E74C3C'])
    for bar, count in zip(bars, sentiment_counts.values):
        ax.text(bar.get_width()-0.2, bar.get_y()+bar.get_height()/2, str(count), va='center', ha='right', color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Number of Reviews", fontsize=12)
    ax.set_title("Sentiment Distribution", fontsize=14, weight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    # Summarization (solo inglés)
    st.subheader("General Summary of Reviews")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    resumen = summarizer(" ".join(df['opinion'].astype(str))[:2000], max_length=80, min_length=20, do_sample=False)[0]['summary_text']
    st.info(resumen)

    # Analyze new review
    st.subheader("Analyze a New Review")
    new_comment = st.text_area("Write a new review here:")
    if st.button("Analyze Review"):
        if new_comment.strip():
            sentiment = safe_sentiment(new_comment)
            resumen_nuevo = summarizer(new_comment, max_length=40, min_length=10, do_sample=False)[0]['summary_text'] if len(new_comment.split()) > 20 else new_comment
            st.write(f"**Sentiment:** {sentiment.capitalize()}")
            st.write(f"**Summary:** {resumen_nuevo}")
        else:
            st.warning("Please write a review.")

else:
    st.info("Please upload a CSV file with a column named 'opinion'.")
