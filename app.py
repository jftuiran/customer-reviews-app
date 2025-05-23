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

# Cachear los pipelines pesados
@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment_pipeline():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

@st.cache_resource(show_spinner="Loading summarization model...")
def get_summarizer_pipeline():
    return pipeline("summarization", model="facebook/bart-large-cnn")

sentiment_pipe = get_sentiment_pipeline()
summarizer = get_summarizer_pipeline()

uploaded_file = st.file_uploader("Upload your CSV file with reviews", type=["csv"])
if uploaded_file:
    content = uploaded_file.read().decode('utf-8')
    st.text_area("File preview (first 1000 chars):", content[:1000], height=150)
    uploaded_file.seek(0)

    try:
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
    stop_words = set(stopwords.words('english'))
    text = " ".join(df['opinion'].astype(str))
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]

    # --- Nube de palabras con fondo oscuro ---
    st.subheader("Word Cloud")
    fig_wc, ax_wc = plt.subplots(figsize=(8, 4), facecolor='#222831')
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='#222831',
        colormap='Blues'
    ).generate(" ".join(filtered_words))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    fig_wc.patch.set_facecolor('#222831')
    st.pyplot(fig_wc)

    # --- Top 10 most frequent words ---
    counter = Counter(filtered_words)
    common_words = counter.most_common(10)
    words_, counts = zip(*common_words)
    fig_bar, ax_bar = plt.subplots(figsize=(9, 5), facecolor='#222831')
    bar_colors = ['#4FC3F7', '#29B6F6', '#039BE5', '#0288D1', '#0277BD', '#01579B', '#B3E5FC', '#81D4FA', '#0288D1', '#00B8D4']
    bars = ax_bar.barh(words_, counts, color=bar_colors[:len(words_)], edgecolor='black', height=0.7)
    ax_bar.set_xlabel("Frequency", fontsize=14, weight='bold', color='white')
    ax_bar.set_title("Top 10 Most Frequent Words", fontsize=18, weight='bold', pad=15, color='white')
    ax_bar.invert_yaxis()
    ax_bar.set_facecolor('#222831')
    fig_bar.patch.set_facecolor('#222831')
    # Etiquetas grandes y legibles
    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', ha='left', color='#1fa2ff', fontsize=15, fontweight='bold')
    ax_bar.tick_params(axis='x', colors='white')
    ax_bar.tick_params(axis='y', colors='white')
    ax_bar.xaxis.label.set_color('white')
    ax_bar.yaxis.label.set_color('white')
    ax_bar.title.set_color('white')
    plt.tight_layout()
    st.pyplot(fig_bar)

    # --- Sentiment classification ---
    st.subheader("Sentiment Classification")
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
    fig_sent, ax_sent = plt.subplots(figsize=(6,4), facecolor='#222831')
    bars = ax_sent.barh(sentiment_counts.index, sentiment_counts.values, color=['#48C9B0', '#F4D03F', '#E74C3C'], edgecolor='black', linewidth=1.5)
    ax_sent.set_xlabel("Number of Reviews", fontsize=12, color='white')
    ax_sent.set_title("Sentiment Distribution", fontsize=14, weight='bold', color='white')
    ax_sent.invert_yaxis()
    ax_sent.set_facecolor('#222831')
    fig_sent.patch.set_facecolor('#222831')
    for bar, count in zip(bars, sentiment_counts.values):
        ax_sent.text(bar.get_width()-0.1, bar.get_y()+bar.get_height()/2, str(count),
                va='center', ha='right', color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    ax_sent.tick_params(axis='x', colors='white')
    ax_sent.tick_params(axis='y', colors='white')
    ax_sent.xaxis.label.set_color('white')
    ax_sent.yaxis.label.set_color('white')
    ax_sent.title.set_color('white')
    plt.tight_layout()
    st.pyplot(fig_sent)

    # --- Summarization ---
    st.subheader("General Summary of Reviews")
    resumen = summarizer(" ".join(df['opinion'].astype(str))[:2000], max_length=80, min_length=20, do_sample=False)[0]['summary_text']
    st.info(resumen)

    # --- Analyze new review ---
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
