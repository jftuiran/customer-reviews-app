import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
import re

# Descarga stopwords únicamente si no existen en el entorno
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Configuración de la página
st.set_page_config(page_title="Customer Reviews Analysis", layout="centered")
st.title("Customer Reviews - NLP Analysis")

# --- Cacheo de los pipelines pesados para reducir uso de memoria ---
@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment_pipeline():
    from transformers import pipeline
    # Modelo ligero y estable para análisis de sentimiento
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource(show_spinner="Loading summarization model...")
def get_summarizer_pipeline():
    from transformers import pipeline
    # Modelo de resumen pequeño y rápido
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Inicializa los pipelines como None para cargarlos solo cuando se necesiten
sentiment_pipe = None
summarizer = None

# --- Carga y procesado del archivo ---
uploaded_file = st.file_uploader("Upload your CSV file with reviews", type=["csv"])
if uploaded_file:
    try:
        # --- LÓGICA DE LECTURA ROBUSTA ---
        # Lee el archivo como texto plano para evitar errores de formato de pandas
        content = uploaded_file.read().decode('utf-8')
        lines = content.splitlines()

        # Quita la cabecera si existe
        if lines and lines[0].strip().lower() == 'opinion':
            lines = lines[1:]

        # Usa regex para extraer todo el texto entre comillas.
        # Esto maneja correctamente comas y saltos de línea dentro de una misma opinión.
        # re.DOTALL hace que '.' también coincida con saltos de línea.
        full_text = "\n".join(lines)
        reviews = re.findall(r'"(.*?)"', full_text, re.DOTALL)

        # Si el regex no encuentra nada (quizás el archivo no usa comillas),
        # se usa cada línea como una opinión.
        if not reviews and lines:
            reviews = [line.strip() for line in lines if line.strip()]
        
        # Crea el DataFrame a partir de la lista de opiniones limpias
        df = pd.DataFrame({'opinion': reviews})

        if df.empty:
            st.error("No reviews found in the file. Please check the file format.")
            st.stop()
        
        # Limita el número de filas para no sobrecargar los modelos
        df = df.head(1000)

    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}")
        st.stop()

    st.write("Preview of uploaded reviews:")
    st.dataframe(df.head())

    # --- Limpieza de texto y Nube de Palabras ---
    stop_words = set(stopwords.words('english'))
    text = " ".join(df['opinion'].astype(str))
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]

    st.subheader("Word Cloud")
    if filtered_words:
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
    else:
        st.write("Not enough words to generate a word cloud.")

    # --- Top 10 palabras más frecuentes ---
    st.subheader("Top 10 Most Frequent Words")
    if filtered_words:
        counter = Counter(filtered_words)
        common_words = counter.most_common(10)
        words_, counts = zip(*common_words) if common_words else ([], [])
        
        fig_bar, ax_bar = plt.subplots(figsize=(9, 5), facecolor='#222831')
        bar_colors = ['#4FC3F7', '#29B6F6', '#039BE5', '#0288D1', '#0277BD', '#01579B', '#B3E5FC', '#81D4FA', '#0288D1', '#00B8D4']
        bars = ax_bar.barh(words_, counts, color=bar_colors[:len(words_)], edgecolor='black', height=0.7)
        ax_bar.set_xlabel("Frequency", fontsize=14, weight='bold', color='white')
        ax_bar.set_title("Top 10 Most Frequent Words", fontsize=18, weight='bold', pad=15, color='white')
        ax_bar.invert_yaxis()
        ax_bar.set_facecolor('#222831')
        fig_bar.patch.set_facecolor('#222831')
        
        for bar, count in zip(bars, counts):
            ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', ha='left', color='#1fa2ff', fontsize=15, fontweight='bold')
        
        ax_bar.tick_params(axis='x', colors='white')
        ax_bar.tick_params(axis='y', colors='white')
        plt.tight_layout()
        st.pyplot(fig_bar)
    else:
        st.write("Not enough words to show frequency.")


    # --- Clasificación de Sentimiento ---
    st.subheader("Sentiment Classification")
    def map_sentiment(label):
        return label.lower()

    def safe_sentiment(text):
        global sentiment_pipe
        if sentiment_pipe is None:
            sentiment_pipe = get_sentiment_pipeline()
        try:
            # Truncar texto para que no exceda el límite del modelo
            result = sentiment_pipe(text[:512])
            return map_sentiment(result[0]['label']) if result else "neutral"
        except Exception:
            return "neutral"

    df['sentiment'] = df['opinion'].astype(str).apply(safe_sentiment)
    st.dataframe(df[['opinion', 'sentiment']].head())

    sentiment_counts = df['sentiment'].value_counts()
    fig_sent, ax_sent = plt.subplots(figsize=(6, 4), facecolor='#222831')
    bar_colors_map = {'positive': '#48C9B0', 'neutral': '#F4D03F', 'negative': '#E74C3C'}
    bar_colors = [bar_colors_map.get(s, '#888888') for s in sentiment_counts.index]
    
    bars = ax_sent.barh(sentiment_counts.index, sentiment_counts.values, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax_sent.set_xlabel("Number of Reviews", fontsize=12, color='white')
    ax_sent.set_title("Sentiment Distribution", fontsize=14, weight='bold', color='white')
    ax_sent.invert_yaxis()
    ax_sent.set_facecolor('#222831')
    fig_sent.patch.set_facecolor('#222831')
    
    for bar, count in zip(bars, sentiment_counts.values):
        ax_sent.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f' {count}',
                va='center', ha='left', color='white', fontsize=12, fontweight='bold')
                
    ax_sent.tick_params(axis='x', colors='white')
    ax_sent.tick_params(axis='y', colors='white')
    plt.tight_layout()
    st.pyplot(fig_sent)

    # --- Resumen General de Opiniones ---
    st.subheader("General Summary of Reviews")
    text_to_summarize = " ".join(df['opinion'].astype(str))[:2000] # Limita texto para el modelo
    if summarizer is None:
        summarizer = get_summarizer_pipeline()
    try:
        resumen = summarizer(text_to_summarize, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        st.info(resumen)
    except Exception as e:
        st.warning(f"Could not generate summary. Error: {e}")

    # --- Analizar una nueva opinión ---
    st.subheader("Analyze a New Review")
    new_comment = st.text_area("Write a new review here:", height=100)
    if st.button("Analyze Review"):
        if new_comment.strip():
            sentiment = safe_sentiment(new_comment)
            try:
                resumen_nuevo = summarizer(new_comment, max_length=50, min_length=15, do_sample=False)[0]['summary_text'] \
                    if len(new_comment.split()) > 20 else new_comment
            except Exception:
                resumen_nuevo = new_comment
            st.write(f"**Sentiment:** {sentiment.capitalize()}")
            st.write(f"**Summary:** {resumen_nuevo}")
        else:
            st.warning("Please write a review to analyze.")

else:
    st.info("Upload a CSV file with a column named 'opinion' to begin.")

