import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from googletrans import Translator
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
import traceback

# Initialize components
nltk.download('vader_lexicon')
translator = Translator()
sia = SentimentIntensityAnalyzer()

# Load models
models = {
    'Logistic Regression': joblib.load('logistic_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl'),
    'Gradient Boosting': joblib.load('gradient_boosting_model.pkl'),
    'BERT': {
        'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        'model': BertForSequenceClassification.from_pretrained('bert-base-uncased')
    }
}

# Load TF-IDF vectorizer
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Core analysis function
def analyze_text(text):
    try:
        # Language detection and translation
        detected = translator.detect(text)
        lang = detected.lang or 'en'
        translated = translator.translate(text, dest='en').text if lang != 'en' else text

        # VADER analysis
        vader_scores = sia.polarity_scores(translated)
        compound_score = vader_scores['compound']

        results = {
            'VADER': vader_scores,
            'Language': f"{lang.upper()}",
            'Translated': translated,
            'Sentiment': 'Positive' if compound_score >= 0.05 else 'Negative' if compound_score <= -0.05 else 'Neutral',
            'Confidence': max(vader_scores['pos'], vader_scores['neg'], vader_scores['neu']) * 100
        }

        # Traditional model predictions
        features = tfidf.transform([translated])
        for model in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
            results[model] = {
                'probs': models[model].predict_proba(features)[0],
                'pred': models[model].predict(features)[0]
            }

        # BERT prediction
        inputs = models['BERT']['tokenizer'](
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        with torch.no_grad():
            bert_output = models['BERT']['model'](**inputs)
        results['BERT'] = {
            'probs': torch.softmax(bert_output.logits, dim=1).numpy()[0],
            'pred': torch.argmax(bert_output.logits).item()
        }

        return results

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}

# Streamlit UI
def main():
    st.title("🔍 Code-Mixed Sentiment Analysis App")
    st.markdown("This app analyzes Telugu-English code-mixed text sentiment using multiple models.")

    text_input = st.text_area("Enter Text for Analysis:")

    if st.button("Analyze"):
        if text_input.strip():
            results = analyze_text(text_input)

            if 'error' in results:
                st.error(f"Error: {results['error']}")
                return

            st.markdown(f"**Detected Language:** {results['Language']}")
            st.markdown(f"**Translated Text:** {results['Translated']}")
            st.markdown(f"**Sentiment:** {results['Sentiment']}")
            st.markdown(f"**Confidence:** {results['Confidence']:.1f}%")
        else:
            st.warning("⚠️ Please enter some text for analysis.")

if __name__ == '__main__':
    main()
