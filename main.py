import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))

# Loading the saved files
dl_model = load_model('emotion_bidrn_lstm_model.h5')

with open('dl_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('vocab_size.pkl', 'rb') as file:
    vocab_size = pickle.load(file)

with open('max_len.pkl', 'rb') as file:
    max_len = pickle.load(file)

positive_emotions = ['joy', 'love', 'surprise']
negative_emotions = ['anger', 'fear', 'sadness']

# Defining emoji for each emotion
emotion_emoji = {
    'joy': '😄',
    'love': '❤️',
    'sadness': '😢',
    'anger': '😡',
    'fear': '😱',
    'surprise': '😲'
}

st.set_page_config(page_title="Emotion Detection App")
st.title("😊 Emotion Detection from Text")
st.write("Enter some text below and the model will predict the emotion.")

input_text = st.text_area("Enter your text here:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocessing
        lemmatizer = WordNetLemmatizer()
        english_stopwords = set(stopwords.words('english'))

        text = re.sub(r'[^a-zA-Z]', ' ', input_text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        words = nltk.word_tokenize(text, language='english')
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in english_stopwords]
        text = " ".join(filtered_words)

        # One-hot encoding and padding
        seq = pad_sequences([one_hot(text, n=vocab_size)], maxlen=max_len, padding='pre')

        # Prediction
        probs = dl_model.predict(seq)[0]
        pred_class = np.argmax(probs)
        pred_label = encoder.inverse_transform([pred_class])[0]
        conf = probs[pred_class]

        col1, col2 = st.columns([1, 2])

        with col1:
            emoji = emotion_emoji.get(pred_label.lower(), "")
            if pred_label.lower() in positive_emotions:
                st.success(f"Emotion: {pred_label.title()}{emoji} ")
            else:
                st.error(f"Emotion: {pred_label.title()}{emoji}")
            st.info(f"Confidence: {conf:.2%}")

        with col2:
            plot_data = pd.DataFrame({
                'Emotion': encoder.classes_,
                'Probability': probs
            })
            fig, ax = plt.subplots(figsize=(7,4))
            sns.barplot(x='Emotion', y='Probability', data=plot_data, palette='pastel', ax=ax)
            ax.set_ylim(0,1)
            ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.7)
            plt.title("Predicted Probabilities for Each Emotion")
            st.pyplot(fig)
