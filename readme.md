# 😊 Emotion Detection from Text

This project implements a deep learning model to predict emotions from text. The model can classify a given text into **six emotions**: joy, love, surprise, anger, fear, and sadness.

---

## Overview

Understanding human emotions from text is crucial for sentiment analysis, chatbots, social media monitoring, and mental health applications. This project uses **Bi-directional LSTM** to capture the context of words in both directions, combined with word embeddings for better semantic understanding.

Key features:

- Predicts the **primary emotion** of a text.
- Shows **confidence scores** and a probability distribution across all six emotions.
- Deployed using **Streamlit** for interactive use.

---

## Model Details

- **Architecture:**
  - Embedding layer to convert words to dense vectors.
  - Bi-directional LSTM to capture sequential context.
  - Dense layers with dropout to reduce overfitting.
  - Softmax output layer for 6-class classification.
- **Hyperparameters:**
  - Embedding dimension: 128
  - LSTM units: 48
  - Dense units: 32
  - Dropout: 0.25–0.3
- **Loss Function:** Categorical cross-entropy
- **Optimizer:** RMSprop
- **Training Epochs:** 10
- **Batch Size:** 32

---

## Training Performance

The model was trained on **~16,000 text samples**, achieving:

- **Training Accuracy:** 95.49%
- **Validation Accuracy:** 90.38%
- **Observations:** Validation loss sometimes fluctuated, indicating the model slightly overfits after several epochs. Early stopping was used to restore the best weights.

---

## Challenges Faced

1. **Overfitting:** Initial models achieved very high training accuracy (~97%) but validation accuracy lagged (~90%).  
   - **Solution:** Added dropout layers and reduced LSTM units.
2. **Text Preprocessing:** Handling punctuation, stopwords, and lemmatization was crucial to improve model generalization.
3. **Visualization:** Users often want to see probabilities for all classes, so bar charts with color coding were added for clarity.
4. **Streamlit Deployment:** Ensuring smooth UI with side-by-side probability charts and emoji-based result highlights.

---

## Deployment

The model is deployed as a **Streamlit web app**. Features include:

- Text input area
- Real-time prediction of emotion
- Confidence display
- Probability bar chart of all emotions
- Color-coded results for positive vs negative emotions

**Run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
