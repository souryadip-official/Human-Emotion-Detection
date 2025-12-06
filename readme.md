# 😊 Emotion Detection from Text

This project implements a deep learning model to predict emotions from text. The model classifies text into **six emotions**: joy, love, surprise, anger, fear, and sadness.

---

## Overview

Understanding human emotions from text is crucial for sentiment analysis, chatbots, social media monitoring, and mental health applications. This project uses a **Bi-directional LSTM** model to capture context from both directions of a text sequence, combined with word embeddings for semantic understanding.

Key features:

- Predicts the **primary emotion** of a text.
- Shows **confidence scores** and a probability distribution across all six emotions.
- Interactive deployment via **Streamlit**.

---

## Model Details

- **Architecture:**
  - Embedding layer to convert words to dense vectors.
  - Bi-directional LSTM to capture sequential context.
  - Dense layers with dropout to prevent overfitting.
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

Trained on **~16,000 text samples**, the model achieved:

- **Training Accuracy:** 96.22%
- **Validation Accuracy:** 92.25%
- **Observations:** The model performs well overall. Some minor fluctuations in validation loss were observed, suggesting slight overfitting after several epochs.

---

## Challenges Faced

1. **Overfitting:** Early models achieved very high training accuracy (~97%) but lower validation accuracy (~90%).  
   - **Solution:** Added dropout layers and optimized LSTM units.
2. **Text Preprocessing:** Handling punctuation, stopwords, and lemmatization was essential for generalization.
3. **Visualization:** Users wanted to see probabilities for all classes, so bar charts with color-coded results were added.
4. **Streamlit Deployment:** Ensuring smooth UI with side-by-side probability charts and clear result highlights.

---

## Deployment

The model is deployed as a **Streamlit web app**. Features include:

- Text input area
- Real-time emotion prediction
- Confidence score display
- Probability distribution bar chart
- Color-coded results for positive vs negative emotions

**Run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py