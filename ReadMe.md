# 📊 Sentiment Classification Using CNN and LSTM

This project compares the performance of two Deep Neural Network architectures — **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** — for **sentiment classification** of TripAdvisor reviews.

## 🧠 Overview

The core of this project is a Python class, `reviewSentimentClassifier`, which automates the following steps:
- Data loading and cleaning
- Exploratory data analysis
- Text preprocessing and tokenization
- Model building (CNN and LSTM)
- Model training with early stopping and checkpoints
- Evaluation and performance comparison using metrics and ROC curves

## 📂 Dataset

- **Source:** TripAdvisor reviews dataset
- **Columns Used:**
  - `Review`: The text review
  - `Rating`: Numerical rating (1 to 5), mapped to sentiment:
    - 1–2: Negative (0)
    - 3: Neutral (1)
    - 4–5: Positive (2)

## ⚙️ Features

- NLP preprocessing using **NLTK**
- **Tokenization and Padding** with Keras
- **Stratified train-test split**
- **Text classification** using:
  - CNN with 1D Convolutions and GlobalMaxPooling
  - LSTM with Bidirectional RNNs
- Performance tracking via:
  - Accuracy, Precision, Recall, F1-Score
  - ROC Curves and AUC

## 🏗️ Model Architectures

### 🧩 CNN

- Embedding Layer
- 1D Convolution + MaxPooling
- GlobalMaxPooling
- Dense + Dropout
- Output: Softmax

### 🔁 LSTM

- Embedding Layer
- Spatial Dropout
- Bidirectional LSTM
- Dense + Dropout
- Output: Softmax

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**, **Recall**, **F1-score** (Weighted)
- **Confusion Matrix**
- **ROC Curves** for each class (Negative, Neutral, Positive)


## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- NLTK
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn


## Installation

1. Make sure Python 3.11 is installed on your system
2. Create a virtual environment:

python3.11 -m venv venv

3. Activate the virtual environment:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```

4. Install the required packages:

pip install pandas numpy matplotlib seaborn nltk tensorflow scikit-learn

5. Run the program!
python3.11

## 📁 Output

- Trained models saved in `models/`
- Tokenizer object saved as `tokenizer.pickle`
- Visualizations:
  - `rating_distribution.png`
  - `sentiment_distribution.png`
  - `review_length_distribution.png`
  - `roc_curves.png`


