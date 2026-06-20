# 🛡️ Roman Urdu Hate Speech Detection System

A Machine Learning-based system that automatically detects Hate Speech in Roman Urdu text (Urdu written using the English alphabet). The project combines Natural Language Processing (NLP), machine learning, and an interactive Streamlit web application to provide real-time hate speech detection and analysis.

---

# Project Overview

This project was developed as part of the Machine Learning course.

The system classifies Roman Urdu comments into two categories:

* ✅ Normal
* 🚫 Hate Speech

Multiple machine learning algorithms were trained and evaluated. After comparing performance and efficiency, Multinomial Naive Bayes was selected as the deployment model due to its excellent balance between accuracy and training speed.

The project also includes an interactive Streamlit dashboard for real-time prediction, batch analysis, chat simulation, and analytics visualization.

---

# 👥 Team Repository

This repository is my personal copy of the team project. The original team repository is:

https://github.com/SaifUllahUmar0317/Hate_Speech_detection_Roman_Urdu

maintained by Saifullah Umar

---

# My Contributions

I was primarily responsible for the machine learning pipeline and project development, including:

* Complete model training and evaluation pipeline (`main.py`)
* Hyperparameter tuning using RandomizedSearchCV
* Model comparison and selection
* Performance evaluation and analysis
* Explainable AI features
* Batch processing functionality
* Chat simulator implementation
* Analytics dashboard
* Project documentation

---

# ✨ Features

* Real-time Hate Speech Detection
* Batch Comment Analysis
* Interactive Chat Simulator
* Analytics Dashboard
* Explainable AI Predictions
* Confidence Score Visualization
* Roman Urdu Text Processing
* Fast Prediction Response

---

# How It Works

## 1. User Input

The user enters a Roman Urdu sentence or comment.

### Example

```text
Tu boht gnda insan hai
```

## 2. Text Preprocessing

The system performs:

* Lowercase conversion
* URL removal
* Punctuation removal
* Number removal
* Emoji removal
* Stopword removal
* Text normalization

## 3. Feature Extraction

The cleaned text is converted into numerical features using:

**TF-IDF Vectorization**

This transforms textual information into a machine-readable format.

## 4. Prediction

The trained machine learning model predicts whether the text contains hate speech.

## 5. Output

### Normal

Safe and non-offensive content.

### Hate Speech

Offensive or toxic content identified by the model.

---

# Models Compared

| Model                      | F1-Score   | Training Time |
| -------------------------- | ---------- | ------------- |
| Logistic Regression        | 0.8404     | 6.43s         |
| Random Forest              | 0.7811     | 39.70s        |
| SVM                        | 0.8462     | 399.94s       |
| **Naive Bayes (Selected)** | **0.8388** | **0.15s**     |

---

# Key Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 86.4%  |
| Precision | 85.2%  |
| Recall    | 82.5%  |
| F1-Score  | 0.8388 |
| ROC-AUC   | 0.9397 |

### Why Naive Bayes?

Although SVM achieved a slightly higher F1-score, Naive Bayes was selected because:

* Comparable predictive performance
* Extremely fast training time
* Approximately **2,666× faster** than SVM
* Better suited for real-time deployment

---

# Dataset

### Sources

* YouTube Comments
* Facebook Posts
* Kaggle Dataset (HS-RU-20)

### Dataset Statistics

| Category       | Samples |
| -------------- | ------- |
| Total Samples  | 10,347  |
| After Cleaning | 8,856   |
| Normal         | 5,276   |
| Hate Speech    | 5,071   |

The dataset was cleaned and balanced before training to improve model performance and reliability.

---

# Technologies Used

| Area                 | Tools                |
| -------------------- | -------------------- |
| Programming Language | Python               |
| Web Framework        | Streamlit            |
| Machine Learning     | Scikit-learn         |
| NLP                  | TF-IDF Vectorization |
| Data Handling        | Pandas, NumPy        |
| Visualization        | Matplotlib, Plotly   |
| Model Persistence    | Joblib               |

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/roman-urdu-hate-speech-detection.git
cd roman-urdu-hate-speech-detection
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Application

```bash
streamlit run app.py
```

## 4. Open Browser

```text
http://localhost:8501
```

---

# Project Structure

```text
Roman_Urdu_Hate_Speech_Detection/
│
├── app.py
├── main.py
├── requirements.txt
├── dataset.csv
├── best_roman_urdu_hate_model.pkl
├── roman_urdu_vectorizer.pkl
├── selection_metadata.pkl
├── model_results.pkl
└── README.md
```

# License

This project was developed for educational and research purposes.

If you use, modify, or distribute any part of this project, please provide appropriate credit to the original authors.

---

# 👩‍💻 Author

**Izza Mustafa**

                                     © Roman Urdu Hate Speech Detection System
