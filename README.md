# Roman Urdu Hate Speech Detection System

A machine learning-based system that automatically detects hate speech in Roman Urdu text (Urdu written in English script).

## Project Overview

This project was developed as part of the **Machine Learning (CS284)** course. The system classifies Roman Urdu comments as either **Normal** or **Hate Speech** using multiple ML algorithms, with Naive Bayes selected as the optimal model for real-time deployment.

## Author

**Izza Mustafa (24)**  
Machine Learning Course - Semester Project

## Team Repository

This is my personal copy of the team project. The original team repository is maintained by Saifullah Umar:
- https://github.com/SaifUllahUmar0317/Hate_Speech_detection_Roman_Urdu

## My Contributions

- Complete model training and evaluation pipeline (`main.py`)
- Hyperparameter tuning with RandomizedSearchCV
- Model selection (Naive Bayes selected over SVM for 2,666x faster training)
- Streamlit web application with explainable AI features (`app.py`)
- Batch processing, chat simulator, and analytics dashboard
- Project documentation

## Models Compared

| Model | F1-Score | Training Time |
|-------|----------|---------------|
| Logistic Regression | 0.8404 | 6.43s |
| Random Forest | 0.7811 | 39.70s |
| SVM | 0.8462 | 399.94s |
| **Naive Bayes (Selected)** | **0.8388** | **0.15s** |

## Key Results

- **Best Model:** Naive Bayes
- **Accuracy:** 86.4%
- **F1-Score:** 0.8388
- **ROC-AUC:** 0.9397
- **Training Time:** 0.15 seconds (2,666x faster than SVM)

## Dataset

- **Source:** YouTube comments, Facebook posts, Kaggle (HS-RU-20)
- **Total samples:** 10,347 (8,856 after cleaning)
- **Normal:** 5,276 (50.99%)
- **Hate Speech:** 5,071 (49.01%)

## Features

- **Real-time Analysis** - Instant hate speech detection
- **Batch Processing** - Analyze multiple comments at once
- **Chat Simulator** - Test in conversation context
- **Analytics Dashboard** - Track prediction history
- **Explainable AI** - Confidence scores and hate word highlighting

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/roman-urdu-hate-speech-detection.git
cd roman-urdu-hate-speech-detection
