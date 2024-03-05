﻿# AI-or-Not-AI

## Overview
This repository contains code for training a machine learning model to detect whether a given text is AI-generated or human-written. The model is based on logistic regression and utilizes TF-IDF vectorization for text feature extraction. Additionally, Synthetic Minority Over-sampling Technique (SMOTE) is applied to address class imbalance in the dataset.

## Files
model.py: Python script for training the AI detection model.
text.csv: Dataset containing text samples labeled as AI-generated (1) or human-written (0).
ai_detection_model.pkl: Trained logistic regression model saved using joblib.
tfidf_vectorizer.pkl: TF-IDF vectorizer saved using joblib.

## Usage
# Install Dependencies:
pip install pandas scikit-learn imbalanced-learn joblib

## Run Training Script:
python train_model.py

# Trained Model:
The trained model (ai_detection_model.pkl) and TF-IDF vectorizer (tfidf_vectorizer.pkl) will be saved in the working directory.

## Evaluation:

The script prints accuracy, classification report, and confusion matrix on the test set.

# Notes
Ensure you have the necessary Python packages installed.
The dataset (text.csv) should be in the same directory as the training script.
Class imbalance is addressed using SMOTE during model training.
