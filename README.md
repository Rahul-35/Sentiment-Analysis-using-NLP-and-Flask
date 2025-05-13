# Sentiment Analysis Using NLTK, TF-IDF, Logistic Regression and Flask
## Project Overview
This project implements a Sentiment Analysis system that classifies text as positive, negative, or neutral using the NLTK library for text processing, TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction, and Logistic Regression for classification.

## Features:
  Text Preprocessing: Tokenization, stopword removal, and lemmatization using NLTK.
  Feature Extraction: Use of TF-IDF to convert text into numerical vectors.
  Classification: Logistic Regression model to predict sentiment.

## Data:
  For the purpose of this project, a CSV file (e.g., review.csv) containing text data and labels (e.g., 0 and 1) is required.

## Sample CSV format:

Text	Sentiment
I love this product!	positive
The movie was a disaster	negative
Not bad not good.	neutral

# How It Works
## 1.Text Preprocessing:

  Tokenization: Breaks the text into words.
  Stopword Removal: Filters out common words that don’t contribute to sentiment (e.g., “and”, “the”).
  Lemmatization: Converts words to their base form (e.g., “running” → “run”).

## 2.Feature Extraction:
  TF-IDF is applied to convert textual data into numerical format.
  
## 3.Model Training:
  Logistic Regression model is trained on the preprocessed text data to classify sentiment.

## Model Evaluation:

  After training, the model is evaluated on a test dataset. Common metrics for evaluation:
  Accuracy: Percentage of correctly predicted sentiments.
  Confusion Matrix: To see the distribution of predicted vs actual labels.
  Flask: Used for the frontend





