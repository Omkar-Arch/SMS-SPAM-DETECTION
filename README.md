# SMS Spam Detection System

## Overview

This project is a Machine Learning based SMS Spam Detection System built using Python, NLTK, Scikit-Learn and Streamlit.

The model classifies SMS messages as Spam or Not Spam (Ham) using Natural Language Processing (NLP) techniques.

## Features

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Text Preprocessing
* Stopword Removal
* Stemming
* Count Vectorization
* TF-IDF Transformation
* Multinomial Naive Bayes Classification
* Interactive Streamlit Web Application

## Dataset

SMS Spam Collection Dataset

## Model Performance

* Accuracy: 95.94%
* Precision: 100%
* Recall: 66%

## Model Performance

| Metric | Score |

| Accuracy | 95.94% |
| Precision | 100.00% |
| Recall | 66.67% |

### Confusion Matrix

[[908   0]
 [ 42  84]]

### Interpretation

- The model achieved 95.94% overall accuracy.
- Precision of 100% means every message classified as spam was actually spam.
- Recall of 66.67% indicates that some spam messages were missed and classified as ham.
- The model prioritizes minimizing false positives over maximizing spam detection.

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Streamlit

## Project Workflow

1. Data Cleaning
2. Exploratory Data Analysis
3. Text Preprocessing
4. Feature Engineering
5. Model Building
6. Model Evaluation
7. Streamlit Web App Development

## Key Learnings

During experimentation, it was observed that the model achieved perfect precision but lower recall. This highlighted the trade-off between precision and recall in machine learning classification problems.

Future work includes experimenting with:
- LinearSVC
- Logistic Regression
- N-gram features
- Larger spam datasets

## Future Improvements

* Improve Recall using LinearSVC
* Add more modern spam datasets
* Add confidence scores
* Extend support for Email Spam Detection

## Author

Omkar Srivastava
