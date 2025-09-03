# -*- coding: utf-8 -*-
"""
YouTube Comments Spam Classifier

This script implements a machine learning solution to classify YouTube comments as spam or legitimate (ham).
It uses a dataset from the UCI Machine Learning Repository containing comments from five popular YouTube videos.

The script compares multiple classification algorithms and evaluates their performance using various metrics.

Original file: https://colab.research.google.com/drive/1dkb8SOHsojtZh-QRE7OuZcBjMkk34ytk
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
import warnings
import glob
import string
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

# Import sklearn packages for building classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import seaborn as sns
from bs4 import BeautifulSoup

def load_data():
    """
    Load and concatenate all dataset files from the data directory.
    
    Returns:
        pandas.DataFrame: Combined dataset from all CSV files
    """
    all_files = glob.glob("data/*.csv")
    data = pd.concat((pd.read_csv(f, encoding="utf-8-sig") for f in all_files))
    return data

def text_clear(text):
    """
    Clean text by replacing URLs with a token and removing HTML tags.
    
    Args:
        text (str): Raw text to be cleaned
        
    Returns:
        str: Cleaned text
    """
    # Replace all links with a token to represent all links as one
    text = re.sub(r'^https?:\/\/.*[\r\n]*', 'HTTPLINK', text)
    
    # Remove HTML tags and UTF-8 BOM mark
    soup = BeautifulSoup(text)
    return soup.get_text().replace("\ufeff", "")

def text_process(text):
    """
    Process text by removing punctuation and stopwords.
    
    Args:
        text (str): Text to be processed
        
    Returns:
        str: Processed text
    """
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Remove stopwords
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)

def text_to_vector(text, word2idx, vocab_size):
    """
    Convert text to a word vector based on vocabulary.
    
    Args:
        text (str): Text to be converted
        word2idx (dict): Dictionary mapping words to indices
        vocab_size (int): Size of the vocabulary
        
    Returns:
        numpy.ndarray: Vector representation of the text
    """
    word_vector = np.zeros(vocab_size)
    for word in text.split(" "):
        if word2idx.get(word) is None:
            continue
        else:
            word_vector[word2idx.get(word)] += 1
    return np.array(word_vector)

def train_model(clf, features, targets):
    """
    Train a classifier model.
    
    Args:
        clf: Classifier model
        features: Training features
        targets: Target labels
        
    Returns:
        None
    """
    clf.fit(features, targets)

def predict(clf, features):
    """
    Make predictions using a trained classifier.
    
    Args:
        clf: Trained classifier model
        features: Features to predict on
        
    Returns:
        numpy.ndarray: Predictions
    """
    return clf.predict(features)

def evaluate_models(clfs, X_train, X_test, y_train, y_test):
    """
    Evaluate multiple classification models and print their performance metrics.
    
    Args:
        clfs (dict): Dictionary of classifier models
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        list: List of model performance scores
    """
    pred_scores = []
    
    print("Confusion matrixes")
    for k, v in clfs.items():
        train_model(v, X_train, y_train)
        pred = predict(v, X_test)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        
        pred_scores.append((k, accuracy, precision, recall, f1))
        
        cm = confusion_matrix(y_test, pred)
        
        print("\n\nModel: ", k)
        print(pd.DataFrame(cm, columns=["Pred Ham", "Pred Spam"], index=[" Ham", "Spam"]))
    
    return pred_scores

def main():
    """
    Main function to execute the YouTube comment spam classification process.
    """
    # Download required NLTK resources
    nltk.download("punkt")
    nltk.download('stopwords')
    warnings.filterwarnings('ignore')
    
    # Load and preprocess data
    data = load_data()
    print("Dataset loaded successfully. Shape:", data.shape)
    
    # Drop unnecessary columns and clean text
    data = data.drop(["COMMENT_ID", "AUTHOR", "DATE"], axis=1)
    data['CONTENT'] = data['CONTENT'].apply(text_clear)
    
    # Process text (remove punctuation and stopwords)
    data['CONTENT'] = data['CONTENT'].apply(text_process)
    
    # Prepare content and class dataframes
    CONTENT = pd.DataFrame(data['CONTENT'])
    CLASS = pd.DataFrame(data['CLASS'])
    
    # Count word frequencies
    total_counts = Counter()
    for i in range(len(CONTENT)):
        for word in CONTENT.values[i][0].split(" "):
            total_counts[word] += 1
    
    print("Total words in data set: ", len(total_counts))
    
    # Create vocabulary sorted by frequency
    vocab = sorted(total_counts, key=total_counts.get, reverse=True)
    print("Top 60 words:", vocab[:60])
    
    # Create word-to-index mapping
    vocab_size = len(vocab)
    word2idx = {}
    for i, word in enumerate(vocab):
        word2idx[word] = i
    
    # Convert all comments to vectors
    word_vectors = np.zeros((len(CONTENT), len(vocab)), dtype=np.int_)
    for i, (_, text_) in enumerate(CONTENT.iterrows()):
        word_vectors[i] = text_to_vector(text_[0], word2idx, vocab_size)
    
    # Use TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['CONTENT'])
    
    # Use TF-IDF vectors as features
    features = vectors
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, data['CLASS'], test_size=0.15, random_state=99
    )
    
    # Initialize classification models
    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=99)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=31, random_state=99)
    
    # Create dictionary of models
    clfs = {
        'Support Vector': svc,
        'KNeighbors': knc,
        'Multinomial Naive Bayes': mnb,
        'Decision Tree': dtc,
        'Logistic Regression': lrc,
        'Random Forest': rfc
    }
    
    # Evaluate models
    pred_scores = evaluate_models(clfs, X_train, X_test, y_train, y_test)
    
    # Display results as a DataFrame
    results_df = pd.DataFrame(
        pred_scores, 
        columns=["Classification model", "Accuracy", "Precision Score", "Recall Score", "F1 Score"]
    )
    print("\n\n", results_df)
    
    # Visualize results (optional)
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy comparison
    plt.subplot(1, 2, 1)
    sns.barplot(x="Classification model", y="Accuracy", data=results_df)
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    
    # Plot F1 score comparison
    plt.subplot(1, 2, 2)
    sns.barplot(x="Classification model", y="F1 Score", data=results_df)
    plt.title("Model F1 Score Comparison")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("Model comparison visualization saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()