#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning Script for YouTube Spam Comment Classifier

This script performs hyperparameter tuning for various classification models
to find the optimal parameters for the YouTube comment spam classification task.
"""

import pandas as pd
import numpy as np
import glob
import re
import string
import warnings
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download('stopwords', quiet=True)

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

def main():
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_data()
    
    # Drop unnecessary columns and clean text
    data = data.drop(["COMMENT_ID", "AUTHOR", "DATE"], axis=1)
    data['CONTENT'] = data['CONTENT'].apply(text_clear)
    data['CONTENT'] = data['CONTENT'].apply(text_process)
    
    # Split data into features and target
    X = data['CONTENT']
    y = data['CLASS']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=99
    )
    
    # Create a directory for results
    import os
    if not os.path.exists('tuning_results'):
        os.makedirs('tuning_results')
    
    # Define parameter grids for each classifier
    param_grids = {
        'SVM': {
            'vect__max_features': [2000, 3000, 5000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['linear', 'rbf', 'sigmoid'],
            'clf__gamma': ['scale', 'auto', 1.0]
        },
        'NaiveBayes': {
            'vect__max_features': [2000, 3000, 5000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'clf__alpha': [0.01, 0.1, 0.2, 0.5, 1.0]
        },
        'DecisionTree': {
            'vect__max_features': [2000, 3000, 5000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [None, 10, 20, 30],
            'clf__min_samples_split': [2, 5, 7, 10]
        },
        'KNN': {
            'vect__max_features': [2000, 3000, 5000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'clf__n_neighbors': [3, 5, 7, 9, 11, 21, 31, 41, 49]
        },
        'RandomForest': {
            'vect__max_features': [2000, 3000, 5000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'clf__n_estimators': [10, 31, 50, 100],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [None, 10, 20, 30],
            'clf__min_samples_split': [2, 5, 7, 10]
        },
        'LogisticRegression': {
            'vect__max_features': [2000, 3000, 5000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']
        }
    }
    
    # Define classifiers
    classifiers = {
        'SVM': SVC(random_state=99),
        'NaiveBayes': MultinomialNB(),
        'DecisionTree': DecisionTreeClassifier(random_state=99),
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(random_state=99),
        'LogisticRegression': LogisticRegression(random_state=99)
    }
    
    # Store best models and their scores
    best_models = {}
    best_scores = {}
    
    # Perform grid search for each classifier
    for name, clf in classifiers.items():
        print(f"\nPerforming grid search for {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', clf)
        ])
        
        # Create grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[name], 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store scores
        best_scores[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_params': grid_search.best_params_
        }
        
        # Print results
        print(f"\nBest parameters for {name}:")
        print(grid_search.best_params_)
        print(f"\nTest set performance for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"tuning_results/{name}_classification_report.csv")
        
        # Save best parameters
        with open(f"tuning_results/{name}_best_params.txt", "w") as f:
            for param, value in grid_search.best_params_.items():
                f.write(f"{param}: {value}\n")
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Model': list(best_scores.keys()),
        'Accuracy': [best_scores[model]['accuracy'] for model in best_scores],
        'Precision': [best_scores[model]['precision'] for model in best_scores],
        'Recall': [best_scores[model]['recall'] for model in best_scores],
        'F1 Score': [best_scores[model]['f1'] for model in best_scores]
    })
    
    # Sort by F1 score
    comparison = comparison.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    # Save comparison
    comparison.to_csv("tuning_results/model_comparison.csv", index=False)
    
    # Print comparison
    print("\nModel Comparison (sorted by F1 Score):")
    print(comparison)
    
    # Identify best overall model
    best_model_name = comparison.iloc[0]['Model']
    print(f"\nBest overall model: {best_model_name}")
    print(f"Best parameters: {best_scores[best_model_name]['best_params']}")
    print(f"F1 Score: {best_scores[best_model_name]['f1']:.4f}")
    
    print("\nHyperparameter tuning completed. Results saved to 'tuning_results' directory.")

if __name__ == "__main__":
    main()