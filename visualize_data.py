#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Visualization Script for YouTube Spam Comment Classifier

This script visualizes the distribution of spam and legitimate comments in the dataset,
as well as other characteristics that might be useful for understanding the data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob
from collections import Counter
import re
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
import nltk

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

def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def main():
    # Load data
    data = load_data()
    print(f"Dataset loaded successfully. Total comments: {len(data)}")
    
    # Create a directory for visualizations
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Visualize class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='CLASS', data=data)
    plt.title('Distribution of Spam vs. Ham Comments')
    plt.xlabel('Class (0: Ham, 1: Spam)')
    plt.ylabel('Count')
    plt.savefig('visualizations/class_distribution.png')
    print("Class distribution visualization saved.")
    
    # 2. Visualize distribution by video source
    plt.figure(figsize=(12, 6))
    video_counts = data['COMMENT_ID'].str.split('_', expand=True)[0].value_counts()
    sns.barplot(x=video_counts.index, y=video_counts.values)
    plt.title('Comment Distribution by Video')
    plt.xlabel('Video ID')
    plt.ylabel('Number of Comments')
    plt.savefig('visualizations/video_distribution.png')
    print("Video distribution visualization saved.")
    
    # 3. Visualize spam percentage by video
    plt.figure(figsize=(12, 6))
    video_id = data['COMMENT_ID'].str.split('_', expand=True)[0]
    spam_by_video = pd.DataFrame({
        'video': video_id,
        'is_spam': data['CLASS']
    }).groupby('video').mean()
    
    sns.barplot(x=spam_by_video.index, y=spam_by_video['is_spam'])
    plt.title('Spam Percentage by Video')
    plt.xlabel('Video ID')
    plt.ylabel('Spam Percentage')
    plt.savefig('visualizations/spam_percentage_by_video.png')
    print("Spam percentage by video visualization saved.")
    
    # 4. Clean and process text
    data['CLEAN_CONTENT'] = data['CONTENT'].apply(text_clear)
    data['PROCESSED_CONTENT'] = data['CLEAN_CONTENT'].apply(text_process)
    
    # 5. Visualize comment length distribution
    data['WORD_COUNT'] = data['CLEAN_CONTENT'].apply(count_words)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='WORD_COUNT', hue='CLASS', bins=30, kde=True)
    plt.title('Comment Length Distribution by Class')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.xlim(0, 50)  # Limit x-axis to focus on the main distribution
    plt.savefig('visualizations/comment_length_distribution.png')
    print("Comment length distribution visualization saved.")
    
    # 6. Create word clouds for spam and ham comments
    spam_text = ' '.join(data[data['CLASS'] == 1]['PROCESSED_CONTENT'])
    ham_text = ' '.join(data[data['CLASS'] == 0]['PROCESSED_CONTENT'])
    
    # Word cloud for spam comments
    plt.figure(figsize=(10, 8))
    wordcloud_spam = WordCloud(width=800, height=600, background_color='white', 
                              max_words=100, contour_width=3, contour_color='steelblue').generate(spam_text)
    plt.imshow(wordcloud_spam, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Spam Comments')
    plt.savefig('visualizations/spam_wordcloud.png')
    print("Spam word cloud visualization saved.")
    
    # Word cloud for ham comments
    plt.figure(figsize=(10, 8))
    wordcloud_ham = WordCloud(width=800, height=600, background_color='white', 
                             max_words=100, contour_width=3, contour_color='steelblue').generate(ham_text)
    plt.imshow(wordcloud_ham, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Ham Comments')
    plt.savefig('visualizations/ham_wordcloud.png')
    print("Ham word cloud visualization saved.")
    
    # 7. Generate summary statistics
    print("\nSummary Statistics:")
    print(f"Total number of comments: {len(data)}")
    print(f"Number of spam comments: {sum(data['CLASS'])}")
    print(f"Number of ham comments: {len(data) - sum(data['CLASS'])}")
    print(f"Spam percentage: {sum(data['CLASS']) / len(data) * 100:.2f}%")
    print(f"Average comment length (words): {data['WORD_COUNT'].mean():.2f}")
    print(f"Average spam comment length: {data[data['CLASS'] == 1]['WORD_COUNT'].mean():.2f}")
    print(f"Average ham comment length: {data[data['CLASS'] == 0]['WORD_COUNT'].mean():.2f}")
    
    # Save summary statistics to file
    with open('visualizations/summary_statistics.txt', 'w') as f:
        f.write(f"Total number of comments: {len(data)}\n")
        f.write(f"Number of spam comments: {sum(data['CLASS'])}\n")
        f.write(f"Number of ham comments: {len(data) - sum(data['CLASS'])}\n")
        f.write(f"Spam percentage: {sum(data['CLASS']) / len(data) * 100:.2f}%\n")
        f.write(f"Average comment length (words): {data['WORD_COUNT'].mean():.2f}\n")
        f.write(f"Average spam comment length: {data[data['CLASS'] == 1]['WORD_COUNT'].mean():.2f}\n")
        f.write(f"Average ham comment length: {data[data['CLASS'] == 0]['WORD_COUNT'].mean():.2f}\n")
    
    print("\nAll visualizations and statistics have been saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main()