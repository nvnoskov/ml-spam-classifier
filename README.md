# YouTube Spam Comment Classifier

A machine learning project that classifies YouTube comments as spam or legitimate (ham) using various classification algorithms.

## Project Overview

This project implements a text classification system to identify spam comments in YouTube videos. It uses a dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) containing comments from five popular YouTube videos, labeled as spam or ham.

The classifier analyzes text content, preprocesses it, and applies various machine learning algorithms to determine whether a comment is spam. The project compares multiple classification models to find the most effective approach for this specific task.

## Key Features

- Text preprocessing including stopword removal and punctuation cleaning
- Link detection and normalization
- Multiple classification models comparison:
  - Support Vector Machines
  - K-Nearest Neighbors
  - Multinomial Naive Bayes
  - Decision Tree
  - Logistic Regression
  - Random Forest
- Performance evaluation using accuracy, precision, recall, and F1 score
- Confusion matrix visualization for model assessment
- Optimized to reduce false positives

## Dataset

The dataset consists of comments from five YouTube videos:
- PSY - Gangnam Style
- Katy Perry - Dark Horse
- LMFAO - Party Rock Anthem
- Eminem - Love The Way You Lie
- Shakira - Waka Waka

Each comment is labeled as:
- 1 for spam
- 0 for ham (legitimate)

## Results

The project demonstrates significant improvement in reducing false-positive results through careful data preprocessing and model selection. The Random Forest Classifier showed particularly good results after optimization:

| Before | After |
|:---:|:---:|
| <img width="339" alt="image" src="https://user-images.githubusercontent.com/988301/175221768-10e97227-5358-4aa4-bce4-5255ccfda87d.png">|<img width="354" alt="image" src="https://user-images.githubusercontent.com/988301/175221839-e9b50935-8a42-4015-abc4-30d27adadf70.png">|

## Implementation

### Main Components

- `main.py`: Contains the complete implementation including:
  - Data loading and preprocessing
  - Feature extraction
  - Model training and evaluation
  - Results visualization

### Notebooks

Two Jupyter notebooks are provided for interactive exploration:

1. **First Iteration**: `first.ipynb` 
   - Initial approach following standard guidelines
   - [Google Colab Link](https://colab.research.google.com/drive/1QlD017zCSuuf9efB5mNS_Xycso71rpLN?usp=sharing)

2. **Second Iteration**: `notebook.ipynb`
   - Improved approach with focus on data quality
   - Reduced false-positive results by half
   - [Google Colab Link](https://colab.research.google.com/drive/1dkb8SOHsojtZh-QRE7OuZcBjMkk34ytk#scrollTo=1d6fbe52-61c7-4613-94d8-b6b75d4cf9b2)

## Technical Approach

1. **Data Preprocessing**:
   - HTML tag removal
   - URL normalization (replacing links with a token)
   - Punctuation removal
   - Stopword filtering
   - Text normalization

2. **Feature Extraction**:
   - TF-IDF vectorization
   - Word frequency analysis

3. **Model Training and Evaluation**:
   - Train-test split (85% training, 15% testing)
   - Multiple model training
   - Performance evaluation using confusion matrices and metrics
   - Model comparison

## Requirements

The project requires the following Python packages:
- jupyter
- matplotlib
- numpy
- pandas
- scipy
- scikit-learn
- nltk
- beautifulsoup4
- seaborn
- wordcloud

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/nvnoskov/ml-spam-classifier.git
cd ml-spam-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python main.py
```

4. For interactive exploration, open the Jupyter notebooks:
```bash
jupyter notebook notebook.ipynb
```

## Project Structure

```
ml-spam-classifier/
├── data/                      # Dataset files
│   ├── Youtube01-Psy.csv
│   ├── Youtube02-KatyPerry.csv
│   ├── Youtube03-LMFAO.csv
│   ├── Youtube04-Eminem.csv
│   └── Youtube05-Shakira.csv
├── first.ipynb                # First iteration notebook
├── notebook.ipynb             # Second iteration notebook
├── main.py                    # Main Python implementation
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── Text Classification Exercise.pdf  # Original task description
```

## License

This project is available for educational and research purposes.

## Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- Original task requirements from "Text Classification Exercise"