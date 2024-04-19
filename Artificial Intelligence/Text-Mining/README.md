# Text Mining for Movie Review Sentiment Analysis

## Description

This notebook provides a comprehensive exploration of text mining techniques for Opinion Mining. It highlights the effectiveness of various machine learning algorithms, n-gram methods, and dimensionality reduction approaches in extracting meaningful insights from customer reviews.

It starts with data preparation and preprocessing using methods including stemming, lemmatization, Tf-Idf... It then progresses to sentiment analysis using different machine learning algorithms. A comparison of the accuracy scores on the test set of all implemented models can be found at the start of the notebook.

## Data

The data was taken from file movie_reviews.csv, containing 50000 lines and 2 columns: review and sentiment.

The notebook can be adapted to run with other datasets containing a similar structure with minimal modifications.

## Outline

Given the notebook is written in French, here is the outline translated to English.

1. **Data**
    * Loading Data
    * Data Preparation

2. **Sentiment Analysis**
    * Data Splitting for Training and Testing
    * Numerical Matrix Calculation
        * Stemming/Lemmatization
            * Lemmatization
        * Tf-Idf Step
    * Analysis
        * Decision Tree Model
        * Random Forest Model
            * Other Evaluation Metrics
            * ROC Curve
            * Comparing Train and Test ROC

3. **Text Mining Classifiers (applied on vector space)**
    * Linear Model
    * Multinomial Naive Bayes (MNB)
    * Random Forest Revisited
        * Iteration on Random Forest
        * Two Best Combinations
        * Additional Combination (200 trees, max depth 100)
    * Gradient Boost
        * Regular Gradient Boosting
        * Extreme Gradient Boosting
    * Linear SVM with OneVsRestClassifier
    * SVM Classifier with OneVsRestClassifier

4. **N-grams**
    * Bi-gram Implementation
        * Applied to Base Data
        * Applied to Lemmatized Data
    * Logistic Regression with Bi-grams
        * Base Data
        * Lemmatized Data
    * Cross-validation
    * MNB on Base Data

5. **Dimensionality Reduction with SVD**
    * Apply SVD
    * Apply Random Forest to SVD result
    * Apply Logistic Regression to SVD result

6. **word2vec**
    * Text Tokenization
    * Apply Word2vec

7. **Doc2Vect (Manual & Experimental)**
    * Logit on word2vec