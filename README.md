# Sentiment Analysis of Financial News Headlines

A text classification project that predicts whether a financial news headline is positive, negative, or neutral. Built around a subtle but important insight: naive text cleaning can accidentally delete the exact words that carry sentiment (e.g. stripping "up"/"down" as stopwords), so a meaningful chunk of this project is diagnosing and fixing that before any model is trained.

**Highlights:**
- Found and fixed a train/test leakage bug caused by deduplicating on raw text instead of cleaned text — verified directly rather than assumed
- Custom cleaning pipeline that preserves direction/negation words and buckets numeric magnitude instead of discarding it
- Compared Naive Bayes vs. Logistic Regression across Bag-of-Words and TF-IDF features
- Improved the best baseline from 76% to 77.7% accuracy through SMOTE rebalancing, bigram features, chi-squared feature selection, and grid search
- Every modeling decision is checked against data, not assumed — including a leakage test that confirms zero shared text between train and test

**Tools:** Python, pandas, scikit-learn, NLTK, imbalanced-learn (SMOTE), matplotlib/seaborn

**Dataset:** [financial_news_headlines_sentiment.csv](https://raw.githubusercontent.com/subashgandyer/datasets/main/financial_news_headlines_sentiment.csv)
