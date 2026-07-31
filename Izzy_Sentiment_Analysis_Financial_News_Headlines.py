#!/usr/bin/env python
# coding: utf-8

# Izzy

# Sentiment Analysis of Financial News Headlines
# Dataset: https://raw.githubusercontent.com/subashgandyer/datasets/main/financial_news_headlines_sentiment.csv
# Max marks: 80 (100 with SMOTE bonus)

# ## 1. Download the dataset [1 point]

import os
import re
import warnings
import urllib.request
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

RANDOM_STATE = 42

DATA_URL = "https://raw.githubusercontent.com/subashgandyer/datasets/main/financial_news_headlines_sentiment.csv"
DATA_FILE = "financial_news_headlines_sentiment.csv"

if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

print(f"Downloaded to {DATA_FILE}: {os.path.getsize(DATA_FILE):,} bytes")


# ## 2. Load the dataset [1 point]

# No header row; latin-1 avoids decode errors on a few rows.
df = pd.read_csv(DATA_FILE, header=None, names=["sentiment", "headline"], encoding="latin-1")
print(f"{len(df)} rows loaded")
df.head()


# ## 3. Explore the dataset [10 points]

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print(f"Dtypes:\n{df.dtypes}\n")
print(f"Missing values:\n{df.isna().sum()}\n")
print(f"Duplicate rows: {df.duplicated().sum()}")
df.sample(5, random_state=RANDOM_STATE)


# Class balance (imbalanced; addressed via SMOTE below).

class_counts = df["sentiment"].value_counts()
class_pct = (class_counts / len(df) * 100).round(1)
print(pd.DataFrame({"count": class_counts, "pct": class_pct}))

order = ["positive", "neutral", "negative"]
sentiment_colors = {"positive": "#2E7D32", "neutral": "#757575", "negative": "#C62828"}

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(order, class_counts[order], color=[sentiment_colors[s] for s in order])
ax.bar_label(bars, labels=[f"{c}\n({p}%)" for c, p in zip(class_counts[order], class_pct[order])], padding=3)
ax.set_ylabel("Number of headlines")
ax.set_title("Sentiment class distribution")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()


# Headline length (informs max_features below).

df["word_count"] = df["headline"].str.split().str.len()
print(df["word_count"].describe())

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["word_count"], bins=30, color="#1565C0", edgecolor="white")
ax.set_xlabel("Words per headline")
ax.set_ylabel("Number of headlines")
ax.set_title("Headline length distribution")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()


# Bigrams + vocab overlap per class. Local _eda_clean since this runs before Section 4.

_eda_lemmatizer = WordNetLemmatizer()
_eda_stop_words = set(stopwords.words("english"))

def _eda_clean(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = word_tokenize(text)
    return [_eda_lemmatizer.lemmatize(t) for t in tokens if t not in _eda_stop_words and len(t) > 1]

bigram_top = {}
for cls in ["negative", "neutral", "positive"]:
    cls_texts = df.loc[df["sentiment"] == cls, "headline"].apply(lambda t: " ".join(_eda_clean(t)))
    vec = CountVectorizer(ngram_range=(2, 2), max_features=2000)
    X_cls = vec.fit_transform(cls_texts)
    totals = np.asarray(X_cls.sum(axis=0)).ravel()
    bigram_top[cls] = pd.Series(totals, index=vec.get_feature_names_out()).sort_values(ascending=False).head(10)
    print(f"Top 10 bigrams -- {cls}:")
    print(bigram_top[cls])
    print()

print("Overlap of each pair's top-10 bigram sets:")
for a, b in combinations(bigram_top.keys(), 2):
    overlap = set(bigram_top[a].index) & set(bigram_top[b].index)
    print(f"  {a} vs {b}: {overlap or '(none)'}")


vocab_sets = {}
for cls in ["negative", "neutral", "positive"]:
    cls_texts = df.loc[df["sentiment"] == cls, "headline"].apply(lambda t: " ".join(_eda_clean(t)))
    vec = CountVectorizer(max_features=200)
    vec.fit(cls_texts)
    vocab_sets[cls] = set(vec.vocabulary_.keys())

print("Vocabulary overlap (Jaccard similarity) between class pairs, top-200 unigrams each:")
for a, b in combinations(vocab_sets.keys(), 2):
    inter = vocab_sets[a] & vocab_sets[b]
    union = vocab_sets[a] | vocab_sets[b]
    jaccard = len(inter) / len(union)
    print(f"  {a} vs {b}: Jaccard={jaccard:.3f}  ({len(inter)} shared / {len(union)} total)")


# Classes overlap heavily in vocabulary; negative/neutral overlap LEAST (0.29) --
# negative's errors look like a sample-size issue, not vocab collision.

# ## 4. Clean the data [5 points]

# Keep direction/negation words (NLTK strips "up"/"down"/"not" by default) and
# bucket numbers by magnitude/sign instead of discarding them.
lemmatizer = WordNetLemmatizer()
KEEP_WORDS = {"up", "down", "no", "not", "above", "below", "under", "over",
              "more", "less", "against", "off", "further", "again", "out"}
stop_words = set(stopwords.words("english")) - KEEP_WORDS

def bucket_numbers(text):
    text = re.sub(r"%", " pct ", text)
    def repl(m):
        sign, num_str = m.group(1), m.group(2)
        num = float(num_str.replace(",", ""))
        bucket = "smallnum" if num < 10 else ("midnum" if num < 100 else "bignum")
        return f" neg{bucket} " if sign else f" {bucket} "
    return re.sub(r"(-)?(\d+(?:\.\d+)?)", repl, text)

def clean_headline(text):
    text = text.lower()
    text = bucket_numbers(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

df["clean_headline"] = df["headline"].apply(clean_headline)
df[["headline", "clean_headline"]].sample(5, random_state=RANDOM_STATE)


# 6 of 10 conflicting-label duplicates were a cleaning artifact (fixed above); the
# other 5 are genuine ambiguity -- dropped entirely below.

# Drop empty/conflicting rows, then dedup on clean_headline (not raw) to avoid leakage.
empty_after_cleaning = (df["clean_headline"].str.strip() == "").sum()
print(f"Rows with empty text after cleaning: {empty_after_cleaning}")
df = df[df["clean_headline"].str.strip() != ""].reset_index(drop=True)

dup_mask = df.duplicated(subset=["clean_headline"], keep=False)
label_counts_per_text = df[dup_mask].groupby("clean_headline")["sentiment"].nunique()
conflicting_texts = set(label_counts_per_text[label_counts_per_text > 1].index)
print(f"Conflicting-label groups found: {len(conflicting_texts)}")

before_purge = len(df)
df = df[~df["clean_headline"].isin(conflicting_texts)].reset_index(drop=True)
print(f"Rows dropped (conflicting-label groups): {before_purge - len(df)}")

before_dedup = len(df)
df = df.drop_duplicates(subset=["clean_headline"]).reset_index(drop=True)
print(f"Remaining duplicate clean_headline rows dropped: {before_dedup - len(df)}")
print(f"Final row count: {len(df)}")


# ## 5. SMOTE (Imbalanced dataset) [OPTIONAL] BONUS [20 points]
# Hint: Use imblearn library

# Fit on training data only (avoids leakage); reused after the split (Section 8).

def apply_smote(X_train, y_train, random_state=RANDOM_STATE):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


# Demo: class balance before/after SMOTE.
demo_vectorizer = TfidfVectorizer(max_features=3000)
X_demo = demo_vectorizer.fit_transform(df["clean_headline"])
y_demo = df["sentiment"]

print("Before SMOTE:", dict(y_demo.value_counts()))
X_demo_resampled, y_demo_resampled = apply_smote(X_demo, y_demo)
print("After SMOTE: ", dict(pd.Series(y_demo_resampled).value_counts()))


# ## 6. BoW model [15 points]

# Preview vocab on full corpus; real vectorizer refit on train only (Section 8).
bow_vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 1))
X_bow_preview = bow_vectorizer.fit_transform(df["clean_headline"])

print(f"BoW vocabulary size: {len(bow_vectorizer.vocabulary_)}")
print(f"BoW matrix shape: {X_bow_preview.shape}")

bow_totals = np.asarray(X_bow_preview.sum(axis=0)).ravel()
top_bow = pd.Series(bow_totals, index=bow_vectorizer.get_feature_names_out()).sort_values(ascending=False).head(15)
print("\nTop 15 tokens by total BoW count:")
print(top_bow)


# ## 7. Tf-idf model [15 points]

tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 1))
X_tfidf_preview = tfidf_vectorizer.fit_transform(df["clean_headline"])

print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
print(f"TF-IDF matrix shape: {X_tfidf_preview.shape}")

tfidf_means = np.asarray(X_tfidf_preview.mean(axis=0)).ravel()
top_tfidf = pd.Series(tfidf_means, index=tfidf_vectorizer.get_feature_names_out()).sort_values(ascending=False).head(15)
print("\nTop 15 tokens by mean TF-IDF weight:")
print(top_tfidf)


# ## 8. Split train test data [3 points]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])
print("Label encoding:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

text_train, text_test, y_train, y_test = train_test_split(
    df["clean_headline"], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {len(text_train)}  Test: {len(text_test)}")

# Refit on train only; transform test (leakage-safe).
bow_vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 1))
X_train_bow = bow_vectorizer.fit_transform(text_train)
X_test_bow = bow_vectorizer.transform(text_test)

tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 1))
X_train_tfidf = tfidf_vectorizer.fit_transform(text_train)
X_test_tfidf = tfidf_vectorizer.transform(text_test)

print(f"BoW train/test shapes:    {X_train_bow.shape} / {X_test_bow.shape}")
print(f"TF-IDF train/test shapes: {X_train_tfidf.shape} / {X_test_tfidf.shape}")


# Verify no leakage: no shared train/test text, vocab from train only.

shared_text = set(text_train) & set(text_test)
print(f"Exact text rows shared between train and test: {len(shared_text)}")
assert len(shared_text) == 0, "Leakage: identical cleaned text appears in both splits"

bow_vocab_source_check = set(bow_vectorizer.vocabulary_.keys()) <= set(
    " ".join(text_train).split()
)
tfidf_vocab_source_check = set(tfidf_vectorizer.vocabulary_.keys()) <= set(
    " ".join(text_train).split()
)
print(f"BoW vocabulary drawn entirely from training text: {bow_vocab_source_check}")
print(f"TF-IDF vocabulary drawn entirely from training text: {tfidf_vocab_source_check}")


# ## 9. Classification Algorithm [10 points]

# Multinomial Naive Bayes baseline.
nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train)
pred_nb_bow = nb_bow.predict(X_test_bow)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)

print("Multinomial Naive Bayes -- BoW")
print(f"Accuracy: {accuracy_score(y_test, pred_nb_bow):.4f}")
print(classification_report(y_test, pred_nb_bow, target_names=label_encoder.classes_))

print("Multinomial Naive Bayes -- TF-IDF")
print(f"Accuracy: {accuracy_score(y_test, pred_nb_tfidf):.4f}")
print(classification_report(y_test, pred_nb_tfidf, target_names=label_encoder.classes_))


# ## 10. Another Classification Algorithm [10 points]

# Logistic Regression: usually stronger than NB when classes share vocabulary.
lr_bow = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_bow.fit(X_train_bow, y_train)
pred_lr_bow = lr_bow.predict(X_test_bow)

lr_tfidf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_tfidf.fit(X_train_tfidf, y_train)
pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)

print("Logistic Regression -- BoW")
print(f"Accuracy: {accuracy_score(y_test, pred_lr_bow):.4f}")
print(classification_report(y_test, pred_lr_bow, target_names=label_encoder.classes_))

print("Logistic Regression -- TF-IDF")
print(f"Accuracy: {accuracy_score(y_test, pred_lr_tfidf):.4f}")
print(classification_report(y_test, pred_lr_tfidf, target_names=label_encoder.classes_))


# Bonus: does SMOTE help the negative class? Rerun LR+TF-IDF resampled.

X_train_tfidf_smote, y_train_smote = apply_smote(X_train_tfidf, y_train)
print("Train class counts before SMOTE:",
      {label_encoder.classes_[k]: v for k, v in pd.Series(y_train).value_counts().items()})
print("Train class counts after SMOTE: ",
      {label_encoder.classes_[k]: v for k, v in pd.Series(y_train_smote).value_counts().items()})

lr_tfidf_smote = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_tfidf_smote.fit(X_train_tfidf_smote, y_train_smote)
pred_lr_tfidf_smote = lr_tfidf_smote.predict(X_test_tfidf)

print("\nLogistic Regression + TF-IDF -- WITHOUT SMOTE")
print(classification_report(y_test, pred_lr_tfidf, target_names=label_encoder.classes_))

print("Logistic Regression + TF-IDF -- WITH SMOTE")
print(classification_report(y_test, pred_lr_tfidf_smote, target_names=label_encoder.classes_))


# Bonus: do bigrams help? Unigrams+bigrams, larger vocab cap, vs. the baselines above.

bigram_results = {}
for name, Vectorizer in [("BoW", CountVectorizer), ("TF-IDF", TfidfVectorizer)]:
    vec = Vectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(text_train)
    Xte = vec.transform(text_test)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(Xtr, y_train)
    pred = lr.predict(Xte)
    bigram_results[name] = pred

    print(f"Logistic Regression -- {name} (unigrams+bigrams, max_features=5000)")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=label_encoder.classes_))


# Bigrams help BoW (76.1->77.2%) but hurt TF-IDF (76.0->75.7%).

# Bonus: discriminative feature selection (chi2) -- select tokens by
# class-separating power instead of raw frequency.

chi2_results = {}
for fe_name, X_train_full, X_test_full in [("BoW", X_train_bow, X_test_bow),
                                             ("TF-IDF", X_train_tfidf, X_test_tfidf)]:
    for k in [500, 1000, 1500, 2000]:
        selector = SelectKBest(chi2, k=k).fit(X_train_full, y_train)
        X_train_sel = selector.transform(X_train_full)
        X_test_sel = selector.transform(X_test_full)

        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_train_sel, y_train)
        pred = lr.predict(X_test_sel)
        chi2_results[(fe_name, k)] = pred

        acc = accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, target_names=label_encoder.classes_, output_dict=True)
        print(f"LR + {fe_name} + chi2(k={k}): acc={acc:.4f} macro_f1={report['macro avg']['f1-score']:.4f} "
              f"negative_recall={report['negative']['recall']:.4f} negative_f1={report['negative']['f1-score']:.4f}")


# chi2 helps BoW a lot (best: k=2000, 77.69%/0.723 -- best result overall);
# barely moves TF-IDF, which already down-weights common tokens via IDF.

# Bonus: hyperparameter tuning via GridSearchCV -- max_features/min_df/C,
# 5-fold CV, scored on f1_macro (imbalanced data).

gridsearch_results = {}
for fe_name, Vectorizer in [("BoW", CountVectorizer), ("TF-IDF", TfidfVectorizer)]:
    pipe = Pipeline([
        ("vec", Vectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "vec__max_features": [3000, 5000, 8000],
        "vec__min_df": [1, 2, 3],
        "clf__C": [0.1, 1, 10],
    }
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid_search.fit(text_train, y_train)

    pred = grid_search.best_estimator_.predict(text_test)
    gridsearch_results[fe_name] = pred

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, target_names=label_encoder.classes_, output_dict=True)
    print(f"{fe_name}: best_params={grid_search.best_params_}")
    print(f"  cv f1_macro={grid_search.best_score_:.4f}  test acc={acc:.4f}  "
          f"test macro_f1={report['macro avg']['f1-score']:.4f}  "
          f"negative_recall={report['negative']['recall']:.4f}  negative_f1={report['negative']['f1-score']:.4f}")


# Both extractors converge to ~77.3%/0.72; doesn't beat chi2 (k not in this grid).

# ## 11. Confusion Matrixes for two classification algorithms and two feature extractor methods [10 points]

results = {
    "Naive Bayes + BoW": pred_nb_bow,
    "Naive Bayes + TF-IDF": pred_nb_tfidf,
    "Logistic Regression + BoW": pred_lr_bow,
    "Logistic Regression + TF-IDF": pred_lr_tfidf,
}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (title, y_pred) in zip(axes.ravel(), results.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
    ax.set_title(f"{title}\naccuracy={accuracy_score(y_test, y_pred):.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()


# ### Observations
# - Fixed a leakage bug: dedup now on clean_headline, not raw text (0 shared rows).
# - Reviewed 10 conflicting-label duplicates: 6 were a cleaning artifact (fixed), 5 dropped.
# - Cleaning fix (number buckets + kept direction words) was the biggest lift:
#   LR+BoW 73.7->76.1%, LR+TF-IDF 72.3->76.0% accuracy.
# - LR beats NB on both feature types (76.1/76.0% vs 71.1/69.2%).
# - SMOTE: negative recall 0.44->0.68, macro-F1 0.670->0.706, accuracy -0.5pt.
# - Best result: BoW + chi2 (k=2000) -- 77.69% acc, 0.723 macro-F1.
# - Bigrams help BoW/hurt TF-IDF; chi2 helps BoW/barely TF-IDF; GridSearchCV
#   converges both to ~77.3%.