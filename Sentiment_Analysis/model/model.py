# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud, STOPWORDS
from scipy.sparse import hstack
import joblib  # For saving models

# Load your local dataset
df = pd.read_csv('dataset.csv')  # 👈 Replace with your actual filename

# Drop missing values
df.dropna(subset=['review_body', 'sentiment'], inplace=True)

# Convert sentiment to string labels
df['sentiment'] = df['sentiment'].map({1: 'Positive', 0: 'Negative'})

# Visualizations
sns.countplot(x=df['star_rating'])
plt.title("Review Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

sns.countplot(x=df['sentiment'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Wordcloud
def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=250,
        max_font_size=30,
        scale=2,
        random_state=42
    ).generate(str(data))

    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    if title:
        plt.title(title, fontsize=20)
    plt.show()

show_wordcloud(df['review_body'], title="WordCloud of Review Text")

# TF-IDF Vectorization
tfidf_word = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',
                             analyzer='word', stop_words='english',
                             token_pattern=r'\w{1,}', max_features=10000)

tfidf_char = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',
                             analyzer='char', stop_words='english',
                             ngram_range=(2, 6), max_features=50000)

word_features = tfidf_word.fit_transform(df['review_body'])
char_features = tfidf_char.fit_transform(df['review_body'])

features = hstack([char_features, word_features])
labels = df['sentiment']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# --- Linear SVC ---
svc = LinearSVC(class_weight='balanced')
svc.fit(X_train, y_train)

y_test_pred = svc.predict(X_test)
print("Linear SVC - Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Confusion Matrix Plot
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

plot_confusion(y_test, y_test_pred, "Confusion Matrix - Linear SVC")

# --- SGD Classifier ---
sgd = SGDClassifier(class_weight='balanced', max_iter=1000, tol=1e-3, n_jobs=-1)
sgd.fit(X_train, y_train)

y_test_sgd = sgd.predict(X_test)
print("SGD Classifier - Test Accuracy:", accuracy_score(y_test, y_test_sgd))
print(classification_report(y_test, y_test_sgd))

plot_confusion(y_test, y_test_sgd, "Confusion Matrix - SGD Classifier")

# --- Save models and vectorizers ---
joblib.dump(svc, 'svc_model.pkl')
joblib.dump(sgd, 'sgd_model.pkl')
joblib.dump(tfidf_word, 'tfidf_word_vectorizer.pkl')
joblib.dump(tfidf_char, 'tfidf_char_vectorizer.pkl')
print("✅ Models and vectorizers saved successfully.")
