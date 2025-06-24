import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import contextlib
import os
from nltk.data import find

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return tokens

w2v_model = api.load("word2vec-google-news-300")

def get_vector(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

df = pd.read_csv("spam.csv", encoding='latin1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['tokens'] = df['text'].apply(preprocess)
df['vector'] = df['tokens'].apply(lambda x: get_vector(x, w2v_model))

X = np.vstack(df['vector'].values)
y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_message_class(model, w2v_model, message):
    tokens = preprocess(message)
    vec = get_vector(tokens, w2v_model).reshape(1, -1)
    pred = model.predict(vec)[0]
    return 'spam' if pred == 1 else 'ham'

print(predict_message_class(clf, w2v_model, "Congratulations! You've won a free iPhone!"))
print(predict_message_class(clf, w2v_model, "Hey, are we still on for the meeting later?"))
