# Airline Sentiment Classification using Word2Vec

import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

contractions = {
    "don't": "do not", "can't": "cannot", "i'm": "i am", "he's": "he is",
    "she's": "she is", "it's": "it is", "that's": "that is", "what's": "what is",
    "there's": "there is", "they're": "they are", "i've": "i have", "you've": "you have",
    "we're": "we are", "didn't": "did not", "doesn't": "does not"
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)           
    text = re.sub(r"@\w+", "", text)              
    text = re.sub(r"#\w+", "", text)              
    text = re.sub(r"[^\w\s]", "", text)           

    for word, replacement in contractions.items():
        text = text.replace(word, replacement)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

df = pd.read_csv("Tweets.csv")[["airline_sentiment", "text"]]
df.dropna(subset=["text", "airline_sentiment"], inplace=True)
df['tokens'] = df['text'].apply(preprocess)

w2v_model = api.load("word2vec-google-news-300")

def get_vector(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

df['vector'] = df['tokens'].apply(lambda x: get_vector(x, w2v_model))

X = np.vstack(df['vector'].values)
y = df['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess(tweet)
    vec = get_vector(tokens, w2v_model).reshape(1, -1)
    return model.predict(vec)[0]

print(predict_tweet_sentiment(clf, w2v_model, "I am so happy with the service. Great job!"))
print(predict_tweet_sentiment(clf, w2v_model, "This is the worst airline experience ever."))
