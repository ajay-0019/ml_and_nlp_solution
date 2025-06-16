import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Create synthetic product feedback dataset
good_feedback = [
    "Loved this product", "Very useful item", "Highly recommend", "Works perfectly",
    "Great quality", "Amazing performance", "Absolutely satisfied", "Worth the price",
    "Will buy again", "Impressed with this", "Fast and reliable", "User friendly",
    "Easy to use", "High quality materials", "Very durable", "Excellent customer service",
    "Packaging was nice", "Product works well", "Met expectations", "Five star experience",
    "Happy with purchase", "Does what it says", "Smooth experience", "Functional and neat",
    "Top notch", "Well built", "Perfect fit", "Neat and clean", "Comfortable to use",
    "Reliable brand", "Super helpful", "Practical and sleek", "Just awesome", "Happy customer",
    "Everything as described", "Perfectly designed", "Feels premium", "Loved the color",
    "Great for daily use", "Super fast", "Nice and compact", "A must have", "Very efficient",
    "Clean design", "Loved the experience", "Good battery life", "Very responsive", "Stylish look",
    "Amazing value", "Highly impressed"
]

bad_feedback = [
    "Terrible product", "Not working at all", "Very disappointed", "Worst purchase ever",
    "Broke after one use", "Waste of money", "Would not recommend", "Looks cheap",
    "Completely useless", "Very bad experience", "Not as described", "Poor quality",
    "Packaging was damaged", "Low durability", "Very slow", "Stopped working quickly",
    "Hard to use", "Complicated setup", "Doesn't work properly", "Feels flimsy",
    "Too expensive", "Bad customer support", "Frustrating to use", "Not satisfied",
    "No instructions", "Cheap materials", "Very noisy", "Doesn't fit", "Disappointing performance",
    "Too bulky", "Returned it immediately", "Horrible design", "Faulty item", "Never buying again",
    "Very glitchy", "Looks different", "Short battery life", "Stopped charging", "Misleading ads",
    "Unreliable product", "Uncomfortable to use", "Totally useless", "Wasted my time",
    "Too slow", "Unclear function", "Feels outdated", "Does not respond", "Overheats easily",
    "Not worth it", "Disappointed overall"
]


reviews = good_feedback + bad_feedback
labels = ['good'] * 50 + ['bad'] * 50
data = list(zip(reviews, labels))
random.shuffle(data)
texts, sentiments = zip(*data)

df = pd.DataFrame({
    'Review': texts,
    'Label': sentiments
})


vectorizer = TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['Review']) 
y = df['Label']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))


def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)


sample_reviews = ["I absolutely love this!", "This is a terrible product."]
sample_vectors = text_preprocess_vectorize(sample_reviews, vectorizer)
sample_predictions = model.predict(sample_vectors)
print("\nPredicted Sentiments for sample reviews:")
for review, pred in zip(sample_reviews, sample_predictions):
    print(f'"{review}" → {pred}')
