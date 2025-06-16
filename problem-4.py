import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Create synthetic dataset
positive_reviews = [
    "Absolutely loved it!", "Great acting and story.", "A must-watch.", "Fantastic cinematography.",
    "Highly recommended.", "Brilliant performance.", "Heartwarming and touching.", "Loved every moment.",
    "Truly inspiring.", "Excellent direction.", "Wonderful cast.", "Engaging from start to end.",
    "Superb experience.", "It was amazing!", "Will watch again.", "Stunning visuals.", "A masterpiece.",
    "Beautifully made.", "One of the best movies ever.", "Impressive work.", "Perfectly executed.",
    "Touching storyline.", "Incredible movie.", "Very well done.", "Top-notch acting.",
    "Five stars!", "Enjoyed a lot.", "Smart and entertaining.", "Loved the characters.", "Pure brilliance.",
    "Truly enjoyable.", "Outstanding direction.", "Flawless plot.", "Made me laugh and cry.",
    "Unforgettable.", "A cinematic gem.", "Magical storytelling.", "So entertaining!",
    "Emotional and powerful.", "Masterclass in film.", "Creative and original.",
    "Absolutely perfect.", "Great from beginning to end.", "One of my favorites.",
    "Loved the soundtrack.", "Amazing production.", "Full of emotions.", "Very touching.",
    "Completely satisfied.", "Well crafted."
]

negative_reviews = [
    "Terrible movie.", "Worst acting ever.", "Don't waste your time.", "Boring and slow.",
    "Very disappointing.", "Bad script.", "Unwatchable.", "Awful experience.", "Not worth it.",
    "Zero stars.", "Painful to watch.", "Poorly made.", "Too predictable.", "Terrible direction.",
    "Lacked emotion.", "Overhyped and dull.", "A complete mess.", "Terribly boring.",
    "Nothing special.", "Ridiculously bad.", "Poor editing.", "I almost slept.",
    "No chemistry between actors.", "Script was weak.", "Cringe-worthy.",
    "Felt like forever.", "Waste of talent.", "Disgusting plot.", "Didn't enjoy at all.",
    "Very badly done.", "Terrible effects.", "Super boring.", "Unrealistic and silly.",
    "I regret watching.", "Low-budget feel.", "Too long and pointless.",
    "Worst film I've seen.", "No character depth.", "Predictable ending.", "Forced emotions.",
    "I left the theatre early.", "Not my type.", "Very repetitive.", "Bad acting and direction.",
    "Plot made no sense.", "Just bad.", "Flat storyline.", "Characters were annoying.",
    "Soundtrack was terrible.", "Really disappointing.", "Avoid it."
]


reviews = positive_reviews + negative_reviews
sentiments = ['positive'] * 50 + ['negative'] * 50
combined = list(zip(reviews, sentiments))
random.shuffle(combined)


shuffled_reviews, shuffled_sentiments = zip(*combined)
df = pd.DataFrame({
    'Review': shuffled_reviews,
    'Sentiment': shuffled_sentiments
})


vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])  
y = df['Sentiment']                        

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])  
    prediction = model.predict(review_vector)
    return prediction[0]

test_review = "The movie was full of emotion and truly amazing."
print("Predicted Sentiment:", predict_review_sentiment(model, vectorizer, test_review))
