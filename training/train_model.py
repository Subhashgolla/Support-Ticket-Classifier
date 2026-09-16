from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/support_tickets.csv")
MODEL_PATH = Path("model/ticket_classifier.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"],
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions, zero_division=0))
joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
