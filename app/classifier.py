from pathlib import Path
import joblib

MODEL_PATH = Path("model/ticket_classifier.joblib")

ROUTING = {
    "Account Access": {"priority": "High", "team": "Account Support"},
    "Billing": {"priority": "Medium", "team": "Billing Support"},
    "Technical Issue": {"priority": "High", "team": "Technical Support"},
    "General Question": {"priority": "Low", "team": "Customer Support"},
}

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python training/train_model.py"
        )
    return joblib.load(MODEL_PATH)

def classify_ticket(text: str, model=None):
    if not text or not text.strip():
        raise ValueError("Ticket text cannot be empty.")

    model = model or load_model()
    category = model.predict([text])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([text])[0]))

    route = ROUTING.get(
        category, {"priority": "Low", "team": "Customer Support"}
    )

    return {
        "category": category,
        "priority": route["priority"],
        "assigned_team": route["team"],
        "confidence": round(confidence, 3) if confidence is not None else None,
    }
