from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.classifier import classify_ticket

app = FastAPI(title="Support Ticket Classifier", version="1.0.0")

class TicketRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return FileResponse(Path("app/static/index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/classify")
def classify(request: TicketRequest):
    try:
        return classify_ticket(request.text)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
