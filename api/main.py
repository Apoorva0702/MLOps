from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
import os
import sqlite3
import requests
import logging
import requests
import logging
import hvac
from logstash_async.handler import AsynchronousLogstashHandler

# Logstash Configuration
LOGSTASH_HOST = os.environ.get('LOGSTASH_HOST', 'logstash')
LOGSTASH_PORT = 5000

logstash_handler = AsynchronousLogstashHandler(
    host=LOGSTASH_HOST, 
    port=LOGSTASH_PORT, 
    database_path='logstash.db'
)

# Root Logger Config
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Keep console handler
logger.addHandler(logging.StreamHandler())
# Add logstash handler
logger.addHandler(logstash_handler)

app = FastAPI()

# Vault Configuration (Mock credentials pull)
def fetch_vault_secrets():
    print("Attempting to fetch secrets from Vault...")
    try:
        client = hvac.Client(url='http://vault:8200', token='dev-only-token')
        if client.is_authenticated():
             print("Vault authenticated.")
             # Dummy secret grab
             # secret_version_response = client.secrets.kv.v2.read_secret_version(path='fake-news-secret')
             # print("Secret grabbed.")
    except Exception as e:
        logger.warning(f"Could not connect to Vault: {e}")

fetch_vault_secrets()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "../frontend")), name="static")

MODEL_PATH = os.path.join(BASE_DIR, "../model/bert_fake_news_model")
DB_PATH = os.path.join(BASE_DIR, "../db/fake_news.db")

model_name_or_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# ---------- Request Models ----------

class InputText(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    id: int
    correct_label: str


# ---------- Home ----------

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "../frontend/index.html"))


# ---------- Predict ----------

@app.post("/predict")
def predict(data: InputText):

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0][pred].item()

    label_map = {0: "FAKE", 1: "REAL"}
    label = label_map[pred]

    # Insert into SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO news_predictions
        (article_text, predicted_label, confidence)
        VALUES (?, ?, ?)
    """, (data.text, label, confidence))

    row_id = cur.lastrowid

    conn.commit()
    conn.close()

    return {
        "id": row_id,
        "prediction": label,
        "confidence": round(confidence, 4)
    }


# ---------- Feedback ----------

@app.post("/feedback")
def feedback(data: FeedbackInput):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get predicted label
    cur.execute(
        "SELECT predicted_label FROM news_predictions WHERE id=?",
        (data.id,)
    )

    row = cur.fetchone()

    if not row:
        conn.close()
        return {"error": "Record not found"}

    predicted = row[0]

    misclassified = 1 if predicted != data.correct_label.upper() else 0

    cur.execute("""
        UPDATE news_predictions
        SET correct_label=?,
            feedback_given=1,
            is_misclassified=?
        WHERE id=?
    """, (data.correct_label.upper(), misclassified, data.id))

    if misclassified == 1:
        cur.execute("SELECT COUNT(*) FROM news_predictions WHERE is_misclassified=1")
        count = cur.fetchone()[0]
        if count > 0 and count % 10 == 0:
            logging.warning(f"Misclassification count hit {count}. Triggering Jenkins retraining pipeline!")
            try:
                # Replace with actual Jenkins webhook URL / token authentication
                webhook_url = "http://jenkins:8080/job/Fake-News-Retraining/build"
                requests.post(webhook_url, timeout=5)
                logger.info("Successfully triggered Jenkins webhook for retraining.")
            except Exception as e:
                logger.error(f"Error triggering Jenkins: {e}")

    conn.commit()
    conn.close()

    return {
        "id": data.id,
        "predicted_label": predicted,
        "correct_label": data.correct_label.upper(),
        "is_misclassified": misclassified
    }