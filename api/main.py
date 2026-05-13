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

JENKINS_AUTH = None

def fetch_vault_secrets():
    global JENKINS_AUTH
    logger.info("Attempting to fetch secrets from HashiCorp Vault...")
    vault_url = os.environ.get('VAULT_ADDR', 'http://vault:8200')
    vault_token = os.environ.get('VAULT_TOKEN')  # Should be provided via environment variable
    
    if not vault_token:
        logger.warning("VAULT_TOKEN not set. Vault integration will be skipped.")
        return

    try:
        client = hvac.Client(url=vault_url, token=vault_token)
        if client.is_authenticated():
             # Attempt to read Jenkins credentials from Vault KV engine
             try:
                 read_response = client.secrets.kv.v2.read_secret_version(path='jenkins', mount_point='secret')
                 secrets = read_response['data']['data']
                 JENKINS_AUTH = (secrets['username'], secrets['password'])
                 logger.info("Successfully authenticated and fetched Jenkins credentials from Vault.")
             except Exception as e:
                 logger.error(f"Vault authenticated but failed to read 'jenkins' secret: {e}")
    except Exception as e:
        logger.warning(f"Could not connect to Vault service at {vault_url}: {e}")

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
        if count > 0 and count % 2 == 0:
            logging.warning(f"Misclassification count hit {count}. Triggering Jenkins retraining pipeline!")
            try:
                # Use dynamic credentials from Vault; fallback to env vars if Vault failed
                username = JENKINS_AUTH[0] if JENKINS_AUTH else os.environ.get('JENKINS_USER', 'admin')
                password = JENKINS_AUTH[1] if JENKINS_AUTH else os.environ.get('JENKINS_PASS')
                
                if not password:
                    logger.error("No Jenkins password found in Vault or Environment. Retraining trigger failed.")
                    return

                auth = (username, password)
                session = requests.Session()
                session.auth = auth
                
                crumb_url = "http://localhost:9090/crumbIssuer/api/json"
                crumb_resp = session.get(crumb_url, timeout=5)
                crumb_data = crumb_resp.json()
                headers = {crumb_data['crumbRequestField']: crumb_data['crumb']}
                
                webhook_url = "http://localhost:9090/job/fake_news_Retraining/build"
                resp = session.post(webhook_url, headers=headers, timeout=5)
                if resp.status_code in [200, 201]:
                    logger.info("Successfully triggered Jenkins webhook for retraining.")
                else:
                    logger.error(f"Failed to trigger Jenkins, status code: {resp.status_code}")
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