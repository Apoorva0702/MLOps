from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
import os
import sqlite3
import requests
import logging
import hvac
from logstash_async.handler import AsynchronousLogstashHandler
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

# Load environment variables from .env file
load_dotenv()

# Logstash Configuration
LOGSTASH_HOST = os.environ.get('LOGSTASH_HOST', 'logstash')
LOGSTASH_PORT = int(os.environ.get('LOGSTASH_PORT', 5000))
ENABLE_LOGSTASH = os.environ.get('ENABLE_LOGSTASH', 'false').lower() == 'true'

# Root Logger Config
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Keep console handler
logger.addHandler(logging.StreamHandler())

if ENABLE_LOGSTASH:
    try:
        logstash_handler = AsynchronousLogstashHandler(
            host=LOGSTASH_HOST, 
            port=LOGSTASH_PORT, 
            database_path='logstash.db'
        )
        logger.addHandler(logstash_handler)
        print(f"Logstash logging enabled. Target: {LOGSTASH_HOST}:{LOGSTASH_PORT}")
    except Exception as e:
        print(f"Failed to initialize Logstash handler: {e}")
else:
    print("Logstash logging is disabled (ENABLE_LOGSTASH is not set to 'true').")
    # Clean up leftover logstash.db if it exists to prevent disk accumulation
    if os.path.exists('logstash.db'):
        try:
            os.remove('logstash.db')
            print("Cleaned up unused logstash.db file.")
        except Exception as e:
            print(f"Could not clean up logstash.db: {e}")

app = FastAPI()

# Instrument FastAPI for Prometheus metrics
Instrumentator().instrument(app).expose(app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "../frontend")), name="static")

MODEL_PATH = os.path.join(BASE_DIR, "../model/bert_fake_news_model")
DB_PATH = os.path.join(BASE_DIR, "../db/fake_news.db")

def init_db():
    """Initialize the database and create tables if they don't exist."""
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created database directory: {db_dir}")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_text TEXT,
            predicted_label TEXT,
            confidence REAL,
            correct_label TEXT,
            feedback_given INTEGER DEFAULT 0,
            is_misclassified INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# Run database initialization
init_db()

model_name_or_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------- Vault Configuration ----------

VAULT_ADDR = os.environ.get('VAULT_ADDR', 'http://vault:8200')
VAULT_TOKEN = os.environ.get('VAULT_TOKEN', 'dev-only-token')

def fetch_vault_secrets():
    """Fetch Jenkins credentials from HashiCorp Vault."""
    try:
        print(f"Attempting to fetch secrets from HashiCorp Vault at {VAULT_ADDR}...")
        # In Docker, 'vault' is the hostname. In local dev, it might be 'localhost'.
        # We use the VAULT_ADDR env var to handle both.
        client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)
        
        # Verify if token is valid
        if not client.is_authenticated():
            print("Vault authentication failed!")
            return None, None

        # Fetch secrets from kv-v2 engine (path: secret/data/jenkins)
        read_response = client.secrets.kv.v2.read_secret_version(path='jenkins')
        secrets = read_response['data']['data']
        
        print("Successfully retrieved secrets from Vault.")
        return secrets.get('username'), secrets.get('password')
    except Exception as e:
        print(f"Could not connect to Vault service at {VAULT_ADDR}: {e}")
        return None, None

# Initial fetch of credentials
VAULT_JENKINS_USER, VAULT_JENKINS_PASS = fetch_vault_secrets()


# ---------- Request Models ----------

class InputText(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    id: int
    correct_label: str



# ---------- Health Check ----------

@app.get("/health")
def health_check():
    """Verify application health (model loaded & database responsive)."""
    # 1. Check if model and tokenizer are initialized
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "BERT model or tokenizer not loaded"}
        )
        
    # 2. Check if SQLite database is responsive
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": f"Database unresponsive: {str(e)}"}
        )

    return {"status": "healthy", "model_loaded": True, "database": "connected"}


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

    # Mirror the DB record to ELK as a structured log event
    logger.info(
        "prediction_stored",
        extra={
            "event_type": "prediction",
            "record_id": row_id,
            "article_text": data.text,
            "predicted_label": label,
            "confidence": round(confidence, 4),
        }
    )

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

    # Mirror the feedback update to ELK as a structured log event
    logger.info(
        "feedback_stored",
        extra={
            "event_type": "feedback",
            "record_id": data.id,
            "predicted_label": predicted,
            "correct_label": data.correct_label.upper(),
            "is_misclassified": bool(misclassified),
        }
    )

    if misclassified == 1:
        cur.execute("SELECT COUNT(*) FROM news_predictions WHERE is_misclassified=1")
        count = cur.fetchone()[0]
        if count > 0 and count % 2 == 0:
            logging.warning(f"Misclassification count hit {count}. Triggering Jenkins retraining pipeline!")
            try:
                # Use credentials directly from Environment Variables (provided by K8s Secrets)
                # Use credentials from Vault if available, fallback to environment variables
                username = VAULT_JENKINS_USER or os.environ.get('JENKINS_USER', 'admin')
                password = VAULT_JENKINS_PASS or os.environ.get('JENKINS_PASS')
                
                if not password:
                    logger.error("No Jenkins password found in Environment Variables. Pipeline trigger will fail.")
                    return {"error": "Missing Jenkins credentials"}

                jenkins_url = os.environ.get('JENKINS_URL', 'http://localhost:9090')
                logger.info(f"Triggering Jenkins for user: {username} at {jenkins_url}")
                auth = (username, password)
                session = requests.Session()
                session.auth = auth
                
                # Fetch Crumb for CSRF protection
                crumb_url = f"{jenkins_url.rstrip('/')}/crumbIssuer/api/json"
                crumb_resp = session.get(crumb_url, timeout=5)
                
                headers = {}
                if crumb_resp.status_code == 200:
                    crumb_data = crumb_resp.json()
                    headers = {crumb_data['crumbRequestField']: crumb_data['crumb']}
                else:
                    logger.warning(f"Could not fetch Jenkins crumb (Status: {crumb_resp.status_code}). Attempting trigger without crumb...")
                
                webhook_url = f"{jenkins_url.rstrip('/')}/job/fake_news_Retraining/build"
                resp = session.post(webhook_url, headers=headers, timeout=5)
                
                if resp.status_code in [200, 201, 202]:
                    logger.info(f"Successfully triggered Jenkins retraining pipeline (Status: {resp.status_code}).")
                else:
                    logger.error(f"Failed to trigger Jenkins. Status: {resp.status_code}, Response: {resp.text}")
            except Exception as e:
                logger.error(f"Critical error during Jenkins trigger: {e}")

    conn.commit()
    conn.close()

    return {
        "id": data.id,
        "predicted_label": predicted,
        "correct_label": data.correct_label.upper(),
        "is_misclassified": misclassified
    }