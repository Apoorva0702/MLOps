from fastapi.testclient import TestClient
import sys
import os

# Add the root directory to sys.path so we can import api.main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_read_main():
    """Test the home page returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_endpoint():
    """Test the prediction endpoint with a sample string."""
    test_text = "Breaking news: New discovery on Mars suggests liquid water once flowed freely."
    response = client.post(
        "/predict",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["FAKE", "REAL"]
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1

def test_feedback_endpoint():
    """Test the feedback endpoint by first making a prediction."""
    # 1. Make a prediction to get a valid ID
    pred_response = client.post(
        "/predict",
        json={"text": "Feedback test article."}
    )
    assert pred_response.status_code == 200
    row_id = pred_response.json()["id"]

    # 2. Submit feedback for that ID
    feedback_response = client.post(
        "/feedback",
        json={"id": row_id, "correct_label": "REAL"}
    )
    assert feedback_response.status_code == 200
    data = feedback_response.json()
    assert data["id"] == row_id
    assert data["correct_label"] == "REAL"
    assert "is_misclassified" in data

def test_health_endpoint():
    """Test the health check endpoint returns 200 OK and correct JSON structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["database"] == "connected"

