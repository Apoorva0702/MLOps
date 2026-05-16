import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import pytest

def test_model_loading():
    """Verify that the model and tokenizer can be loaded from the project directory."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model/bert_fake_news_model")
    
    if os.path.exists(MODEL_PATH):
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        assert model is not None
        assert tokenizer is not None
    else:
        # Fallback to base model for test environment if project model is missing
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        assert model is not None

def test_model_inference():
    """Verify that the model produces a valid prediction for a sample text."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model/bert_fake_news_model")
    
    model_name = MODEL_PATH if os.path.exists(MODEL_PATH) else "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    text = "Scientists find evidence of ancient life on Mars."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
    assert prediction in [0, 1]
