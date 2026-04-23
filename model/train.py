import sqlite3
import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../db/fake_news.db")
MODEL_PATH = os.path.join(BASE_DIR, "../model/bert_fake_news_model")
DATA_PATH = os.path.join(BASE_DIR, "../data/original_dataset.csv")

def fetch_misclassified_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT article_text, correct_label FROM news_predictions WHERE is_misclassified=1")
    rows = cur.fetchall()
    conn.close()
    
    texts = []
    labels = []
    label_map = {"FAKE": 0, "REAL": 1}
    for row in rows:
        texts.append(row[0])
        labels.append(label_map.get(row[1], 0))
        
    return texts, labels

def train(original_only=False):
    texts = []
    labels = []
    
    if not original_only:
        texts, labels = fetch_misclassified_data()
        if texts:
            print(f"Loaded {len(texts)} misclassified examples from DB.")
    else:
        print("Running in original-data-only mode. Skipping DB fetch.")
    
    # Also load the original dataset to prevent catastrophic forgetting
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if 'text' in df.columns and 'label' in df.columns:
                orig_texts = df['text'].tolist()
                label_map = {"FAKE": 0, "REAL": 1}
                if df['label'].dtype == object:
                    orig_labels = [label_map.get(str(l).upper(), 0) for l in df['label'].tolist()]
                else:
                    orig_labels = df['label'].tolist()
                
                texts.extend(orig_texts)
                labels.extend(orig_labels)
                print(f"Successfully appended {len(df)} original training records.")
        except Exception as e:
            print(f"Error loading original dataset: {e}")

    if not texts:
        print("No training data found (neither misclassified nor original).")
        return

    print(f"Retraining on {len(texts)} new examples...")

    model_name_or_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(model_name_or_path)
    
    # Tokenize data
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save the fine-tuned model
    print("Saving updated model...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("Retraining complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-only", action="store_true", help="Train on original data only, skip database fetch.")
    args = parser.parse_args()
    
    train(original_only=args.original_only)


#test comment 123
#dhgh
#dfkbbkdfpdfhdp