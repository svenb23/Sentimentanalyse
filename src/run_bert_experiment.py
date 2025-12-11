import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def load_data():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    return train_df, val_df, test_df


def prepare_dataset(df, tokenizer):
    texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()
    labels = (df['rating'] - 1).tolist()
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    return dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=256), batched=True)


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return {
        'accuracy': accuracy_score(eval_pred.label_ids, preds),
        'f1_macro': f1_score(eval_pred.label_ids, preds, average='macro')
    }


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    experiment_dir = EXPERIMENTS_DIR / f"bert_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_data()
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)
    test_dataset = prepare_dataset(test_df, tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(experiment_dir),
            num_train_epochs=5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            fp16=torch.cuda.is_available(),
            report_to="none",
            logging_steps=100
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate(test_dataset)
    y_pred = np.argmax(trainer.predict(test_dataset).predictions, axis=-1) + 1
    y_true = test_df['rating'].values

    print(f"\nTest Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test F1 Macro: {results['eval_f1_macro']:.4f}")
    print(f"\n{classification_report(y_true, y_pred)}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig(experiment_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')

    with open(experiment_dir / "summary.json", 'w') as f:
        json.dump({'accuracy': results['eval_accuracy'], 'f1_macro': results['eval_f1_macro']}, f, indent=2)

    trainer.save_model(experiment_dir / "model")
    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
