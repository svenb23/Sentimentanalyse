import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def main():
    experiment_dir = EXPERIMENTS_DIR / f"bert_pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv").sample(n=2000, random_state=42)
    texts = (test_df['title'].fillna('') + ' ' + test_df['text'].fillna('')).tolist()
    y_true = test_df['rating'].values

    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    y_pred = []
    for text in tqdm(texts, desc="Predicting"):
        result = classifier(text[:512], truncation=True)[0]
        stars = int(result['label'].split()[0])
        y_pred.append(stars)

    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Macro: {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred)}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig(experiment_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')

    with open(experiment_dir / "summary.json", 'w') as f:
        json.dump({'accuracy': accuracy, 'f1_macro': f1}, f, indent=2)

    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
