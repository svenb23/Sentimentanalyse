import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TextPreprocessor, FeatureVectorizer, ModelTrainer

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

SENTIMENT_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
SENTIMENT_LABELS = ['negativ', 'neutral', 'positiv']


def rating_to_sentiment(rating: int) -> int:
    return SENTIMENT_MAP.get(rating, 1)


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_raw_data() -> tuple:
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df

    raise FileNotFoundError("No data found. Run data_preprocessing.py first.")


def print_sentiment_distribution(y, name: str):
    counts = y.value_counts().sort_index()
    total = len(y)
    print(f"      {name} Sentiment-Verteilung:")
    for sentiment_id, count in counts.items():
        label = SENTIMENT_LABELS[sentiment_id]
        pct = count / total * 100
        print(f"        {label:8}: {count:6} ({pct:5.1f}%)")


def run_sentiment_experiment(config_path: Path, experiment_name: str = None):
    config = load_config(config_path)

    if experiment_name is None:
        experiment_name = "sentiment_" + config.get('experiment_name', config_path.stem)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"SENTIMENT-POLARITÄT Experiment: {experiment_id}")
    print(f"Mapping: 1-2→negativ, 3→neutral, 4-5→positiv")
    print(f"{'='*60}\n")

    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config['sentiment_analysis'] = True
    config['sentiment_mapping'] = SENTIMENT_MAP
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    print("[1/6] Loading data...")
    train_df, val_df, test_df = load_raw_data()
    print(f"      Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n[2/6] Transforming ratings to sentiment...")
    train_df['sentiment'] = train_df['rating'].map(rating_to_sentiment)
    val_df['sentiment'] = val_df['rating'].map(rating_to_sentiment)
    test_df['sentiment'] = test_df['rating'].map(rating_to_sentiment)

    print_sentiment_distribution(train_df['sentiment'], "Train")
    print_sentiment_distribution(test_df['sentiment'], "Test")

    print("\n[3/6] Preprocessing...")
    preprocessor = TextPreprocessor(config)
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)

    print("\n[4/6] Feature extraction...")
    vectorizer = FeatureVectorizer(config)
    X_train = vectorizer.fit_transform(train_df['processed_text'].fillna(''))
    X_val = vectorizer.transform(val_df['processed_text'].fillna(''))
    X_test = vectorizer.transform(test_df['processed_text'].fillna(''))

    y_train = train_df['sentiment']
    y_val = val_df['sentiment']
    y_test = test_df['sentiment']

    print(f"      Features: {vectorizer.get_actual_feature_count()}")

    print("\n[5/6] Training...")
    trainer = ModelTrainer(config)
    trainer.train(X_train, y_train)
    print(f"      Model: {trainer.model_type}")

    print("\n[6/6] Evaluation...")
    val_metrics = trainer.evaluate(X_val, y_val, 'validation')
    test_metrics = trainer.evaluate(X_test, y_test, 'test')

    print(f"\n      Val Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"      Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"      Test F1 Macro: {test_metrics['f1_macro']:.4f}")

    y_pred = trainer.model.predict(X_test)
    print(f"\n{classification_report(y_test, y_pred, target_names=SENTIMENT_LABELS)}")

    trainer.plot_confusion_matrix(
        X_test, y_test,
        save_path=experiment_dir / "confusion_matrix.png",
        labels=[0, 1, 2]
    )

    if 'category' in test_df.columns:
        print("\n      By category:")
        for cat in test_df['category'].unique():
            mask = test_df['category'] == cat
            X_cat = X_test[mask.values]
            y_cat = y_test[mask]
            cat_metrics = trainer.evaluate(X_cat, y_cat, f'test_{cat}')
            print(f"        {cat}: Acc={cat_metrics['accuracy']:.4f}, F1={cat_metrics['f1_macro']:.4f}")

    all_metrics = trainer.get_all_metrics()
    all_metrics['experiment_id'] = experiment_id
    all_metrics['sentiment_analysis'] = True
    all_metrics['sentiment_labels'] = SENTIMENT_LABELS

    trainer.save_metrics(experiment_dir / "metrics.json")
    joblib.dump(trainer.model, experiment_dir / "model.joblib")
    joblib.dump(vectorizer.vectorizer, experiment_dir / "vectorizer.joblib")

    summary = {
        'experiment_id': experiment_id,
        'type': 'sentiment_polarity',
        'classes': SENTIMENT_LABELS,
        'model': trainer.model_type,
        'val_accuracy': val_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'features': vectorizer.get_actual_feature_count()
    }

    with open(experiment_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {experiment_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--name', '-n', type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    run_sentiment_experiment(config_path, args.name)


if __name__ == "__main__":
    main()
