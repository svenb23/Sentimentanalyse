import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TextPreprocessor, FeatureVectorizer, ModelTrainer

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df

    raise FileNotFoundError("No data found. Run data_preprocessing.py first.")


def run_experiment(config_path: Path, experiment_name: str = None):
    config = load_config(config_path)

    if experiment_name is None:
        experiment_name = config.get('experiment_name', config_path.stem)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_id}")
    print(f"{'='*60}\n")

    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, experiment_dir / "config.yaml")

    print("[1/5] Loading data...")
    train_df, val_df, test_df = load_raw_data(config)
    print(f"      Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("[2/5] Preprocessing...")
    preprocessor = TextPreprocessor(config)
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)

    print("[3/5] Feature extraction...")
    vectorizer = FeatureVectorizer(config)
    X_train = vectorizer.fit_transform(train_df['processed_text'].fillna(''))
    X_val = vectorizer.transform(val_df['processed_text'].fillna(''))
    X_test = vectorizer.transform(test_df['processed_text'].fillna(''))

    y_train = train_df['rating']
    y_val = val_df['rating']
    y_test = test_df['rating']

    print(f"      Features: {vectorizer.get_actual_feature_count()}")

    print("[4/5] Training...")
    trainer = ModelTrainer(config)
    trainer.train(X_train, y_train)
    print(f"      Model: {trainer.model_type}")

    print("[5/5] Evaluation...")
    val_metrics = trainer.evaluate(X_val, y_val, 'validation')
    test_metrics = trainer.evaluate(X_test, y_test, 'test')

    print(f"\n      Val Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"      Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"      Test F1:       {test_metrics['f1_macro']:.4f}")

    if 'category' in test_df.columns:
        category_metrics = trainer.evaluate_by_category(X_test, y_test, test_df['category'])
        print("\n      By category:")
        for cat, m in category_metrics.items():
            print(f"        {cat}: {m['accuracy']:.4f}")

    print(f"\n{trainer.get_classification_report(X_test, y_test)}")

    trainer.plot_confusion_matrix(X_test, y_test, save_path=experiment_dir / "confusion_matrix.png", labels=[1, 2, 3, 4, 5])

    all_metrics = trainer.get_all_metrics()
    all_metrics['experiment_id'] = experiment_id
    all_metrics['config'] = {
        'preprocessing': preprocessor.get_config_summary(),
        'features': vectorizer.get_config_summary(),
        'model': trainer.get_config_summary()
    }

    trainer.save_metrics(experiment_dir / "metrics.json")
    joblib.dump(trainer.model, experiment_dir / "model.joblib")
    joblib.dump(vectorizer.vectorizer, experiment_dir / "vectorizer.joblib")

    summary = {
        'experiment_id': experiment_id,
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

    run_experiment(config_path, args.name)


if __name__ == "__main__":
    main()
