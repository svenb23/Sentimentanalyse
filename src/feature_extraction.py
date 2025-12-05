import joblib
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def load_data():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_preprocessed.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val_preprocessed.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_preprocessed.csv")
    return train_df, val_df, test_df


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_data()

    train_df['processed_text'] = train_df['processed_text'].fillna('')
    val_df['processed_text'] = val_df['processed_text'].fillna('')
    test_df['processed_text'] = test_df['processed_text'].fillna('')

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train_df['processed_text'])
    X_val = vectorizer.transform(val_df['processed_text'])
    X_test = vectorizer.transform(test_df['processed_text'])

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(X_train, PROCESSED_DATA_DIR / "X_train.joblib")
    joblib.dump(X_val, PROCESSED_DATA_DIR / "X_val.joblib")
    joblib.dump(X_test, PROCESSED_DATA_DIR / "X_test.joblib")

    train_df['rating'].to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    val_df['rating'].to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    test_df['rating'].to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
