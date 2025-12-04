import re
import string
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def preprocess_dataframe(df):
    df['combined_text'] = df.apply(
        lambda row: f"{row['title'] if isinstance(row['title'], str) else ''} {row['text'] if isinstance(row['text'], str) else ''}".strip(),
        axis=1
    )
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    return df


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df)

    columns_to_save = ['processed_text', 'rating', 'category']
    train_df[columns_to_save].to_csv(PROCESSED_DATA_DIR / "train_preprocessed.csv", index=False)
    val_df[columns_to_save].to_csv(PROCESSED_DATA_DIR / "val_preprocessed.csv", index=False)
    test_df[columns_to_save].to_csv(PROCESSED_DATA_DIR / "test_preprocessed.csv", index=False)

    print("Fertig! Dateien gespeichert.")
