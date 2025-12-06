import json
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

CATEGORIES = ["Automotive", "Pet_Supplies", "Video_Games"]


def load_category_data(category, sample_size=10000, random_seed=42):
    filepath = RAW_DATA_DIR / f"{category}.jsonl"
    print(f"\nLade {category}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"  Gesamt: {total_lines:,}")

    random.seed(random_seed)
    if total_lines <= sample_size:
        selected_indices = set(range(total_lines))
    else:
        selected_indices = set(random.sample(range(total_lines), sample_size))

    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in selected_indices:
                try:
                    data = json.loads(line)
                    samples.append({
                        'title': data.get('title', ''),
                        'text': data.get('text', ''),
                        'rating': int(data['rating']),
                        'category': category
                    })
                except:
                    continue

    print(f"  Geladen: {len(samples):,}")
    return samples


def load_all_data(sample_size_per_category=10000, random_seed=42):
    all_data = []
    for category in CATEGORIES:
        samples = load_category_data(category, sample_size_per_category, random_seed)
        all_data.extend(samples)

    df = pd.DataFrame(all_data)
    print(f"\nGesamt: {len(df):,} Rezensionen")
    return df


def split_data(df, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42):
    print(f"\nSplitte Daten: {train_size*100:.0f}% Train, {val_size*100:.0f}% Val, {test_size*100:.0f}% Test")

    # Erst Train vom Rest trennen
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_seed,
        stratify=df['rating']
    )

    # Dann Val und Test trennen
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_seed,
        stratify=temp_df['rating']
    )

    print(f"  Train: {len(train_df):,}")
    print(f"  Validation: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")

    return train_df, val_df, test_df


def print_distribution(df, name):
    print(f"\n{name} - Verteilung der Bewertungen:")
    for rating in sorted(df['rating'].unique()):
        count = len(df[df['rating'] == rating])
        print(f"  {rating} Sterne: {count:,} ({count/len(df)*100:.1f}%)")


def save_data(train_df, val_df, test_df):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"\nDaten gespeichert in {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    df = load_all_data(sample_size_per_category=50000)

    train_df, val_df, test_df = split_data(df)

    print_distribution(train_df, "Training")
    print_distribution(val_df, "Validation")
    print_distribution(test_df, "Test")

    save_data(train_df, val_df, test_df)

    print("\nBeispiel aus Training:")
    print(train_df[['title', 'text', 'rating']].head(2))
