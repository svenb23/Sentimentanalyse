import argparse
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def load_experiment_results(experiment_dir: Path) -> dict:
    summary_path = experiment_dir / "summary.json"
    metrics_path = experiment_dir / "metrics.json"

    if not summary_path.exists():
        return None

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        if 'test' in metrics:
            summary['test_precision_macro'] = metrics['test'].get('precision_macro')
            summary['test_recall_macro'] = metrics['test'].get('recall_macro')
            summary['test_f1_weighted'] = metrics['test'].get('f1_weighted')

        if 'by_category' in metrics:
            for cat, cat_metrics in metrics['by_category'].items():
                summary[f'acc_{cat}'] = cat_metrics.get('accuracy')

        if 'config' in metrics:
            config = metrics['config']
            if 'preprocessing' in config:
                prep = config['preprocessing']
                summary['stemming'] = prep.get('stemming', False)
                summary['lemmatization'] = prep.get('lemmatization', False)

            if 'features' in config:
                feat = config['features']
                summary['vectorizer_type'] = feat.get('type', 'tfidf')
                summary['ngram_range'] = str(feat.get('ngram_range', [1, 2]))

    return summary


def compare_all_experiments() -> pd.DataFrame:
    if not EXPERIMENTS_DIR.exists():
        return pd.DataFrame()

    results = []
    for experiment_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not experiment_dir.is_dir():
            continue
        result = load_experiment_results(experiment_dir)
        if result:
            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    priority_cols = ['experiment_id', 'model', 'test_accuracy', 'test_f1_macro', 'val_accuracy', 'features']
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]
    df = df.sort_values('test_accuracy', ascending=False)

    return df


def print_comparison_table(df: pd.DataFrame):
    if df.empty:
        return

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80 + "\n")

    main_cols = ['experiment_id', 'model', 'test_accuracy', 'test_f1_macro', 'features']
    available_cols = [c for c in main_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest: {best['experiment_id']}")
    print(f"  Accuracy: {best['test_accuracy']:.4f}")
    print(f"  F1: {best['test_f1_macro']:.4f}")

    cat_cols = [c for c in df.columns if c.startswith('acc_')]
    if cat_cols:
        print("\nBy category:")
        print(df[['experiment_id', 'model'] + cat_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--latex', '-l', type=str, default=None)
    args = parser.parse_args()

    df = compare_all_experiments()

    if df.empty:
        print("No experiments found.")
        return

    print_comparison_table(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nExported to: {args.output}")

    if args.latex:
        main_cols = ['model', 'test_accuracy', 'test_f1_macro', 'features']
        latex_df = df[[c for c in main_cols if c in df.columns]].copy()
        latex_df.to_latex(args.latex, index=False)
        print(f"LaTeX exported to: {args.latex}")


if __name__ == "__main__":
    main()
