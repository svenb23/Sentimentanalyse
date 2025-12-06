import json
from typing import Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


class ModelTrainer:

    MODEL_REGISTRY = {
        'logistic_regression': LogisticRegression,
        'naive_bayes': MultinomialNB,
        'svm': LinearSVC,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier
    }

    DEFAULT_PARAMS = {
        'logistic_regression': {'max_iter': 1000, 'random_state': 42},
        'naive_bayes': {},
        'svm': {'max_iter': 2000, 'random_state': 42},
        'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'random_state': 42}
    }

    def __init__(self, config: dict):
        self.config = config.get('model', {})
        self.model_type = self.config.get('type', 'logistic_regression')
        self.model_params = self.config.get('params', {})
        self.model = self._create_model()
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}

    def _create_model(self):
        model_class = self.MODEL_REGISTRY.get(self.model_type)

        if model_class is None:
            raise ValueError(f"Unknown model type: {self.model_type}")

        params = self.DEFAULT_PARAMS.get(self.model_type, {}).copy()
        params.update(self.model_params)

        return model_class(**params)

    def train(self, X_train: spmatrix, y_train) -> 'ModelTrainer':
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.predict(X)

    def evaluate(self, X: spmatrix, y_true, dataset_name: str = 'test') -> Dict[str, Any]:
        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }

        self.metrics[dataset_name] = metrics
        return metrics

    def evaluate_by_category(self, X: spmatrix, y_true, categories: pd.Series) -> Dict[str, Dict]:
        y_pred = self.predict(X)
        category_metrics = {}

        for category in categories.unique():
            mask = categories == category
            if mask.sum() == 0:
                continue

            y_true_cat = y_true[mask] if hasattr(y_true, '__getitem__') else y_true.iloc[mask]
            y_pred_cat = y_pred[mask]

            category_metrics[category] = {
                'accuracy': accuracy_score(y_true_cat, y_pred_cat),
                'f1_macro': f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0),
                'count': int(mask.sum())
            }

        self.metrics['by_category'] = category_metrics
        return category_metrics

    def get_classification_report(self, X: spmatrix, y_true) -> str:
        y_pred = self.predict(X)
        return classification_report(y_true, y_pred)

    def plot_confusion_matrix(self, X: spmatrix, y_true, save_path: Path = None, labels: list = None) -> plt.Figure:
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        if labels is None:
            labels = sorted(np.unique(y_true))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {self.model_type}')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def get_config_summary(self) -> dict:
        return {
            'type': self.model_type,
            'params': {**self.DEFAULT_PARAMS.get(self.model_type, {}), **self.model_params}
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def save_metrics(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
