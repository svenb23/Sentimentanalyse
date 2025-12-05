import joblib
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def load_features():
    X_train = joblib.load(PROCESSED_DATA_DIR / "X_train.joblib")
    X_val = joblib.load(PROCESSED_DATA_DIR / "X_val.joblib")
    X_test = joblib.load(PROCESSED_DATA_DIR / "X_test.joblib")

    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")['rating']
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv")['rating']
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")['rating']

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_features()

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(max_iter=2000, random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_val, model.predict(X_val))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    y_test_pred = best_model.predict(X_test)

    print(f"Bestes Modell: {best_name}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"\n{classification_report(y_test, y_test_pred)}")
    print(confusion_matrix(y_test, y_test_pred))

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
