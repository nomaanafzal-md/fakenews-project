import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def main():
    print("Loading datasets...")
    fake_path = "Fake.csv"
    true_path = "True.csv"
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Make sure Fake.csv and True.csv are in the current working directory.")

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)

    # Assign labels: 0 for Fake, 1 for Real
    fake["label"] = 0
    real["label"] = 1

    print(f"Fake news samples: {len(fake):,}")
    print(f"Real news samples: {len(real):,}")

    # Combine title and text to enrich feature representations
    fake["content"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
    real["content"] = real["title"].fillna("") + " " + real["text"].fillna("")

    df = pd.concat([fake, real], ignore_index=True)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total combined samples: {len(df):,}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    print(f"Training set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # Feature extraction with TF-IDF
    print("\nFitting TF-IDF Vectorizer...")
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    print(f"TF-IDF matrix shape: {X_train_vec.shape}")

    # Model training with Logistic Regression
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluation
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Fake (0)", "Real (1)"], digits=4)

    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(f"TN (True Fake): {cm[0][0]:<6} | FP (False Real): {cm[0][1]}")
    print(f"FN (False Fake): {cm[1][0]:<6} | TP (True Real):  {cm[1][1]}")
    print("\nClassification Report:\n")
    print(cr)
    print("=" * 60)

    # Save artifacts
    print("Saving model and vectorizer...")
    joblib.dump(model, "model.pkl")
    joblib.dump(vec, "vec.pkl")
    print("Saved 'model.pkl' and 'vec.pkl' successfully.")

    # Export model to JSON format for lightweight runtime inference
    import json
    data = {
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0])
    }
    with open("model.json", "w") as f:
        json.dump(data, f)
    print("model.json saved successfully!")

    # Write evaluation metrics to a markdown report
    metrics_report = f"""# ML Model Training & Evaluation Report

## Dataset Summary
- **Fake news samples**: {len(fake):,}
- **Real news samples**: {len(real):,}
- **Total samples**: {len(df):,}
- **Split**: 80% Train ({len(X_train):,}), 20% Test ({len(X_test):,}) (Stratified)

## Performance Metrics
- **Accuracy**: {acc * 100:.2f}%
- **Precision**: {prec * 100:.2f}%
- **Recall**: {rec * 100:.2f}%
- **F1-Score**: {f1 * 100:.2f}%

## Confusion Matrix
| Metric / Actual | Predicted Fake (0) | Predicted Real (1) |
| :--- | :--- | :--- |
| **Actual Fake (0)** | {cm[0][0]:,} (True Negative) | {cm[0][1]:,} (False Positive) |
| **Actual Real (1)** | {cm[1][0]:,} (False Negative) | {cm[1][1]:,} (True Positive) |

## Classification Report
```
{cr}
```

## Saved Artifacts
- `model.pkl`: Serialized scikit-learn Logistic Regression model
- `vec.pkl`: Serialized TF-IDF vectorizer (max 50,000 features, unigrams + bigrams)
"""
    with open("training_report.md", "w", encoding="utf-8") as f:
        f.write(metrics_report)
    print("Saved 'training_report.md'.")

if __name__ == "__main__":
    main()
