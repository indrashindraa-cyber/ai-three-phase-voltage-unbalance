"""Train and evaluate voltage-unbalance severity classifiers."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "voltage_unbalance_dataset.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
FEATURES = ["Vr_V", "Vy_V", "Vb_V", "Average_V", "Unbalance_Percent"]
TARGET = "Severity"
LABEL_ORDER = ["Balanced", "Mildly Unbalanced", "Severely Unbalanced"]

def calculate_features(frame):
    """Recalculate average and chosen illustrative voltage-unbalance metric."""
    result = frame.copy()
    v = result[["Vr_V", "Vy_V", "Vb_V"]].astype(float)
    result["Average_V"] = v.mean(axis=1)
    result["Unbalance_Percent"] = (
        v.sub(result["Average_V"], axis=0).abs().max(axis=1)
        / result["Average_V"].abs().replace(0, np.nan) * 100
    )
    return result

def main():
    df = pd.read_csv(DATA)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Vr_V", "Vy_V", "Vb_V", TARGET])
    df = calculate_features(df)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12,
                                                 class_weight="balanced", random_state=42,
                                                 n_jobs=-1)
    }
    metrics = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, estimator in models.items():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator)
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        result = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(y_test, pred, labels=LABEL_ORDER, zero_division=0, output_dict=True)
        }
        metrics[name] = result
        if result["f1_weighted"] > best_f1:
            best_name, best_model, best_f1 = name, model, result["f1_weighted"]
        cm = confusion_matrix(y_test, pred, labels=LABEL_ORDER)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(OUT / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=160)
        plt.close()

    joblib.dump(best_model, OUT / "best_model.joblib")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "models": metrics,
                   "thresholds_percent": {"balanced_max": 1.0, "mild_max": 2.0},
                   "data_note": "Synthetic educational data; not validated for protection decisions."}, f, indent=2)

    # Separate unseen-data demonstration
    unseen_path = ROOT / "data" / "unseen_test_data.csv"
    unseen = calculate_features(pd.read_csv(unseen_path))
    unseen_pred = best_model.predict(unseen[FEATURES])
    unseen["Predicted_Severity"] = unseen_pred
    unseen.to_csv(OUT / "unseen_predictions.csv", index=False)

    print(f"Best model by weighted F1: {best_name}")
    for name, result in metrics.items():
        print(f"{name}: accuracy={result['accuracy']:.4f}, "
              f"precision={result['precision_weighted']:.4f}, "
              f"recall={result['recall_weighted']:.4f}, f1={result['f1_weighted']:.4f}")
    print(f"Unseen demo rows: {len(unseen)}")
    print(f"Unseen demo accuracy: {accuracy_score(unseen['Severity'], unseen_pred):.4f}")
    print(f"Saved outputs in: {OUT}")

if __name__ == "__main__":
    main()
