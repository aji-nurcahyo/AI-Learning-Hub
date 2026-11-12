"""Interpretable Baseline Models (Dummy Majority & Logistic Regression)."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from preprocessing import prepare_splits


def evaluate_classifier(model, X, y) -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
    }


def main():
    data = prepare_splits()
    dummy = DummyClassifier(strategy="most_frequent").fit(data.X_train, data.y_train)
    logreg = LogisticRegression(random_state=42).fit(data.X_train, data.y_train)

    print("=" * 78)
    print("CHAPTER 40 CAPSTONE: BASELINE MODELS (VALIDATION SET)")
    print("=" * 78)
    for name, mdl in [("Dummy Majority", dummy), ("Logistic Regression", logreg)]:
        m = evaluate_classifier(mdl, data.X_val, data.y_val)
        print(
            f"- {name:20s} | Acc={m['accuracy']*100:6.2f}% | Prec={m['precision']:.3f} | "
            f"Rec={m['recall']:.3f} | F1={m['f1']:.3f} | AUC={m['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()

