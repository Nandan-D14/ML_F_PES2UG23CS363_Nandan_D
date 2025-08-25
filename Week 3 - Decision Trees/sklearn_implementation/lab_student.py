# PES2UG23CS363
import argparse, csv, sys
from typing import List, Dict, Any, Tuple
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("Empty CSV")
    header = rows[0]
    data = rows[1:]
    return header, data


def split_X_y(header: List[str], data: List[List[str]], label_col: str):
    if label_col not in header:
        raise ValueError(f"Label column '{label_col}' not in CSV header: {header}")
    y_idx = header.index(label_col)

    X_raw, y_raw = [], []
    for r in data:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        y_raw.append(r[y_idx])
        X_raw.append([v for i, v in enumerate(r) if i != y_idx])
    feat_names = [c for i, c in enumerate(header) if i != y_idx]
    return feat_names, X_raw, y_raw


def encode_categorical(X_raw: List[List[str]], y_raw: List[str]):
    enc_X = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc_X.fit_transform(X_raw).astype(np.int64)

    enc_y = LabelEncoder()
    y = enc_y.fit_transform(y_raw).astype(np.int64)
    return X, y, enc_X, enc_y


def main():
    ap = argparse.ArgumentParser(description="Sklearn Decision Tree on categorical CSV")
    ap.add_argument("--data", required=True, help="Path to CSV (e.g., mushroom.csv)")
    ap.add_argument("--label", default="class", help="Label column name (default: class)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--criterion", choices=["gini", "entropy", "log_loss"], default="entropy")
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_split", type=int, default=2)
    ap.add_argument("--grid", action="store_true", help="Run small GridSearchCV for depth/criterion")
    args = ap.parse_args()

    header, data = load_csv(args.data)
    feat_names, X_raw, y_raw = split_X_y(header, data, args.label)
    X, y, enc_X, enc_y = encode_categorical(X_raw, y_raw)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    if args.grid:
        param_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
        gs = GridSearchCV(DecisionTreeClassifier(random_state=args.seed),
                          param_grid=param_grid,
                          n_jobs=-1, cv=5)
        gs.fit(Xtr, ytr)
        clf = gs.best_estimator_
        print(f"[GridSearch] best params: {gs.best_params_}")
    else:
        clf = DecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.seed,
        )
        clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print("\n=== sklearn Decision Tree Results ===")
    print(f"Data: {args.data}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(yte, pred))
    print("\nClassification report:\n", classification_report(yte, pred, target_names=enc_y.classes_))

    try:
        txt = export_text(clf, feature_names=list(feat_names))
        print("\n--- Learned Tree (text) ---")
        print(txt[:8000])
    except Exception as e:
        print(f"(Tree text not shown): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
