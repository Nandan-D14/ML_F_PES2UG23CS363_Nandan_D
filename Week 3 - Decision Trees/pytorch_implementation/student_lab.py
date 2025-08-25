# PES2UG23CS363

import argparse, csv, math, os
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_csv(path: str):
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Empty CSV")
    header, data = rows[0], rows[1:]
    return header, data


def split_X_y(header: List[str], data: List[List[str]], label_col: str):
    if label_col not in header:
        raise ValueError(f"Label column '{label_col}' not found.")
    yi = header.index(label_col)
    X_raw, y_raw = [], []
    for r in data:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        y_raw.append(r[yi])
        X_raw.append([v for i, v in enumerate(r) if i != yi])
    feat_names = [c for i, c in enumerate(header) if i != yi]
    return feat_names, X_raw, y_raw


class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: List[int], p_drop: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p_drop)]
            last = h
        layers += [nn.Linear(last, n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def encode_onehot(X_raw: List[List[str]], y_raw: List[str]):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = ohe.fit_transform(X_raw).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    return X, y, ohe, le


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        trues.append(yb.numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    acc = (p == t).mean().item()
    return acc, p, t


def main():
    ap = argparse.ArgumentParser(description="PyTorch MLP for categorical CSV (one-hot)")
    ap.add_argument("--data", required=True, help="Path to CSV")
    ap.add_argument("--label", default="class", help="Label column name")
    ap.add_argument("--val_size", type=float, default=0.1, help="Validation split from train")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split from full data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,128", help="Comma-separated hidden sizes")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--early", type=int, default=8, help="Early stopping patience (epochs)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    header, data = load_csv(args.data)
    feat_names, X_raw, y_raw = split_X_y(header, data, args.label)
    X, y, ohe, le = encode_onehot(X_raw, y_raw)

    # Split into train/val/test
    N = X.shape[0]
    n_test = max(1, int(N * args.test_size))
    n_trainval = N - n_test

    # Use torch random_split for reproducibility
    full = TabDataset(X, y)
    gen = torch.Generator().manual_seed(args.seed)
    trainval_set, test_set = random_split(full, [n_trainval, n_test], generator=gen)

    n_val = max(1, int(len(trainval_set) * args.val_size))
    n_train = len(trainval_set) - n_val
    train_set, val_set = random_split(trainval_set, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    in_dim = X.shape[1]
    n_classes = len(np.unique(y))
    hidden = [int(h) for h in args.hidden.split(",") if h.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim, n_classes, hidden, p_drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    patience = args.early

    print(f"\n=== PyTorch MLP Training ===")
    print(f"Data: {args.data}")
    print(f"Input dim: {in_dim}, Classes: {n_classes}, Hidden: {hidden}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        val_acc, _, _ = evaluate(model, val_loader, device)
        train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = args.early
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, p, t = evaluate(model, test_loader, device)
    print(f"\n=== Test Results ===")
    print(f"Accuracy: {test_acc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(t, p))
    print("\nClassification report:\n", classification_report(t, p, target_names=le.classes_))


if __name__ == "__main__":
    main()
