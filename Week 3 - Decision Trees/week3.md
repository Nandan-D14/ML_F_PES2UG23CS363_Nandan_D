
# 📘 README – Week 3: Decision Trees (Sklearn & PyTorch)

This folder contains two implementations for **classification on categorical datasets** (`mushroom.csv`, `tictactoe.csv`, `Nursery.csv`):

1. **Sklearn Decision Tree (`student_sklearn.py`)**
   Uses `scikit-learn`’s `DecisionTreeClassifier` with categorical encoding (OrdinalEncoder).
   Mirrors ID3-style trees using entropy as default criterion.

2. **PyTorch MLP (`student_pytorch.py`)**
   A simple feed-forward neural network trained on one-hot encoded categorical features.
   Serves as a neural baseline for comparison against decision trees.

---

## 📂 Files

* `student_sklearn.py` – Train/test DecisionTreeClassifier with configurable hyperparameters.
* `student_pytorch.py` – PyTorch MLP for categorical classification with early stopping.
* `mushroom.csv`, `tictactoe.csv`, `Nursery.csv` – Datasets provided for classification tasks.

---

## ⚙️ Dependencies

Install requirements (Python ≥ 3.8):

```bash
pip install scikit-learn torch numpy
```

---

## ▶️ Sklearn Implementation

### Run

```bash
python student_sklearn.py --data "mushroom.csv" --label class
```

### Options

| Argument              | Default      | Description                                           |
| --------------------- | ------------ | ----------------------------------------------------- |
| `--data`              | **required** | Path to CSV dataset                                   |
| `--label`             | `class`      | Name of label/target column                           |
| `--criterion`         | `entropy`    | Splitting criterion: `gini`, `entropy`, or `log_loss` |
| `--max_depth`         | `None`       | Limit tree depth                                      |
| `--min_samples_split` | `2`          | Minimum samples required to split                     |
| `--grid`              | *False*      | Run small GridSearchCV for hyperparam tuning          |
| `--test_size`         | `0.2`        | Test set proportion                                   |
| `--seed`              | `0`          | Random seed                                           |

### Example Commands

```bash
# Mushroom dataset, default settings
python student_sklearn.py --data "mushroom.csv" --label class

# TicTacToe with grid search
python student_sklearn.py --data "tictactoe.csv" --label class --grid

# Nursery dataset with controlled depth
python student_sklearn.py --data "Nursery.csv" --label class --criterion entropy --max_depth 10
```

### Output

* **Accuracy** on test set
* **Confusion matrix**
* **Classification report (precision/recall/F1)**
* **Text visualization** of learned tree

---

## ▶️ PyTorch Implementation

### Run

```bash
python student_pytorch.py --data "mushroom.csv" --label class --epochs 40 --hidden 256,128
```

### Options

| Argument      | Default      | Description                          |
| ------------- | ------------ | ------------------------------------ |
| `--data`      | **required** | Path to CSV dataset                  |
| `--label`     | `class`      | Name of label/target column          |
| `--epochs`    | `50`         | Training epochs                      |
| `--batch`     | `256`        | Batch size                           |
| `--lr`        | `1e-3`       | Learning rate                        |
| `--hidden`    | `256,128`    | Hidden layer sizes (comma-separated) |
| `--dropout`   | `0.1`        | Dropout probability                  |
| `--early`     | `8`          | Early stopping patience (epochs)     |
| `--test_size` | `0.2`        | Test set proportion                  |
| `--val_size`  | `0.1`        | Validation split from train set      |
| `--seed`      | `0`          | Random seed                          |

### Example Commands

```bash
# Mushroom dataset, default hidden layers (256,128)
python student_pytorch.py --data "mushroom.csv" --label class --epochs 40

# TicTacToe dataset, deeper network, patience 10
python student_pytorch.py --data "tictactoe.csv" --label class --epochs 60 --hidden 128,64 --early 10

# Nursery dataset, larger batch size
python student_pytorch.py --data "Nursery.csv" --label class --epochs 50 --batch 512
```

### Output

* **Training loss per epoch**
* **Validation accuracy per epoch (early stopping applied)**
* **Test set accuracy**
* **Confusion matrix**
* **Classification report (precision/recall/F1)**

---

## 📝 Notes

* Both scripts **auto-handle categorical features**:

  * Sklearn → Ordinal encoding for features, Label encoding for target
  * PyTorch → One-hot encoding for features, Label encoding for target

* **Sklearn tree** mimics ID3 (entropy) by default, but you can also try Gini/Log Loss.

* **PyTorch MLP** provides a neural approach; not a tree but useful as a baseline comparison.

* Ensure datasets are in **CSV format with headers**, with the label column named `class` (or override with `--label`).

---

## 📊 Suggested Workflow

1. Run `student_lab.py` on each dataset and record **accuracy + confusion matrix**.
2. Run `lab_student.py` on each dataset with suitable hidden layers.
3. Compare results:

   * Trees are interpretable (rule-based).
   * Neural network may achieve higher/lower accuracy depending on dataset size & complexity.

---

## ✨ Example Result Snapshot

For `mushroom.csv` with Sklearn (entropy):

```
=== sklearn Decision Tree Results ===
Accuracy: 1.0000
Confusion matrix:
 [[4200    0]
  [   0 3916]]

Classification report:
              precision    recall  f1-score   support
         e       1.00      1.00      1.00      4200
         p       1.00      1.00      1.00      3916
```

For the same dataset with PyTorch MLP:

```
Epoch 001 | train_loss=0.2103 | val_acc=0.9512
Epoch 002 | train_loss=0.0856 | val_acc=0.9874
...
=== Test Results ===
Accuracy: 0.9995
Confusion matrix:
 [[ 839    0]
  [   1  804]]
```
------
