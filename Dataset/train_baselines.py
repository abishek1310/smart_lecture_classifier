import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")


# CONFIG

EMBEDDINGS_DIR = r"E:\ML_project\embeddings"
DATASET_PATH   = r"E:\ML_project\dataset\lecture_dataset.json"
RESULTS_DIR    = r"E:\ML_project\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_NAMES = ["Background Context", "Concept Definition", "Worked Example", "Exam Relevant"]
N_FOLDS     = 5


# LOAD DATA


print("LOADING DATA")


text_embeds  = np.load(os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"))
image_embeds = np.load(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"))
labels       = np.load(os.path.join(EMBEDDINGS_DIR, "labels.npy"))

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [s["text"] for s in data]

print(f"Text embeddings  : {text_embeds.shape}")
print(f"Image embeddings : {image_embeds.shape}")
print(f"Labels           : {labels.shape}")

from collections import Counter
counts  = Counter(labels)
total   = len(labels)
n_class = len(counts)
class_weights = {i: round(total / (n_class * counts[i]), 4) for i in range(n_class)}
print(f"Class weights    : {class_weights}")


# FEATURE ENGINEERING

print("\nBuilding features")

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(texts).toarray().astype(np.float32)
print(f"TF-IDF features         : {X_tfidf.shape}")

# Normalize BERT and CLIP
scaler_text  = StandardScaler()
scaler_image = StandardScaler()
X_bert  = scaler_text.fit_transform(text_embeds).astype(np.float32)
X_clip  = scaler_image.fit_transform(image_embeds).astype(np.float32)

# Feature sets
X_tfidf_only    = X_tfidf
X_tfidf_bert    = np.concatenate([X_tfidf, X_bert],         axis=1)
X_tfidf_bert_clip = np.concatenate([X_tfidf, X_bert, X_clip], axis=1)

print(f"TF-IDF only             : {X_tfidf_only.shape}")
print(f"TF-IDF + BERT           : {X_tfidf_bert.shape}")
print(f"TF-IDF + BERT + CLIP    : {X_tfidf_bert_clip.shape}")


# EVALUATION HELPER

def evaluate_cv(model_name, X, y, model, n_folds=5):
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros((len(y), n_class))
    fold_accs = []
    fold_f1s  = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        all_preds[val_idx] = preds

        if hasattr(model, "predict_proba"):
            all_probs[val_idx] = model.predict_proba(X_val)
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_val)
            df = df - df.min(axis=1, keepdims=True)
            df = df / (df.sum(axis=1, keepdims=True) + 1e-8)
            all_probs[val_idx] = df

        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold+1}: Accuracy={acc:.4f}  Macro F1={f1:.4f}")

    print(f"\nCross-Validation Results ({n_folds} folds):")
    print(f"  Mean Accuracy : {np.mean(fold_accs):.4f} +- {np.std(fold_accs):.4f}")
    print(f"  Mean Macro F1 : {np.mean(fold_f1s):.4f} +- {np.std(fold_f1s):.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y, all_preds, target_names=LABEL_NAMES))

    cm = confusion_matrix(y, all_preds)
    print(f"Confusion Matrix:")
    print(cm)

    try:
        auc = roc_auc_score(y, all_probs, multi_class="ovr", average="macro")
        print(f"\nAUC-ROC (macro ovr): {auc:.4f}")
    except Exception as e:
        print(f"AUC-ROC: could not compute ({e})")

    results = {
        "model":            model_name,
        "mean_accuracy":    float(np.mean(fold_accs)),
        "std_accuracy":     float(np.std(fold_accs)),
        "mean_macro_f1":    float(np.mean(fold_f1s)),
        "std_macro_f1":     float(np.std(fold_f1s)),
        "fold_accuracies":  [float(a) for a in fold_accs],
        "fold_f1s":         [float(f) for f in fold_f1s],
        "confusion_matrix": cm.tolist(),
    }
    fname = model_name.replace(" ", "_").replace("/", "_").replace("+", "").replace("(", "").replace(")", "") + "_results.json"
    with open(os.path.join(RESULTS_DIR, fname), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {fname}")

    return float(np.mean(fold_accs)), float(np.mean(fold_f1s))


# MODEL 1 — LOGISTIC REGRESSION + TF-IDF (TEXT ONLY BASELINE)

lr1 = LogisticRegression(
    C=10,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
    n_jobs=-1
)
lr1_acc, lr1_f1 = evaluate_cv(
    "LR TF-IDF only", X_tfidf_only, labels, lr1
)


# MODEL 2 — LOGISTIC REGRESSION + TF-IDF + BERT

lr2 = LogisticRegression(
    C=10,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
    n_jobs=-1
)
lr2_acc, lr2_f1 = evaluate_cv(
    "LR TF-IDF+BERT", X_tfidf_bert, labels, lr2
)


# MODEL 3 — LOGISTIC REGRESSION + TF-IDF + BERT + CLIP

lr3 = LogisticRegression(
    C=10,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
    n_jobs=-1
)
lr3_acc, lr3_f1 = evaluate_cv(
    "LR TF-IDF+BERT+CLIP", X_tfidf_bert_clip, labels, lr3
)


# MODEL 4 — SVM + TF-IDF + BERT + CLIP (MULTIMODAL BASELINE)

svm = SVC(
    C=10,
    kernel="rbf",
    gamma="scale",
    class_weight="balanced",
    random_state=42,
    probability=True
)
svm_acc, svm_f1 = evaluate_cv(
    "SVM TF-IDF+BERT+CLIP", X_tfidf_bert_clip, labels, svm
)


# SUMMARY

print("FINAL COMPARISON SUMMARY")
print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10}")

print(f"{'LR TF-IDF only':<35} {lr1_acc:>10.4f} {lr1_f1:>10.4f}")
print(f"{'LR TF-IDF+BERT':<35} {lr2_acc:>10.4f} {lr2_f1:>10.4f}")
print(f"{'LR TF-IDF+BERT+CLIP':<35} {lr3_acc:>10.4f} {lr3_f1:>10.4f}")
print(f"{'SVM TF-IDF+BERT+CLIP':<35} {svm_acc:>10.4f} {svm_f1:>10.4f}")

best_f1   = max(lr1_f1, lr2_f1, lr3_f1, svm_f1)
best_name = ["LR TF-IDF", "LR TF-IDF+BERT", "LR TF-IDF+BERT+CLIP", "SVM TF-IDF+BERT+CLIP"][
    [lr1_f1, lr2_f1, lr3_f1, svm_f1].index(best_f1)
]
print(f"\nBest baseline : {best_name}  (Macro F1={best_f1:.4f})")
print(f"Results saved : {RESULTS_DIR}")
