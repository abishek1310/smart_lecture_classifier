import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# CONFIG

EMBEDDINGS_DIR = r"E:\ML_project\embeddings"
DATASET_PATH   = r"E:\ML_project\dataset\lecture_dataset.json"
RESULTS_DIR    = r"E:\ML_project\results"
MODELS_DIR     = r"E:\ML_project\models"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Background Context", "Concept Definition", "Worked Example", "Exam Relevant"]
N_CLASSES   = 4
N_FOLDS     = 5
BATCH_SIZE  = 64
DROPOUT     = 0.3

print(f"Device: {DEVICE}")


# LOAD DATA

print("LOADING DATA")

text_embeds  = np.load(os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"))
image_embeds = np.load(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"))
labels       = np.load(os.path.join(EMBEDDINGS_DIR, "labels.npy"))
has_image    = np.load(os.path.join(EMBEDDINGS_DIR, "has_image.npy"))

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [s["text"] for s in data]

print(f"Samples          : {len(labels)}")
print(f"Has image        : {has_image.sum()} / {len(has_image)}")

counts       = Counter(labels)
total        = len(labels)
weight_array = np.array([total / (N_CLASSES * counts[i]) for i in range(N_CLASSES)], dtype=np.float32)
class_weights_tensor = torch.tensor(weight_array).to(DEVICE)


# FEATURES

print("\nBuilding features")
tfidf   = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(texts).toarray().astype(np.float32)

scaler_tfidf = StandardScaler()
scaler_text  = StandardScaler()
scaler_image = StandardScaler()
X_tfidf_sc   = scaler_tfidf.fit_transform(X_tfidf).astype(np.float32)
X_text_sc    = scaler_text.fit_transform(text_embeds).astype(np.float32)
X_image_sc   = scaler_image.fit_transform(image_embeds).astype(np.float32)

# Combined features for XGBoost
X_combined = np.concatenate([X_tfidf_sc, X_text_sc, X_image_sc], axis=1)

TFIDF_DIM = X_tfidf_sc.shape[1]
TEXT_DIM  = X_text_sc.shape[1]
IMAGE_DIM = X_image_sc.shape[1]

print(f"TF-IDF           : {X_tfidf_sc.shape}")
print(f"Text (SBERT)     : {X_text_sc.shape}")
print(f"Image (CLIP)     : {X_image_sc.shape}")
print(f"Combined (XGB)   : {X_combined.shape}")


# DNN MODEL

def build_model(dropout=DROPOUT):
    tfidf_branch = nn.Sequential(
        nn.Linear(TFIDF_DIM, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
    )
    text_branch = nn.Sequential(
        nn.Linear(TEXT_DIM, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
    )
    image_branch = nn.Sequential(
        nn.Linear(IMAGE_DIM, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
    )
    image_gate = nn.Sequential(
        nn.Linear(IMAGE_DIM + 1, 64), nn.ReLU(),
        nn.Linear(64, 1),             nn.Sigmoid()
    )
    cross_attn = nn.MultiheadAttention(
        embed_dim=128, num_heads=4, dropout=dropout, batch_first=True
    )
    fusion = nn.Sequential(
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, N_CLASSES)
    )
    return nn.ModuleDict({
        "tfidf_branch": tfidf_branch,
        "text_branch":  text_branch,
        "image_branch": image_branch,
        "image_gate":   image_gate,
        "cross_attn":   cross_attn,
        "fusion":       fusion,
    })

def forward_pass(model, tfidf, text_emb, image_emb, has_img):
    tfidf_feat  = model["tfidf_branch"](tfidf)
    text_feat   = model["text_branch"](text_emb)
    image_feat  = model["image_branch"](image_emb)
    gate_input  = torch.cat([image_emb, has_img.unsqueeze(1)], dim=1)
    gate        = model["image_gate"](gate_input)
    image_feat  = image_feat * gate
    text_q      = text_feat.unsqueeze(1)
    image_k     = image_feat.unsqueeze(1)
    attended, _ = model["cross_attn"](text_q, image_k, image_k)
    attended    = attended.squeeze(1)
    fused       = torch.cat([tfidf_feat, text_feat, attended], dim=1)
    return model["fusion"](fused)

def get_dnn_probs(model, idx):
    ds = TensorDataset(
        torch.tensor(X_tfidf_sc[idx],  dtype=torch.float32),
        torch.tensor(X_text_sc[idx],   dtype=torch.float32),
        torch.tensor(X_image_sc[idx],  dtype=torch.float32),
        torch.tensor(has_image[idx],   dtype=torch.float32),
        torch.tensor(labels[idx],      dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    for m in model.values():
        m.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            tfidf_b, text_b, image_b, has_img_b, _ = [b.to(DEVICE) for b in batch]
            output = forward_pass(model, tfidf_b, text_b, image_b, has_img_b)
            probs  = torch.softmax(output, dim=1).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


# 5-FOLD ENSEMBLE — LR + XGB + DNN


print("ENSEMBLE — LR + XGBoost + DNN")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# XGBoost class weights
xgb_weights = {i: float(total / (N_CLASSES * counts[i])) for i in range(N_CLASSES)}

# Weights to try
LR_WEIGHT  = 0.4
XGB_WEIGHT = 0.3
DNN_WEIGHT = 0.3

ensemble_preds = np.zeros(len(labels), dtype=int)
ensemble_probs = np.zeros((len(labels), N_CLASSES))
fold_accs      = []
fold_f1s       = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf_sc, labels)):
    print(f"\nFold {fold+1}/{N_FOLDS}")

    # Train LR 
    lr = LogisticRegression(
        C=10, max_iter=1000,
        class_weight="balanced",
        solver="lbfgs", random_state=42, n_jobs=-1
    )
    lr.fit(X_tfidf_sc[train_idx], labels[train_idx])
    lr_probs = lr.predict_proba(X_tfidf_sc[val_idx])

    #  Train XGBoost 
    sample_weights = np.array([xgb_weights[l] for l in labels[train_idx]])
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        device="cuda"
    )
    xgb.fit(X_combined[train_idx], labels[train_idx], sample_weight=sample_weights)
    xgb_probs = xgb.predict_proba(X_combined[val_idx])
    print(f"  XGBoost trained")

    #  Load DNN 
    model_path = os.path.join(MODELS_DIR, f"dnn_fold_{fold+1}.pt")
    if os.path.exists(model_path):
        model = {k: v.to(DEVICE) for k, v in build_model().items()}
        saved = torch.load(model_path, map_location=DEVICE)
        for k, m in model.items():
            m.load_state_dict(saved[k])
        dnn_probs = get_dnn_probs(model, val_idx)
        print(f"  DNN loaded")
    else:
        print(f"  WARNING: No saved DNN using LR only")
        dnn_probs = lr_probs

    #  Ensemble 
    combined_probs = LR_WEIGHT * lr_probs + XGB_WEIGHT * xgb_probs + DNN_WEIGHT * dnn_probs
    combined_preds = np.argmax(combined_probs, axis=1)

    ensemble_preds[val_idx] = combined_preds
    ensemble_probs[val_idx] = combined_probs

    acc = accuracy_score(labels[val_idx], combined_preds)
    f1  = f1_score(labels[val_idx], combined_preds, average="macro")
    fold_accs.append(acc)
    fold_f1s.append(f1)
    print(f"  Fold {fold+1} — Accuracy={acc:.4f}  Macro F1={f1:.4f}")


# RESULTS

print(f"ENSEMBLE RESULTS  (LR={LR_WEIGHT}, XGB={XGB_WEIGHT}, DNN={DNN_WEIGHT})")
print(f"Mean Accuracy : {np.mean(fold_accs):.4f} +- {np.std(fold_accs):.4f}")
print(f"Mean Macro F1 : {np.mean(fold_f1s):.4f} +- {np.std(fold_f1s):.4f}")

print(f"\nClassification Report:")
print(classification_report(labels, ensemble_preds, target_names=LABEL_NAMES))

cm = confusion_matrix(labels, ensemble_preds)
print(f"Confusion Matrix:")
print(cm)

try:
    auc = roc_auc_score(labels, ensemble_probs, multi_class="ovr", average="macro")
    print(f"\nAUC-ROC (macro ovr): {auc:.4f}")
except Exception as e:
    print(f"AUC-ROC: could not compute ({e})")

print(f"\n{'='*60}")
print(f"COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<40} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 62)
print(f"{'LR TF-IDF only':<40} {'0.8341':>10} {'0.7192':>10}")
print(f"{'DNN Multimodal':<40} {'0.8212':>10} {'0.6610':>10}")
print(f"{'Ensemble LR+DNN (0.7/0.3)':<40} {'0.8159':>10} {'0.6941':>10}")
print(f"{'Ensemble LR+XGB+DNN':<40} {np.mean(fold_accs):>10.4f} {np.mean(fold_f1s):>10.4f}")
print("-" * 62)


# SAVE

results = {
    "model":            "Ensemble LR+XGB+DNN",
    "lr_weight":        LR_WEIGHT,
    "xgb_weight":       XGB_WEIGHT,
    "dnn_weight":       DNN_WEIGHT,
    "mean_accuracy":    float(np.mean(fold_accs)),
    "std_accuracy":     float(np.std(fold_accs)),
    "mean_macro_f1":    float(np.mean(fold_f1s)),
    "std_macro_f1":     float(np.std(fold_f1s)),
    "fold_accuracies":  [float(a) for a in fold_accs],
    "fold_f1s":         [float(f) for f in fold_f1s],
    "confusion_matrix": cm.tolist(),
}
with open(os.path.join(RESULTS_DIR, "ensemble_xgb_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {RESULTS_DIR}/ensemble_xgb_results.json")
