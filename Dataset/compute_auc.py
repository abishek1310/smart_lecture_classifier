import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
DATASET_PATH   = os.path.join(BASE_DIR, "Dataset", "lecture_dataset.json")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
MODELS_DIR     = os.path.join(BASE_DIR, "models")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS    = 5
N_CLASSES  = 4
BATCH_SIZE = 64
DROPOUT    = 0.3

print(f"Device: {DEVICE}")


# LOAD DATA

print("\nLoading data")
text_embeds  = np.load(os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"))
image_embeds = np.load(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"))
labels       = np.load(os.path.join(EMBEDDINGS_DIR, "labels.npy"))
has_image    = np.load(os.path.join(EMBEDDINGS_DIR, "has_image.npy"))

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
texts = [s["text"] for s in data]
print(f"Samples : {len(labels)}")


# BUILD FEATURES

print("\nBuilding features")
tfidf   = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(texts).toarray().astype(np.float32)

scaler_tfidf = StandardScaler()
scaler_text  = StandardScaler()
scaler_image = StandardScaler()
X_tfidf_sc   = scaler_tfidf.fit_transform(X_tfidf).astype(np.float32)
X_bert       = scaler_text.fit_transform(text_embeds).astype(np.float32)
X_clip       = scaler_image.fit_transform(image_embeds).astype(np.float32)

X_tfidf_only      = X_tfidf_sc
X_tfidf_bert      = np.concatenate([X_tfidf_sc, X_bert],         axis=1)
X_tfidf_bert_clip = np.concatenate([X_tfidf_sc, X_bert, X_clip], axis=1)

TFIDF_DIM = X_tfidf_sc.shape[1]
TEXT_DIM  = X_bert.shape[1]
IMAGE_DIM = X_clip.shape[1]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


# DNN MODEL DEFINITION

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
        torch.tensor(X_bert[idx],      dtype=torch.float32),
        torch.tensor(X_clip[idx],      dtype=torch.float32),
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


# HELPER

def compute_auc_sklearn(X, y, model):
    all_probs = np.zeros((len(y), N_CLASSES))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold+1}/{N_FOLDS}", end="\r")
        model.fit(X[train_idx], y[train_idx])
        all_probs[val_idx] = model.predict_proba(X[val_idx])
    return round(roc_auc_score(y, all_probs, multi_class="ovr", average="macro"), 4)


# BASELINE MODELS — RECOMPUTE AUC-ROC

print("COMPUTING AUC-ROC FOR BASELINE MODELS")


print("\nLR TF-IDF only")
auc = compute_auc_sklearn(X_tfidf_only, labels, LogisticRegression(
    C=10, max_iter=1000, class_weight="balanced",
    solver="lbfgs", random_state=42, n_jobs=-1))
path = os.path.join(RESULTS_DIR, "LR_TF-IDF_only_results.json")
with open(path) as f: res = json.load(f)
res["auc_roc"] = auc
with open(path, "w") as f: json.dump(res, f, indent=2)
print(f"  AUC-ROC = {auc}")

print("\nLR TF-IDF+BERT")
auc = compute_auc_sklearn(X_tfidf_bert, labels, LogisticRegression(
    C=10, max_iter=1000, class_weight="balanced",
    solver="lbfgs", random_state=42, n_jobs=-1))
path = os.path.join(RESULTS_DIR, "LR_TF-IDFBERT_results.json")
with open(path) as f: res = json.load(f)
res["auc_roc"] = auc
with open(path, "w") as f: json.dump(res, f, indent=2)
print(f"  AUC-ROC = {auc}")

print("\nLR TF-IDF+BERT+CLIP")
auc = compute_auc_sklearn(X_tfidf_bert_clip, labels, LogisticRegression(
    C=10, max_iter=1000, class_weight="balanced",
    solver="lbfgs", random_state=42, n_jobs=-1))
path = os.path.join(RESULTS_DIR, "LR_TF-IDFBERTCLIP_results.json")
with open(path) as f: res = json.load(f)
res["auc_roc"] = auc
with open(path, "w") as f: json.dump(res, f, indent=2)
print(f"  AUC-ROC = {auc}")

print("\nSVM TF-IDF+BERT+CLIP")
auc = compute_auc_sklearn(X_tfidf_bert_clip, labels, SVC(
    C=10, kernel="rbf", gamma="scale",
    class_weight="balanced", random_state=42, probability=True))
path = os.path.join(RESULTS_DIR, "SVM_TF-IDFBERTCLIP_results.json")
with open(path) as f: res = json.load(f)
res["auc_roc"] = auc
with open(path, "w") as f: json.dump(res, f, indent=2)
print(f"  AUC-ROC = {auc}")


# DNN — LOAD SAVED FOLD MODELS AND COMPUTE AUC-ROC

print("COMPUTING AUC-ROC FOR DNN USING SAVED FOLD MODELS")

dnn_all_probs = np.zeros((len(labels), N_CLASSES))
folds_found   = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf_sc, labels)):
    model_path = os.path.join(MODELS_DIR, f"dnn_fold_{fold+1}.pt")
    if not os.path.exists(model_path):
        print(f"  Fold {fold+1} — model not found skipping")
        continue
    print(f"  Fold {fold+1} — loading model")
    model = {k: v.to(DEVICE) for k, v in build_model().items()}
    saved = torch.load(model_path, map_location=DEVICE)
    for k, m in model.items():
        m.load_state_dict(saved[k])
    dnn_all_probs[val_idx] = get_dnn_probs(model, val_idx)
    folds_found += 1

if folds_found == N_FOLDS:
    dnn_auc = round(roc_auc_score(labels, dnn_all_probs, multi_class="ovr", average="macro"), 4)
    print(f"\nDNN Multimodal AUC-ROC = {dnn_auc}")
    path = os.path.join(RESULTS_DIR, "dnn_results.json")
    with open(path) as f: res = json.load(f)
    res["multimodal_dnn"]["auc_roc"] = dnn_auc
    with open(path, "w") as f: json.dump(res, f, indent=2)
    print(f"  Updated dnn_results.json")
else:
    print(f"  Only {folds_found}/{N_FOLDS} fold models found — skipping DNN AUC-ROC")


# ENSEMBLE — READ FROM SAVED JSON 

print("READING ENSEMBLE AUC-ROC FROM SAVED JSON")

path = os.path.join(RESULTS_DIR, "ensemble_xgb_results.json")
with open(path) as f: res = json.load(f)
ens_auc = res.get("auc_roc", None)
print(f"  Ensemble LR+XGB+DNN AUC-ROC = {ens_auc}")


# SUMMARY

print("FINAL SUMMARY — ALL AUC-ROC SCORES")
print(f"{'Model':<35} {'AUC-ROC':>10}")
print("-" * 47)

files = {
    "LR TF-IDF only":       "LR_TF-IDF_only_results.json",
    "LR TF-IDF+BERT":       "LR_TF-IDFBERT_results.json",
    "LR TF-IDF+BERT+CLIP":  "LR_TF-IDFBERTCLIP_results.json",
    "SVM TF-IDF+BERT+CLIP": "SVM_TF-IDFBERTCLIP_results.json",
    "Ensemble LR+XGB+DNN":  "ensemble_xgb_results.json",
}
for name, fname in files.items():
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        res = json.load(f)
    print(f"{name:<35} {str(res.get('auc_roc', '—')):>10}")

with open(os.path.join(RESULTS_DIR, "dnn_results.json")) as f:
    res = json.load(f)
print(f"{'DNN Multimodal':<35} {str(res['multimodal_dnn'].get('auc_roc', '—')):>10}")
print(f"{'DNN Text Only':<35} {str(res['ablation_text_only'].get('auc_roc', '—')):>10}")

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Done")