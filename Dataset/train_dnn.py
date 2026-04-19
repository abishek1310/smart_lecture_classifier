import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# CONFIG

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
DATASET_PATH   = os.path.join(BASE_DIR, "Dataset", "lecture_dataset.json")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Background Context", "Concept Definition", "Worked Example", "Exam Relevant"]
N_CLASSES   = 4
N_FOLDS     = 5
EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
DROPOUT     = 0.3
PATIENCE    = 5

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

print(f"Text embeddings  : {text_embeds.shape}")
print(f"Image embeddings : {image_embeds.shape}")
print(f"Labels           : {labels.shape}")
print(f"Has image        : {has_image.sum()} / {len(has_image)}")

# Class weights
counts       = Counter(labels)
total        = len(labels)
weight_array = np.array([total / (N_CLASSES * counts[i]) for i in range(N_CLASSES)], dtype=np.float32)
class_weights_tensor = torch.tensor(weight_array).to(DEVICE)
print(f"Class weights    : {weight_array.round(4)}")


# TF-IDF FEATURES

print("\nBuilding TF-IDF features")
tfidf   = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(texts).toarray().astype(np.float32)
print(f"TF-IDF shape     : {X_tfidf.shape}")

# Scale all features
scaler_tfidf = StandardScaler()
scaler_text  = StandardScaler()
scaler_image = StandardScaler()
X_tfidf_sc   = scaler_tfidf.fit_transform(X_tfidf).astype(np.float32)
X_text_sc    = scaler_text.fit_transform(text_embeds).astype(np.float32)
X_image_sc   = scaler_image.fit_transform(image_embeds).astype(np.float32)

TFIDF_DIM  = X_tfidf_sc.shape[1]
TEXT_DIM   = X_text_sc.shape[1]
IMAGE_DIM  = X_image_sc.shape[1]


# BUILD MULTIMODAL MODEL

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


# BUILD TEXT ONLY MODEL

def build_text_only_model(dropout=DROPOUT):
    tfidf_branch = nn.Sequential(
        nn.Linear(TFIDF_DIM, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
    )
    text_branch = nn.Sequential(
        nn.Linear(TEXT_DIM, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
    )
    fusion = nn.Sequential(
        nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, N_CLASSES)
    )
    return nn.ModuleDict({
        "tfidf_branch": tfidf_branch,
        "text_branch":  text_branch,
        "fusion":       fusion,
    })

def forward_text_only(model, tfidf, text_emb):
    tfidf_feat = model["tfidf_branch"](tfidf)
    text_feat  = model["text_branch"](text_emb)
    fused      = torch.cat([tfidf_feat, text_feat], dim=1)
    return model["fusion"](fused)


# DATALOADERS

def make_loader(idx, shuffle):
    ds = TensorDataset(
        torch.tensor(X_tfidf_sc[idx],  dtype=torch.float32),
        torch.tensor(X_text_sc[idx],   dtype=torch.float32),
        torch.tensor(X_image_sc[idx],  dtype=torch.float32),
        torch.tensor(has_image[idx],   dtype=torch.float32),
        torch.tensor(labels[idx],      dtype=torch.long),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

def make_text_loader(idx, shuffle):
    ds = TensorDataset(
        torch.tensor(X_tfidf_sc[idx], dtype=torch.float32),
        torch.tensor(X_text_sc[idx],  dtype=torch.float32),
        torch.tensor(labels[idx],     dtype=torch.long),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


# TRAIN ONE EPOCH — MULTIMODAL

def train_one_epoch(model, loader, optimizer, criterion):
    for m in model.values():
        m.train()
    all_preds, all_labels = [], []
    total_loss = 0
    for batch in loader:
        tfidf_b, text_b, image_b, has_img_b, label_b = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        output = forward_pass(model, tfidf_b, text_b, image_b, has_img_b)
        loss   = criterion(output, label_b)
        loss.backward()
        for m in model.values():
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(output.argmax(1).cpu().numpy())
        all_labels.extend(label_b.cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds, average="macro")


# EVALUATE — MULTIMODAL

def evaluate(model, loader):
    for m in model.values():
        m.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            tfidf_b, text_b, image_b, has_img_b, label_b = [b.to(DEVICE) for b in batch]
            output = forward_pass(model, tfidf_b, text_b, image_b, has_img_b)
            probs  = torch.softmax(output, dim=1).cpu().numpy()
            preds  = output.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(label_b.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_probs, all_labels


# TRAIN ONE EPOCH — TEXT ONLY

def train_one_epoch_text(model, loader, optimizer, criterion):
    for m in model.values():
        m.train()
    for batch in loader:
        tfidf_b, text_b, label_b = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        output = forward_text_only(model, tfidf_b, text_b)
        loss   = criterion(output, label_b)
        loss.backward()
        optimizer.step()


# EVALUATE — TEXT ONLY

def evaluate_text(model, loader):
    for m in model.values():
        m.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            tfidf_b, text_b, label_b = [b.to(DEVICE) for b in batch]
            output = forward_text_only(model, tfidf_b, text_b)
            probs  = torch.softmax(output, dim=1).cpu().numpy()
            preds  = output.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(label_b.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_probs, all_labels


# 5-FOLD CV — MULTIMODAL DNN

print("TRAINING MULTIMODAL DNN — 5-FOLD CV")

skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
all_preds = np.zeros(len(labels), dtype=int)
all_probs = np.zeros((len(labels), N_CLASSES))
fold_accs = []
fold_f1s  = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf_sc, labels)):
    print(f"\nFold {fold+1}/{N_FOLDS}")

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)

    model     = {k: v.to(DEVICE) for k, v in build_model().items()}
    params    = [p for m in model.values() for p in m.parameters()]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_f1      = 0
    best_states  = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_acc, vl_f1, _, _, _ = evaluate(model, val_loader)
        scheduler.step(vl_f1)
        print(f"  Epoch {epoch+1:2d}: train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | val_acc={vl_acc:.4f} val_f1={vl_f1:.4f}")

        if vl_f1 > best_f1:
            best_f1      = vl_f1
            best_states  = {k: {pk: pv.clone() for pk, pv in m.state_dict().items()} for k, m in model.items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    for k, m in model.items():
        m.load_state_dict(best_states[k])

    vl_acc, vl_f1, fold_preds, fold_probs, _ = evaluate(model, val_loader)
    all_preds[val_idx] = fold_preds
    all_probs[val_idx] = fold_probs
    fold_accs.append(vl_acc)
    fold_f1s.append(vl_f1)
    print(f"  Fold {fold+1} Best — Accuracy={vl_acc:.4f}  Macro F1={vl_f1:.4f}")
    torch.save(best_states, os.path.join(MODELS_DIR, f"dnn_fold_{fold+1}.pt"))


# MULTIMODAL RESULTS

print(f"MULTIMODAL DNN RESULTS")
print(f"Mean Accuracy : {np.mean(fold_accs):.4f} +- {np.std(fold_accs):.4f}")
print(f"Mean Macro F1 : {np.mean(fold_f1s):.4f} +- {np.std(fold_f1s):.4f}")
print(f"\nClassification Report:")
print(classification_report(labels, all_preds, target_names=LABEL_NAMES))
cm = confusion_matrix(labels, all_preds)
print(f"Confusion Matrix:")
print(cm)
try:
    auc = roc_auc_score(labels, all_probs, multi_class="ovr", average="macro")
    print(f"AUC-ROC (macro ovr): {auc:.4f}")
except Exception as e:
    print(f"AUC-ROC: could not compute ({e})")


# ABLATION STUDY — TEXT ONLY

print(f"ABLATION STUDY — TEXT ONLY")

abl_preds = np.zeros(len(labels), dtype=int)
abl_probs = np.zeros((len(labels), N_CLASSES))
abl_accs  = []
abl_f1s   = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf_sc, labels)):
    print(f"\nAblation Fold {fold+1}/{N_FOLDS}")

    train_loader = make_text_loader(train_idx, shuffle=True)
    val_loader   = make_text_loader(val_idx,   shuffle=False)

    model     = {k: v.to(DEVICE) for k, v in build_text_only_model().items()}
    params    = [p for m in model.values() for p in m.parameters()]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_f1      = 0
    best_states  = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        train_one_epoch_text(model, train_loader, optimizer, criterion)
        vl_acc, vl_f1, _, _, _ = evaluate_text(model, val_loader)

        if vl_f1 > best_f1:
            best_f1      = vl_f1
            best_states  = {k: {pk: pv.clone() for pk, pv in m.state_dict().items()} for k, m in model.items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    for k, m in model.items():
        m.load_state_dict(best_states[k])

    vl_acc, vl_f1, fold_preds, fold_probs, _ = evaluate_text(model, val_loader)
    abl_preds[val_idx] = fold_preds
    abl_probs[val_idx] = fold_probs
    abl_accs.append(vl_acc)
    abl_f1s.append(vl_f1)
    print(f"  Fold {fold+1} — Accuracy={vl_acc:.4f}  Macro F1={vl_f1:.4f}")


# ABLATION COMPARISON

print(f"ABLATION COMPARISON")
print(f"\nText Only DNN:")
print(f"  Mean Accuracy : {np.mean(abl_accs):.4f} +- {np.std(abl_accs):.4f}")
print(f"  Mean Macro F1 : {np.mean(abl_f1s):.4f} +- {np.std(abl_f1s):.4f}")
print(f"\nMultimodal DNN:")
print(f"  Mean Accuracy : {np.mean(fold_accs):.4f} +- {np.std(fold_accs):.4f}")
print(f"  Mean Macro F1 : {np.mean(fold_f1s):.4f} +- {np.std(fold_f1s):.4f}")

image_contribution = np.mean(fold_f1s) - np.mean(abl_f1s)
print(f"\nImage contribution : {image_contribution:+.4f} Macro F1")

print(f"\nPer-class F1 comparison:")
print(f"{'Class':<25} {'Text Only':>12} {'Multimodal':>12} {'Diff':>10}")
print("-" * 61)
for i, name in enumerate(LABEL_NAMES):
    f1_text  = f1_score(labels == i, abl_preds == i)
    f1_multi = f1_score(labels == i, all_preds == i)
    diff     = f1_multi - f1_text
    sign     = "+" if diff >= 0 else ""
    print(f"{name:<25} {f1_text:>12.4f} {f1_multi:>12.4f} {sign}{diff:.4f}")


# SAVE ALL RESULTS

results = {
    "multimodal_dnn": {
        "mean_accuracy":    float(np.mean(fold_accs)),
        "std_accuracy":     float(np.std(fold_accs)),
        "mean_macro_f1":    float(np.mean(fold_f1s)),
        "std_macro_f1":     float(np.std(fold_f1s)),
        "fold_accuracies":  [float(a) for a in fold_accs],
        "fold_f1s":         [float(f) for f in fold_f1s],
        "confusion_matrix": cm.tolist(),
    },
    "ablation_text_only": {
        "mean_accuracy": float(np.mean(abl_accs)),
        "std_accuracy":  float(np.std(abl_accs)),
        "mean_macro_f1": float(np.mean(abl_f1s)),
        "std_macro_f1":  float(np.std(abl_f1s)),
    },
    "image_contribution": float(image_contribution),
}

with open(os.path.join(RESULTS_DIR, "dnn_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved : {RESULTS_DIR}/dnn_results.json")
print(f"Models saved  : {MODELS_DIR}")
