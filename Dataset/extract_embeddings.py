import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import open_clip


# CONFIG

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH   = os.path.join(BASE_DIR, "Dataset", "lecture_dataset.json")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# LOAD DATASET

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples     : {len(data)}")
print(f"Samples with image: {sum(1 for s in data if s.get('has_image'))}")


# LOAD SENTENCE-BERT

print("\nLoading Sentence-BERT (all-MiniLM-L6-v2)")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEVICE))
print("Sentence-BERT loaded")


# LOAD CLIP

print("\nLoading CLIP")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(DEVICE)
clip_model.eval()
print("CLIP loaded")


# EXTRACT SENTENCE-BERT TEXT EMBEDDINGS

print("\nExtracting Sentence-BERT text embeddings")

texts      = [s["text"] for s in data]
BATCH_SIZE = 64

text_embeds = sbert_model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    device=str(DEVICE)
)

print(f"Text embeddings shape: {text_embeds.shape}")

# EXTRACT CLIP IMAGE EMBEDDINGS

print("\nExtracting CLIP image embeddings")

image_embeds = np.zeros((len(data), 512), dtype=np.float32)

for i, s in enumerate(tqdm(data)):
    if not s.get("has_image") or not s.get("frame_path"):
        image_embeds[i] = np.zeros(512, dtype=np.float32)
        continue

    frame_path = s["frame_path"]
    if not os.path.exists(frame_path):
        image_embeds[i] = np.zeros(512, dtype=np.float32)
        continue

    try:
        image = Image.open(frame_path).convert("RGB")
        image = clip_preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = clip_model.encode_image(image)
            embedding = embedding.cpu().numpy().squeeze()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        image_embeds[i] = embedding

    except Exception as e:
        print(f"\nFailed on {frame_path}: {e}")
        image_embeds[i] = np.zeros(512, dtype=np.float32)

print(f"Image embeddings shape: {image_embeds.shape}")


# LABELS AND METADATA

label_map = {
    "Background Context": 0,
    "Concept Definition": 1,
    "Worked Example":     2,
    "Exam Relevant":      3
}

labels    = np.array([label_map[s["label"]] for s in data], dtype=np.int64)
has_image = np.array([1 if s.get("has_image") else 0 for s in data], dtype=np.int64)
courses   = np.array([s["course"] for s in data])

print(f"\nLabel distribution:")
from collections import Counter
counts = Counter(labels)
for label, idx in label_map.items():
    print(f"  {label:25s}: {counts[idx]}")


# SAVE

print("\nSaving embeddings")

np.save(os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"),  text_embeds)
np.save(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"), image_embeds)
np.save(os.path.join(EMBEDDINGS_DIR, "labels.npy"),           labels)
np.save(os.path.join(EMBEDDINGS_DIR, "has_image.npy"),        has_image)
np.save(os.path.join(EMBEDDINGS_DIR, "courses.npy"),          courses)


# VERIFY

print("\nVerifying saved files:")
for fname in ["text_embeddings.npy", "image_embeddings.npy", "labels.npy"]:
    path = os.path.join(EMBEDDINGS_DIR, fname)
    arr  = np.load(path)
    size = os.path.getsize(path) / (1024 * 1024)
    print(f"  {fname:30s}: shape={arr.shape}  size={size:.1f}MB")


print(f"EMBEDDING EXTRACTION COMPLETE")
print(f"Text embeddings   : {text_embeds.shape}  (Sentence-BERT all-MiniLM-L6-v2)")
print(f"Image embeddings  : {image_embeds.shape}  (CLIP ViT-B/32)")
print(f"Real images       : {np.sum(has_image)} samples")
print(f"Zero image        : {np.sum(has_image == 0)} samples")
print(f"Saved to          : {EMBEDDINGS_DIR}")
print(f"\nReady for model training")