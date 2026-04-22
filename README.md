# Smart Lecture Segment Classifier

**CS 6140 Machine Learning — Northeastern University, Spring 2026**  
**Authors:** Abishek Udayakrishna Surya Narayanan, Varun Mandepudi  
**Professor:** Ehsan Elhamifar

---

## Overview

This project builds a multimodal classifier that automatically labels 60-second lecture segments into 4 pedagogical categories using transcript text and slide frame images. The goal is to help graduate students navigate lecture content efficiently without re-watching full recordings.

### 4 Output Classes
- **Background Context** — supplementary or contextual information
- **Concept Definition** — professor introducing a new idea or definition
- **Worked Example** — professor solving a problem step by step
- **Exam Relevant** — content flagged as important for assessment

---

## Results Summary

| Model | Accuracy | Macro F1 | AUC-ROC |
|---|---|---|---|
| LR TF-IDF only | 0.834 | 0.719 | 0.888 |
| LR TF-IDF+SBERT | 0.764 | 0.613 | 0.891 |
| LR TF-IDF+SBERT+CLIP | 0.654 | 0.467 | 0.897 |
| SVM TF-IDF+SBERT+CLIP | 0.702 | 0.367 | 0.904 |
| DNN Text Only (ablation) | 0.793 | 0.638 | 0.840 |
| DNN Multimodal (PyTorch) | 0.822 | 0.665 | 0.855 |
| **Ensemble LR+XGB+DNN** | **0.873** | **0.761** | **0.930** |

---

## Repository Structure

```
smart_lecture_classifier/
├── Dataset/
│   ├── build_complete_dataset.py       # Step 1: Build dataset from MIT OCW + YouTube
│   ├── extract_frames.py               # Step 2: Extract slide images from YouTube
│   ├── retry_frames.py                 # Step 2b: Retry failed frame extractions
│   ├── extract_embeddings.py           # Step 3: Extract SBERT + CLIP embeddings
│   ├── train_baselines.py              # Step 4: Train LR + SVM baselines
│   ├── train_dnn.py                    # Step 5: Train multimodal DNN + ablation
│   ├── ensemble.py                     # Step 6: Train LR + XGBoost + DNN ensemble
│   ├── compute_auc.py                  # Step 7: Compute AUC-ROC for all models
│   ├── verify_dataset.py               # Utility: Verify dataset quality
│   ├── verify_match.py                 # Utility: Verify text-image alignment
│   ├── lecture_dataset.json            # Pre-built dataset (7,349 samples)
│   ├── dataset_metadata.json           # Dataset statistics and label map
│   ├── static_resources/               # MIT 6.034 AI raw SRT + PDF files
│   └── static_resources_fall_2024/     # MIT 6.7960 Deep Learning 2024 WebVTT + PDF files
├── results/
│   ├── LR_TF-IDF_only_results.json
│   ├── LR_TF-IDFBERT_results.json
│   ├── LR_TF-IDFBERTCLIP_results.json
│   ├── SVM_TF-IDFBERTCLIP_results.json
│   ├── dnn_results.json
│   ├── ensemble_results.json
│   └── ensemble_xgb_results.json
├── .gitignore
└── README.md
```

---

## Installation

### Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended — tested on RTX 5060, CUDA 12.8)
- Node.js (for YouTube JavaScript challenge solver)
- ffmpeg (for frame extraction)

### Step 1 — Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 2 — Install Python Dependencies

```bash
pip install sentence-transformers open-clip-torch scikit-learn xgboost
pip install yt-dlp Pillow tqdm transformers numpy
```

### Step 3 — Install ffmpeg

Download from https://github.com/BtbN/ffmpeg-builds/releases  
Extract and add the `bin/` folder to your system PATH.

### Step 4 — Install Node.js

Download from https://nodejs.org and install.

---

## Replicating Results

### Option A — Use Pre-built Dataset (Recommended)

The dataset is already included as `Dataset/lecture_dataset.json`. Skip Steps 1 and 2.

```bash
# Step 3 — Extract embeddings (~5 minutes on GPU)
python Dataset/extract_embeddings.py

# Step 4 — Train baseline models (~15 minutes)
python Dataset/train_baselines.py

# Step 5 — Train multimodal DNN + ablation (~45 minutes on GPU)
python Dataset/train_dnn.py

# Step 6 — Run ensemble (~5 minutes)
python Dataset/ensemble.py

# Step 7 — Compute AUC-ROC for all models 
python Dataset/compute_auc.py
```

Results will be saved to the `results/` folder.

---

### Option B — Rebuild Dataset From Scratch

#### Step 1 — Build Dataset

```bash
python Dataset/build_complete_dataset.py
```

This parses MIT 6.034 SRT files, MIT 6.7960 WebVTT files, and downloads YouTube transcripts for 3 courses. Saves `Dataset/lecture_dataset.json` with 7,349 labeled samples.

#### Step 2 — Extract Slide Frame Images

YouTube frame extraction requires browser cookies to avoid rate limiting:

1. Install the **"Get cookies.txt LOCALLY"** Chrome extension
2. Go to youtube.com and log in
3. Click the extension and export cookies
4. Save as `cookies.txt` in the project root

```bash
python Dataset/extract_frames.py
```

If some videos fail, run the retry script:

```bash
python Dataset/retry_frames.py
```

Frames are saved to `frames/` folder (~1GB, not included in repo).

#### Step 3 — Extract Embeddings

```bash
python Dataset/extract_embeddings.py
```

Saves to `embeddings/` folder (not included in repo — regenerate locally).

#### Step 4 — Train Baselines

```bash
python Dataset/train_baselines.py
```

#### Step 5 — Train DNN

```bash
python Dataset/train_dnn.py
```

Saves fold models to `models/` folder (not included in repo).

#### Step 6 — Run Ensemble

```bash
python Dataset/ensemble.py
```

#### Step 7 — Compute AUC-ROC

```bash
python Dataset/compute_auc.py
```

---

## Dataset Details

| Property | Value |
|---|---|
| Total samples | 7,349 |
| Segment duration | 60 seconds |
| Average words per segment | 149 |
| Number of courses | 5 |
| Number of classes | 4 |
| Samples with slide images | 4,573 (62%) |

| Course | Source | Samples |
|---|---|---|
| MIT 6.034 Artificial Intelligence | MIT OCW SRT | 1,409 |
| MIT 6.7960 Deep Learning 2024 | MIT OCW WebVTT | 1,727 |
| Stanford CS231n Computer Vision | YouTube | 1,137 |
| Andrew NG CS229 Machine Learning | YouTube | 1,523 |
| MIT 18.06 Linear Algebra | YouTube | 1,553 |

| Class | Samples | Class Weight |
|---|---|---|
| Background Context | 5,284 | 0.35 |
| Concept Definition | 854 | 2.15 |
| Worked Example | 667 | 2.75 |
| Exam Relevant | 544 | 3.38 |

---

## Model Architecture

### Model 1 — Logistic Regression (Text Baseline)
- Features: TF-IDF vectors (3,000 unigrams/bigrams, sublinear TF)
- Class weights: balanced
- Evaluation: 5-fold stratified CV

### Model 2 — SVM (Multimodal Baseline)
- Features: TF-IDF (3,000) + SBERT (384) + CLIP (512) = 3,896 dim
- Kernel: RBF, class_weight=balanced
- Evaluation: 5-fold stratified CV

### Model 3 — Multimodal DNN (PyTorch)
Three parallel input branches fused via cross-attention:
- TF-IDF branch: Linear(3000→512→256) + BatchNorm + ReLU + Dropout
- SBERT branch: Linear(384→256→128) + BatchNorm + ReLU + Dropout
- CLIP branch: Linear(512→256→128) + BatchNorm + ReLU + Dropout + image gate
- Cross-attention: text tokens attend to image patch features
- Fusion: concatenate(256+128+128=512) → Linear(512→256→128→4)
- Training: AdamW, CrossEntropyLoss with class weights, early stopping (patience=5)

### Model 4 — Ensemble LR + XGBoost + DNN
- Weighted probability average: LR(0.4) + XGB(0.3) + DNN(0.3)
- XGBoost: 300 estimators, max_depth=6, GPU accelerated
- Uses saved DNN fold models — no retraining required

---

## Key Findings

1. **TF-IDF outperforms BERT** on keyword-driven weakly supervised labels — surface-level lexical features are maximally discriminative when labels are generated by keyword matching.

2. **Slide images contribute +0.027 Macro F1** overall. The visual modality helps Exam Relevant most (+0.035 F1) because whiteboard equations are visually distinctive.

3. **Ensemble achieves best results** (0.761 Macro F1, 0.930 AUC-ROC) by combining complementary strengths of keyword matching (LR), non-linear feature interactions (XGBoost), and semantic+visual understanding (DNN).

### Ablation Study — Image Contribution Per Class

| Class | Text Only F1 | Multimodal F1 | Contribution |
|---|---|---|---|
| Background Context | 0.871 | 0.892 | +0.021 |
| Concept Definition | 0.656 | 0.682 | +0.026 |
| Worked Example | 0.629 | 0.653 | +0.024 |
| Exam Relevant | 0.399 | 0.434 | +0.035 |
| **Overall** | **0.638** | **0.665** | **+0.027** |

---

## Verifying Dataset Quality

```bash
python Dataset/verify_dataset.py
python Dataset/verify_match.py
```

---

## Notes

- Frame extraction requires `cookies.txt` — see Step 2 above
- Embeddings and model weights are not included due to size — regenerate using Steps 3-5
- All results are pre-computed in `results/` and can be viewed without rerunning any code
- Training was done on NVIDIA RTX 5060 (8GB VRAM, CUDA 12.8, PyTorch 2.10)
- AUC-ROC scores for baseline models and DNN are computed automatically during training and saved to results JSON files
- To recompute all AUC-ROC scores after retraining run: `python Dataset/compute_auc.py`