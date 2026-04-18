# Smart Lecture Segment Classifier

Multimodal classification of 60-second lecture segments into 4 pedagogical categories using transcript text and slide frame images.

## Problem Statement

Graduate students waste hours re-watching lecture recordings to find exam-relevant content, worked examples, and key definitions. This project automatically classifies lecture segments to help students navigate lecture content efficiently.

## Task

Given a 60-second lecture segment (transcript text + slide frame image), classify it into one of 4 pedagogical categories:

- **Background Context** — supplementary or contextual information
- **Concept Definition** — professor introducing a new idea or definition
- **Worked Example** — professor solving a problem step by step
- **Exam Relevant** — content flagged as important for assessment

## Dataset

- 7,349 labeled lecture segments from 5 university courses
- 4,573 slide frame images extracted from YouTube at 60-second midpoints
- Average 149 words per segment

| Course | Samples |
|---|---|
| MIT 6.034 Artificial Intelligence | 1,409 |
| MIT 6.7960 Deep Learning 2024 | 1,727 |
| Stanford CS231n Computer Vision | 1,137 |
| Andrew NG CS229 Machine Learning | 1,523 |
| MIT 18.06 Linear Algebra | 1,553 |

| Class | Samples |
|---|---|
| Background Context | 5,284 |
| Concept Definition | 854 |
| Worked Example | 667 |
| Exam Relevant | 544 |

## Models and Results

| Model | Accuracy | Macro F1 | AUC-ROC |
|---|---|---|---|
| LR TF-IDF (text baseline) | 0.834 | 0.719 | 0.901 |
| SVM TF-IDF+BERT+CLIP | 0.702 | 0.367 | - |
| DNN Multimodal (PyTorch) | 0.821 | 0.661 | 0.855 |
| DNN Text Only (ablation) | 0.801 | 0.642 | - |
| Ensemble LR+XGB+DNN | 0.874 | 0.762 | 0.930 |

## Key Findings

- TF-IDF outperforms BERT on keyword-driven weakly supervised labels
- Slide images contribute +0.019 Macro F1 overall
- Image modality helps Worked Example most (+0.034 F1)
- Ensemble of LR + XGBoost + DNN achieves best overall performance (0.762 Macro F1)

## Ablation Study

| Class | Text Only F1 | Multimodal F1 | Image Contribution |
|---|---|---|---|
| Background Context | 0.875 | 0.892 | +0.017 |
| Concept Definition | 0.674 | 0.681 | +0.007 |
| Worked Example | 0.626 | 0.660 | +0.034 |
| Exam Relevant | 0.395 | 0.411 | +0.015 |

## Project Structure
```
smart-lecture-classifier/
├── Dataset/
│   ├── static_resources/               # MIT 6.034 raw SRT + PDF files
│   ├── static_resources_fall_2024/     # MIT DL 2024 WebVTT + PDF files
│   ├── build_complete_dataset.py       # Build dataset from MIT OCW + YouTube
│   ├── extract_frames.py               # Extract slide images from YouTube
│   ├── retry_frames.py                 # Retry failed frame extractions
│   ├── extract_embeddings.py           # Extract SBERT + CLIP embeddings
│   ├── train_baselines.py              # Train LR + SVM baselines
│   ├── train_dnn.py                    # Train multimodal DNN + ablation
│   ├── ensemble.py                     # Train LR + XGBoost + DNN ensemble
│   ├── verify_dataset.py               # Verify dataset quality
│   ├── verify_match.py                 # Verify text-image alignment
│   └── dataset_metadata.json           # Dataset statistics and label map
├── results/
│   ├── LR_TF-IDF_only_results.json
│   ├── LR_TF-IDFBERT_results.json
│   ├── LR_TF-IDFBERTCLIP_results.json
│   ├── SVM_TF-IDFBERTCLIP_results.json
│   ├── dnn_results.json
│   ├── ensemble_results.json
│   └── ensemble_xgb_results.json
└── README.md
```

## Setup
```
pip install torch torchvision sentence-transformers open-clip-torch scikit-learn xgboost yt-dlp Pillow tqdm transformers
```

## Pipeline
```
# Step 1 - Build dataset
python Dataset/build_complete_dataset.py

# Step 2 - Extract slide frame images
python Dataset/extract_frames.py

# Step 3 - Extract SBERT + CLIP embeddings
python Dataset/extract_embeddings.py

# Step 4 - Train baseline models
python Dataset/train_baselines.py

# Step 5 - Train multimodal DNN
python Dataset/train_dnn.py

# Step 6 - Run ensemble
python Dataset/ensemble.py
```

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- yt-dlp + ffmpeg for frame extraction
- Node.js for YouTube JavaScript challenge solver

## Course

CS 6140 Machine Learning — Northeastern University, Spring 2026
Professor Ehsan Elhamifar