import os
import re
import json
import glob
import random
from collections import Counter
from youtube_transcript_api import YouTubeTranscriptApi

# CONFIG
MIT_SRT_DIR = r"E:\ML_project\Dataset\static_resources"
MIT_VTT_DIR = r"E:\ML_project\Dataset\static_resources_fall_2024"
OUTPUT_DIR  = r"E:\ML_project\dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset_path  = os.path.join(OUTPUT_DIR, "lecture_dataset.json")
metadata_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")

YOUTUBE_VIDEOS = {
    "Andrew_NG_CS229": [
        "jGwO_UgTS7I", "4b4MUYve_U8", "het9HFqo1TQ",
        "iZTeva0WSTQ", "nt63k3bfXS0", "lDwow4aOrtg",
        "8NYoQiRANpg", "rjbkWSTjHzM", "iVOxMcumR4A",
        "wr9gUr-eWdA", "MfIjxPh6Pys", "zUazLXZZA2U",
        "ORrStCArmP4", "rVfZHWTwXSA", "tw6cmL5STuY",
        "dyb_cFywuik", "YQA9lLdLig8", "d5gaWTo6kDM",
        "0rt2CsEQv6U", "pLhPQynL0tY"
    ],
    "Stanford_CS231n": [
        "vT1JzLTH4G4", "OoUX-nOEjG0", "h7iBpEHGVNc",
        "d14TUNcbn1k", "bNb2fEVKeEo", "wEoyxE0GP2M",
        "_JB0AO7QxSA", "6SlgtELqOWc", "DAOcjicFr1Y",
        "6niqTuYFZLQ", "nDPWywWRIRo", "6wcs6szJWMY",
        "5WoItGTWV54", "lvoHnicueoE", "eZdOkDtYMoo",
        "CIfsB_EYsVI"
    ],
    "MIT_LinearAlgebra": [
        "ZK3O402wf1c", "QVKj3LADCnA", "FX4C-JpTFgY",
        "5hO3MrzPa0A", "JibVXBElKL0", "8o5Cmfpeo6g",
        "VqP2tREMvt0", "9Q1q7s1jTzU", "yjBerM5jWsc",
        "nHlE7EgJFds", "2IdtqGM6KWU", "6-wh6yvk6uc",
        "l88D4r74gtM", "YzZUIYRCE38", "Y_Ac6KiQ1t0",
        "osh80YCg_GM", "srxexLishgY", "23LLB9mNJvc",
        "QNpj-gOXW9M", "lXNXrLcoerU", "13r9QY6cmjc",
        "IZqwi0wJovM", "8MF3pz-oYHo", "sFxA8eIS6tA",
        "umt6BB1nJ4w", "M0Sa8fLOajA", "vF7eyJ2g3kU",
        "z_zYQHmrh08", "Nx0lRBaXoz4", "Ts3o2I8_Mxc",
        "vGkn-3NFGck", "HgC1l_6ySkc", "Go2aLo7ZOlU",
        "RWvi4Vx4CDc"
    ]
}

# LABELING FUNCTION

def label_chunk(text):
    t = text.lower()

    example_kw = [
        "for example", "as an example",
        "here is an example", "here's an example",
        "let me show", "let me demonstrate",
        "let me walk", "let me work through",
        "let me go through", "let's compute",
        "let's solve", "let's work through",
        "let's go through", "let's trace",
        "let's build", "step by step", "walkthrough",
        "consider the following example",
        "suppose we have", "imagine we have"
    ]

    definition_kw = [
        "is defined as", "we define", "formally defined",
        "by definition", "definition of", "is called",
        "we call this", "we say that", "also known as",
        "also called", "is known as", "refers to",
        "we denote", "denoted by", "we write this as",
        "the definition", "formally,", "can be defined",
        "we can define", "defined by", "we now define",
        "we introduce", "what we mean by",
        "what is meant by", "in other words",
        "which means", "that means", "this means",
        "means that", "we use the term", "the term",
        "the concept of", "we call a", "we say a",
        "is what we call", "this is called",
        "think of this as", "we can think of"
    ]

    exam_regex = [
        r'\bexam\b', r'\bmidterm\b',
        r'\bquiz\b', r'\btest\b'
    ]

    exam_plain = [
        "will be on", "you need to know",
        "you should know", "you must know",
        "make sure you know", "don't forget",
        "remember that", "remember this",
        "recall that", "keep in mind",
        "bear in mind", "this is key",
        "key takeaway", "the takeaway",
        "the main takeaway", "to summarize",
        "in summary", "to recap", "let me summarize",
        "bottom line", "the bottom line",
        "key result", "important result",
        "key concept", "key idea", "main idea",
        "main point", "the key insight",
        "a key point", "key observation",
        "important observation", "important concept",
        "important property", "important theorem",
        "important fact", "fundamental concept",
        "fundamental idea", "fundamental theorem",
        "fundamental result", "core idea", "core concept",
        "crucial", "critical point", "very important",
        "especially important", "this is important",
        "this is essential", "worth noting",
        "worth remembering", "i want to emphasize",
        "i want to stress", "let me emphasize",
        "note that", "important to note"
    ]

    # Priority 1 — Worked Example
    if any(k in t for k in example_kw):
        return "Worked Example"
    # Priority 2 — Concept Definition
    if any(k in t for k in definition_kw):
        return "Concept Definition"
    # Priority 3 — Exam Relevant 
    if any(re.search(p, t) for p in exam_regex):
        return "Exam Relevant"
    if any(k in t for k in exam_plain):
        return "Exam Relevant"
    # Default
    return "Background Context"


# CLEAN TEXT

def clean_text(text):
    artifacts = [
        r'\[squeaking\]', r'\[rustling\]', r'\[clicking\]',
        r'\[music\]', r'\[applause\]', r'\[laughter\]',
        r'\[inaudible\]', r'\[crosstalk\]', r'\[silence\]',
        r'\[noise\]', r'\[cheering\]', r'\[clapping\]',
        r'\[background noise\]', r'\[music playing\]',
        r'\[audience\]', r'\[speaking\]'
    ]
    for a in artifacts:
        text = re.sub(a, '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z][A-Z\s]+:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# MERGE INTO 60-SECOND CHUNKS

def merge_into_chunks(entries, chunk_duration=60):
    chunks        = []
    current_text  = []
    current_start = None
    current_end   = None

    for entry in entries:
        start    = entry["start"]
        duration = entry.get("duration", 5)
        text     = entry["text"].strip()
        if not text:
            continue
        if current_start is None:
            current_start = start
        current_text.append(text)
        current_end = start + duration
        if (current_end - current_start) >= chunk_duration:
            merged = clean_text(" ".join(current_text))
            if len(merged.split()) >= 10:
                chunks.append({
                    "start": round(current_start, 2),
                    "end":   round(current_end,   2),
                    "text":  merged
                })
            current_text  = []
            current_start = None

    if current_text and current_start is not None:
        merged = clean_text(" ".join(current_text))
        if len(merged.split()) >= 10:
            chunks.append({
                "start": round(current_start, 2),
                "end":   round(current_end,   2),
                "text":  merged
            })
    return chunks


# PARSE SRT FILES 

def parse_srt(filepath):
    segments = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        match = re.match(
            r"(\d+):(\d+):(\d+),(\d+)\s-->\s(\d+):(\d+):(\d+),(\d+)",
            lines[1]
        )
        if not match:
            continue
        h1,m1,s1,ms1,h2,m2,s2,ms2 = map(int, match.groups())
        start    = h1*3600 + m1*60 + s1 + ms1/1000
        end      = h2*3600 + m2*60 + s2 + ms2/1000
        text     = " ".join(lines[2:]).strip()
        segments.append({
            "start":    start,
            "duration": end - start,
            "text":     text
        })
    return segments


# PARSE WEBVTT FILES 

def parse_webvtt(filepath):
    segments = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    content = re.sub(r'^WEBVTT.*?\n', '', content, flags=re.MULTILINE)
    blocks  = content.strip().split("\n\n")

    for block in blocks:
        lines      = block.strip().split("\n")
        time_line  = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                time_line = line
            elif time_line is not None and line.strip():
                text_lines.append(line.strip())

        if not time_line or not text_lines:
            continue

        match = re.match(
            r"(\d+):(\d+):(\d+)\.(\d+)\s-->\s(\d+):(\d+):(\d+)\.(\d+)",
            time_line.strip()
        )
        if not match:
            continue

        h1,m1,s1,ms1,h2,m2,s2,ms2 = map(int, match.groups())
        start = h1*3600 + m1*60 + s1 + ms1/1000
        end   = h2*3600 + m2*60 + s2 + ms2/1000
        text  = " ".join(text_lines).strip()

        if text:
            segments.append({
                "start":    start,
                "duration": end - start,
                "text":     text
            })
    return segments


# SAVE HELPER

def save_dataset(data):
    counts  = Counter(s["label"] for s in data)
    total   = len(data)
    n_class = len(counts)
    weights = {
        label: round(total / (n_class * count), 4)
        for label, count in counts.items()
    }
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    metadata = {
        "total_samples": total,
        "num_classes":   n_class,
        "label_counts":  dict(counts),
        "class_weights": weights,
        "sources": {
            "MIT_OCW": sum(1 for s in data if s["source"] == "MIT_OCW"),
            "YouTube": sum(1 for s in data if s["source"] == "YouTube")
        },
        "label_map": {
            "Background Context": 0,
            "Concept Definition": 1,
            "Worked Example":     2,
            "Exam Relevant":      3
        },
        "courses": dict(Counter(s["course"] for s in data))
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return counts, weights


# PRINT SUMMARY HELPER

def print_summary(data, counts, weights):
    print(f"\nTotal samples   : {len(data)}")
    print(f"\nLabel distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label:25s}: {count}")
    print(f"\nClass weights:")
    for label, w in sorted(weights.items()):
        print(f"  {label:25s}: {w}")
    print(f"\nFiles:")
    print(f"  {dataset_path}")
    print(f"  {metadata_path}")
    all_ok = all(c >= 500 for c in counts.values())
    if all_ok:
        print(f"\nAll classes above 500")
    else:
        low = {l: c for l, c in counts.items() if c < 500}
        print(f"\nWARNING: Classes below 500: {low}")


# FIX EXAM/EXAMPLE BUG

def fix_exam_example_bug(data):
    fixed = 0
    for s in data:
        t = s["text"].lower()
        if s["label"] == "Exam Relevant":
            has_real_exam = bool(re.search(r'\bexam\b', t))
            has_example   = "example" in t
            if has_example and not has_real_exam:
                s["label"] = "Worked Example"
                fixed += 1
    if fixed > 0:
        print(f"Fixed {fixed} exam/example mislabeled samples")
    return data


# DATASET EXISTS — LOAD, FIX, AND EXIT

if os.path.exists(dataset_path):
    print("=" * 60)
    print("Dataset already exists")
    print("=" * 60)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    courses = set(s["course"] for s in data)

    # Add Fall 2024 data if not already included
    if "MIT_DeepLearning_2024" not in courses:
        print("\nFall 2024 data not found adding now")
        vtt_files   = glob.glob(os.path.join(MIT_VTT_DIR, "*.webvtt"))
        new_samples = []
        print(f"Found {len(vtt_files)} WebVTT files")
        for i, vtt_path in enumerate(vtt_files):
            filename = os.path.basename(vtt_path)
            file_id  = filename.replace("_transcript.webvtt", "")
            entries  = parse_webvtt(vtt_path)
            if not entries:
                continue
            chunks = merge_into_chunks(entries)
            for j, chunk in enumerate(chunks):
                new_samples.append({
                    "video_id":    file_id,
                    "course":      "MIT_DeepLearning_2024",
                    "chunk_index": j,
                    "start":       chunk["start"],
                    "end":         chunk["end"],
                    "text":        chunk["text"],
                    "label":       label_chunk(chunk["text"]),
                    "source":      "MIT_OCW",
                    "has_pdf":     False,
                    "pdf_path":    ""
                })
            print(f"  [{i+1}/{len(vtt_files)}] {file_id[:30]} — {len(chunks)} chunks")
        data = data + new_samples
        print(f"Added {len(new_samples)} samples from Fall 2024")

    # Fix exam/example bug
    data = fix_exam_example_bug(data)

    # Remove duplicates
    seen    = set()
    deduped = []
    for s in data:
        if s["text"] not in seen:
            seen.add(s["text"])
            deduped.append(s)
    removed = len(data) - len(deduped)
    if removed > 0:
        print(f"Removed {removed} duplicates")

    # Boost Exam Relevant if below 500
    counts = Counter(s["label"] for s in deduped)
    if counts.get("Exam Relevant", 0) < 500:
        print(f"\nExam Relevant below 500 — boosting with strong signals")
        STRONG_EXAM = [
            r'\bexam\b', r'\bmidterm\b', r'\bquiz\b',
            r'\bfinal exam\b', r'\bon the exam\b',
            r'\bfor the exam\b', r'\bwill be tested\b'
        ]
        promoted = 0
        for s in deduped:
            if s["label"] != "Background Context":
                continue
            t = s["text"].lower()
            if any(re.search(p, t) for p in STRONG_EXAM):
                s["label"] = "Exam Relevant"
                promoted += 1
        print(f"Promoted {promoted} samples")

    # Save and print
    counts, weights = save_dataset(deduped)
    print_summary(deduped, counts, weights)

    # Verify bug is gone
    bug = sum(
        1 for s in deduped
        if s["label"] == "Exam Relevant"
        and "example" in s["text"].lower()
        and not re.search(r'\bexam\b', s["text"].lower())
    )
    print(f"\nExam/example bug count: {bug}  (must be 0)")
    print(f"\nDataset is ready for model training")
    print("(Delete lecture_dataset.json to rebuild from scratch)")
    exit()


# STEP 1 — MIT 6.034 SRT FILES


print("STEP 1: Loading MIT 6.034 SRT transcripts")


srt_files   = glob.glob(os.path.join(MIT_SRT_DIR, "*.srt"))
pdf_files   = glob.glob(os.path.join(MIT_SRT_DIR, "*.pdf"))
mit_samples = []
print(f"Found {len(srt_files)} SRT files")

for i, srt_path in enumerate(srt_files):
    filename = os.path.basename(srt_path)
    video_id = filename.split("_", 1)[1].replace(".srt", "")
    entries  = parse_srt(srt_path)
    if not entries:
        continue
    chunks = merge_into_chunks(entries)
    matching_pdf = next((p for p in pdf_files if video_id in p), None)
    for j, chunk in enumerate(chunks):
        mit_samples.append({
            "video_id":    video_id,
            "course":      "MIT_6034_AI",
            "chunk_index": j,
            "start":       chunk["start"],
            "end":         chunk["end"],
            "text":        chunk["text"],
            "label":       label_chunk(chunk["text"]),
            "source":      "MIT_OCW",
            "has_pdf":     matching_pdf is not None,
            "pdf_path":    matching_pdf or ""
        })
    print(f"  [{i+1}/{len(srt_files)}] {video_id} — {len(chunks)} chunks")

print(f"\nMIT 6.034 samples: {len(mit_samples)}")


# STEP 2 — MIT 6.7960 DEEP LEARNING 2024 WEBVTT FILES

print("STEP 2: Loading MIT Deep Learning 2024 transcripts")


vtt_files    = glob.glob(os.path.join(MIT_VTT_DIR, "*.webvtt"))
dl24_samples = []
print(f"Found {len(vtt_files)} WebVTT files")

for i, vtt_path in enumerate(vtt_files):
    filename = os.path.basename(vtt_path)
    file_id  = filename.replace("_transcript.webvtt", "")
    entries  = parse_webvtt(vtt_path)
    if not entries:
        continue
    chunks = merge_into_chunks(entries)
    for j, chunk in enumerate(chunks):
        dl24_samples.append({
            "video_id":    file_id,
            "course":      "MIT_DeepLearning_2024",
            "chunk_index": j,
            "start":       chunk["start"],
            "end":         chunk["end"],
            "text":        chunk["text"],
            "label":       label_chunk(chunk["text"]),
            "source":      "MIT_OCW",
            "has_pdf":     False,
            "pdf_path":    ""
        })
    print(f"  [{i+1}/{len(vtt_files)}] {file_id[:30]} — {len(chunks)} chunks")

print(f"\nMIT DL 2024 samples: {len(dl24_samples)}")


# STEP 3 — YOUTUBE TRANSCRIPTS

print("STEP 3: Downloading YouTube transcripts")

ytt_api         = YouTubeTranscriptApi()
youtube_samples = []
total_videos    = sum(len(v) for v in YOUTUBE_VIDEOS.values())
processed       = 0
failed          = []

for course_name, video_ids in YOUTUBE_VIDEOS.items():
    print(f"\n  -- {course_name} --")
    for video_id in video_ids:
        processed += 1
        print(f"  [{processed}/{total_videos}] {video_id}", end=" ... ")
        try:
            fetched  = ytt_api.fetch(video_id, languages=["en"])
            raw_data = fetched.to_raw_data()
            chunks   = merge_into_chunks(raw_data)
            for j, chunk in enumerate(chunks):
                youtube_samples.append({
                    "video_id":    video_id,
                    "course":      course_name,
                    "chunk_index": j,
                    "start":       chunk["start"],
                    "end":         chunk["end"],
                    "text":        chunk["text"],
                    "label":       label_chunk(chunk["text"]),
                    "source":      "YouTube",
                    "has_pdf":     False,
                    "pdf_path":    ""
                })
            print(f"{len(chunks)} chunks")
        except Exception as e:
            print(f"FAILED — {str(e).split(chr(10))[0]}")
            failed.append(video_id)

print(f"\nYouTube samples : {len(youtube_samples)}")
print(f"Videos failed   : {len(failed)}")


# STEP 4 — COMBINE AND CLEAN


print("STEP 4: Combining and cleaning")


all_samples = mit_samples + dl24_samples + youtube_samples
random.seed(42)
random.shuffle(all_samples)

# Fix exam/example bug
all_samples = fix_exam_example_bug(all_samples)

# Remove duplicates
seen    = set()
deduped = []
for s in all_samples:
    if s["text"] not in seen:
        seen.add(s["text"])
        deduped.append(s)

print(f"Total before dedup : {len(all_samples)}")
print(f"Duplicates removed : {len(all_samples) - len(deduped)}")
print(f"Total after dedup  : {len(deduped)}")

# Boost Exam Relevant if below 500
counts = Counter(s["label"] for s in deduped)
if counts.get("Exam Relevant", 0) < 500:
    print(f"\nExam Relevant below 500 — boosting")
    STRONG_EXAM = [
        r'\bexam\b', r'\bmidterm\b', r'\bquiz\b',
        r'\bfinal exam\b', r'\bon the exam\b',
        r'\bfor the exam\b', r'\bwill be tested\b'
    ]
    promoted = 0
    for s in deduped:
        if s["label"] != "Background Context":
            continue
        t = s["text"].lower()
        if any(re.search(p, t) for p in STRONG_EXAM):
            s["label"] = "Exam Relevant"
            promoted += 1
    print(f"Promoted {promoted} samples")


# STEP 5 — SAVE AND SUMMARIZE

counts, weights = save_dataset(deduped)

print("\n" + "=" * 60)
print("DATASET BUILD COMPLETE")
print("=" * 60)
print(f"MIT 6.034       : {len(mit_samples)}")
print(f"MIT DL 2024     : {len(dl24_samples)}")
print(f"YouTube         : {len(youtube_samples)}")
print_summary(deduped, counts, weights)

bug = sum(
    1 for s in deduped
    if s["label"] == "Exam Relevant"
    and "example" in s["text"].lower()
    and not re.search(r'\bexam\b', s["text"].lower())
)
print(f"\nExam/example bug count: {bug}  (must be 0)")
print(f"\nDataset is ready for model training")
