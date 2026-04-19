import os
import shutil
import json
import subprocess
import time
from collections import defaultdict


# CONFIG

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH  = os.path.join(BASE_DIR, "Dataset", "lecture_dataset.json")
FRAMES_DIR    = os.path.join(BASE_DIR, "frames")
NODE_PATH     = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
YTDLP         = "python -m yt_dlp"
os.makedirs(FRAMES_DIR, exist_ok=True)


# LOAD DATASET

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Only process samples with real YouTube video IDs
# Skip MIT DL 2024 (Google Drive IDs, not YouTube)
youtube_courses = {"MIT_6034_AI", "Andrew_NG_CS229", "Stanford_CS231n", "MIT_LinearAlgebra"}
yt_samples = [s for s in data if s["course"] in youtube_courses]
dl_samples = [s for s in data if s["course"] == "MIT_DeepLearning_2024"]

print(f"Total samples        : {len(data)}")
print(f"YouTube samples      : {len(yt_samples)}  (will extract frames)")
print(f"MIT DL 2024 samples  : {len(dl_samples)}  (zero embedding fallback)")


# FRAME EXTRACTION FUNCTION

def get_stream_url(video_id):
    """Get direct video stream URL using yt-dlp."""
    cmd = [
        "python", "-m", "yt_dlp",
        "--js-runtimes", f"node:{NODE_PATH}",
        "--remote-components", "ejs:github",
        "--format", "best[ext=mp4]/best",
        "--get-url",
        "--quiet",
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # Return first URL (may return multiple lines)
    return result.stdout.strip().split("\n")[0]

def extract_frame(stream_url, timestamp, output_path):
    """Extract one frame at timestamp using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", stream_url,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return os.path.exists(output_path)


# GROUP SAMPLES BY VIDEO ID
# (get stream URL once per video, use for all chunks)

video_chunks = defaultdict(list)
for i, s in enumerate(data):
    if s["course"] in youtube_courses:
        video_chunks[s["video_id"]].append(i)

print(f"\nUnique videos to process: {len(video_chunks)}")
print(f"Frames directory: {FRAMES_DIR}")


# MAIN EXTRACTION LOOP

total_videos    = len(video_chunks)
success_frames  = 0
failed_frames   = 0
failed_videos   = []
video_num       = 0

for video_id, sample_indices in video_chunks.items():
    video_num += 1
    print(f"\n[{video_num}/{total_videos}] Video: {video_id}  ({len(sample_indices)} chunks)")

    # Get stream URL once for this video
    print(f"  Getting stream URL", end=" ", flush=True)
    try:
        stream_url = get_stream_url(video_id)
    except Exception as e:
        print(f"FAILED — {e}")
        failed_videos.append(video_id)
        failed_frames += len(sample_indices)
        # Mark all chunks of this video as no image
        for idx in sample_indices:
            data[idx]["frame_path"] = ""
            data[idx]["has_image"]  = False
        continue

    if not stream_url:
        print(f"FAILED no URL returned")
        failed_videos.append(video_id)
        failed_frames += len(sample_indices)
        for idx in sample_indices:
            data[idx]["frame_path"] = ""
            data[idx]["has_image"]  = False
        continue

    print(f"OK")

    # Extract one frame per chunk
    for idx in sample_indices:
        s         = data[idx]
        midpoint  = (s["start"] + s["end"]) / 2
        fname     = f"{video_id}_chunk_{s['chunk_index']:04d}.jpg"
        fpath     = os.path.join(FRAMES_DIR, fname)

        # Skip if already extracted
        if os.path.exists(fpath):
            data[idx]["frame_path"] = fpath
            data[idx]["has_image"]  = True
            success_frames += 1
            continue

        try:
            ok = extract_frame(stream_url, midpoint, fpath)
            if ok:
                data[idx]["frame_path"] = fpath
                data[idx]["has_image"]  = True
                success_frames += 1
            else:
                data[idx]["frame_path"] = ""
                data[idx]["has_image"]  = False
                failed_frames += 1
        except Exception as e:
            data[idx]["frame_path"] = ""
            data[idx]["has_image"]  = False
            failed_frames += 1

    print(f"  Chunks done: {len(sample_indices)}  |  Total so far: {success_frames} ok, {failed_frames} failed")

    # Save progress every video in case of interruption
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# HANDLE MIT DL 2024 — zero image fallback

for i, s in enumerate(data):
    if s["course"] == "MIT_DeepLearning_2024":
        if "has_image" not in s:
            data[i]["frame_path"] = ""
            data[i]["has_image"]  = False


# FINAL SAVE AND SUMMARY

with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("FRAME EXTRACTION COMPLETE")
print(f"Frames extracted     : {success_frames}")
print(f"Frames failed        : {failed_frames}")
print(f"MIT DL 2024 fallback : {len(dl_samples)}")
print(f"Failed videos        : {failed_videos if failed_videos else 'None'}")

has_image = sum(1 for s in data if s.get("has_image"))
no_image  = sum(1 for s in data if not s.get("has_image"))
print(f"\nDataset coverage:")
print(f"  With image         : {has_image} ({100*has_image//len(data)}%)")
print(f"  Without image      : {no_image}  ({100*no_image//len(data)}%)")
print(f"\nFrames saved to: {FRAMES_DIR}")
print(f"Dataset updated: {DATASET_PATH}")