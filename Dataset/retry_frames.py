import os
import json
import subprocess
import time
import random
from collections import defaultdict

DATASET_PATH = r"E:\ML_project\dataset\lecture_dataset.json"
FRAMES_DIR   = r"E:\ML_project\frames"
NODE_PATH    = r"C:\Program Files\nodejs\node.exe"
COOKIES_PATH = r"E:\ML_project\cookies.txt"
os.makedirs(FRAMES_DIR, exist_ok=True)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Get only failed YouTube samples
failed_videos = defaultdict(list)
for i, s in enumerate(data):
    if s.get("has_image") == False and s.get("source") == "YouTube":
        failed_videos[s["video_id"]].append(i)

print(f"Failed videos to retry : {len(failed_videos)}")
print(f"Failed samples         : {sum(len(v) for v in failed_videos.values())}")

def get_stream_url(video_id):
    try:
        cmd = [
            "python", "-m", "yt_dlp",
            "--cookies", COOKIES_PATH,
            "--js-runtimes", f"node:{NODE_PATH}",
            "--remote-components", "ejs:github",
            "--format", "best[ext=mp4]/best",
            "--get-url", "--quiet",
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception as e:
        print(f"    Error: {e}")
    return None

def extract_frame(stream_url, timestamp, output_path):
    try:
        cmd = [
            "ffmpeg", "-ss", str(timestamp),
            "-i", stream_url,
            "-frames:v", "1", "-q:v", "2", "-y",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return os.path.exists(output_path)
    except:
        return False

total        = len(failed_videos)
success      = 0
still_failed = []

for vid_num, (video_id, indices) in enumerate(failed_videos.items()):
    print(f"\n[{vid_num+1}/{total}] Video: {video_id}  ({len(indices)} chunks)")

    print(f"  Getting stream URL...", end=" ", flush=True)
    stream_url = get_stream_url(video_id)

    if not stream_url:
        print(f"FAILED")
        still_failed.append(video_id)
        continue

    print(f"OK")

    chunk_success = 0
    for idx in indices:
        s        = data[idx]
        midpoint = (s["start"] + s["end"]) / 2
        fname    = f"{video_id}_chunk_{s['chunk_index']:04d}.jpg"
        fpath    = os.path.join(FRAMES_DIR, fname)

        if os.path.exists(fpath):
            data[idx]["frame_path"] = fpath
            data[idx]["has_image"]  = True
            chunk_success += 1
            continue

        ok = extract_frame(stream_url, midpoint, fpath)
        if ok:
            data[idx]["frame_path"] = fpath
            data[idx]["has_image"]  = True
            chunk_success += 1
        else:
            data[idx]["frame_path"] = ""
            data[idx]["has_image"]  = False

    success += chunk_success
    print(f"  Chunks done: {chunk_success}/{len(indices)}")

    # Save after every video
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Final save
with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"RETRY COMPLETE")
print(f"{'='*60}")
print(f"Newly recovered  : {success}")
print(f"Still failed     : {len(still_failed)}")
if still_failed:
    print(f"Still failed IDs : {still_failed}")

has_image = sum(1 for s in data if s.get("has_image"))
no_image  = sum(1 for s in data if not s.get("has_image"))
print(f"\nDataset coverage:")
print(f"  With image     : {has_image} ({100*has_image//len(data)}%)")
print(f"  Without image  : {no_image}  ({100*no_image//len(data)}%)")