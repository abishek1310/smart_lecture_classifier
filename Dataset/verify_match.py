import json, os, random
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.join(BASE_DIR, "Dataset", "lecture_dataset.json")))
samples = random.sample([s for s in data if s.get('has_image')], 5)
for s in samples:
    frame_file = os.path.basename(s['frame_path'])
    vid_in_frame = frame_file.split('_chunk_')[0]
    vid_in_text  = s['video_id']
    match = 'OK' if vid_in_frame == vid_in_text else 'MISMATCH'
    print(match, 'text=', vid_in_text, 'frame=', vid_in_frame, 'chunk=', s['chunk_index'], 'start=', s['start'])
