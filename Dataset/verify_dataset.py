import json
import os
from collections import Counter

data = json.load(open('E:/ML_project/dataset/lecture_dataset.json'))

print('DATASET VERIFICATION')

print(f'Total samples         : {len(data)}')

fields = ['video_id', 'course', 'chunk_index', 'start', 'end', 'text', 'label', 'source', 'has_image', 'frame_path']
for field in fields:
    missing = sum(1 for s in data if field not in s)
    print(f'Missing {field:15s} : {missing}')

print('Label distribution:')
counts = Counter(s['label'] for s in data)
for label, count in sorted(counts.items()):
    print(f'  {label:25s}: {count}')

print('Course distribution:')
courses = Counter(s['course'] for s in data)
for course, count in sorted(courses.items()):
    print(f'  {course:30s}: {count}')

has_image = sum(1 for s in data if s.get('has_image') == True)
no_image  = sum(1 for s in data if s.get('has_image') == False)
print(f'Has image             : {has_image}')
print(f'No image              : {no_image}')

missing_files = sum(1 for s in data if s.get('has_image') == True and not os.path.exists(s.get('frame_path','')))
print(f'Missing frame files   : {missing_files}')

short = sum(1 for s in data if len(s['text'].split()) < 10)
dupes = len(data) - len(set(s['text'] for s in data))
print(f'Short samples         : {short}')
print(f'Exact duplicates      : {dupes}')

by_video = {}
for s in data:
    vid = s['video_id']
    if vid not in by_video:
        by_video[vid] = []
    by_video[vid].append(s['chunk_index'])
gap_videos = sum(1 for chunks in by_video.values() if sorted(chunks) != list(range(len(chunks))))
print(f'Videos with gaps      : {gap_videos}')

all_ok = all(c >= 500 for c in counts.values())
print(f'All classes above 500 : {all_ok}')

issues = []
if short > 0: issues.append(f'{short} short samples')
if dupes > 0: issues.append(f'{dupes} duplicates')
if missing_files > 0: issues.append(f'{missing_files} missing frame files')
if gap_videos > 0: issues.append(f'{gap_videos} videos with chunk gaps')
if not all_ok: issues.append('Some classes below 500')
if not issues:
    print('VERDICT: dataset ready for embedding extraction')
else:
    print('VERDICT: Issues found:')
    for issue in issues:
        print(f'  - {issue}')
