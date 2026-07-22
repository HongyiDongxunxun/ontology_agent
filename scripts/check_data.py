"""Check gold_flat and input data statistics."""
import os, json, sys
from collections import defaultdict

gold_dir = 'D:/code/AcademicEvaluation_exam/baseline/gold_flat'
total_lines = 0
total_files = 0
unique_sentences = set()

for f in sorted(os.listdir(gold_dir)):
    if f.endswith('.jsonl'):
        total_files += 1
        with open(os.path.join(gold_dir, f), 'r', encoding='utf-8') as fh:
            for line in fh:
                d = json.loads(line.strip())
                total_lines += 1
                key = (d.get('source_file', ''), d.get('sentence_id', ''))
                unique_sentences.add(key)

print(f'Total gold_flat files: {total_files}')
print(f'Total gold_flat lines (entities): {total_lines}')
print(f'Unique (source_file, sentence_id) pairs: {len(unique_sentences)}')
print(f'Avg entities per sentence: {total_lines/len(unique_sentences):.1f}')
