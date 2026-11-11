"""
Verify the re-split dataset integrity.
Checks: pairing, class distribution, leakage, augmentation grouping.
"""
import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

SPLIT_ROOT = Path(r"C:\Users\Asus\Downloads\rm task\Emergency_Vehicle_Project\data\split_dataset")
CLASS_NAMES = {0:'car',1:'motorcycle',2:'autorickshaw',3:'truck',4:'bus',5:'bicycle',6:'ambulance',7:'police_car'}
EMERGENCY = {6, 7}

def extract_group_id(filename):
    m = re.match(r'^(Image_\d+)', Path(filename).stem)
    return m.group(1) if m else Path(filename).stem

def md5(path):
    h = hashlib.md5()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_classes(label_path):
    classes = set()
    with open(label_path,'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try: classes.add(int(line.split()[0]))
                except: pass
    return classes

def count_objects(label_path):
    counts = Counter()
    with open(label_path,'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try: counts[int(line.split()[0])] += 1
                except: pass
    return counts

def validate_annotation(label_path):
    """Returns (total_lines, errors_list)."""
    errors = []
    total = 0
    with open(label_path,'r') as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line: continue
            total += 1
            tokens = line.split()
            if len(tokens) != 5:
                errors.append(f"Line {i}: {len(tokens)} tokens")
                continue
            try:
                cid = int(tokens[0])
                if cid < 0 or cid > 7:
                    errors.append(f"Line {i}: class {cid} out of range")
            except ValueError:
                errors.append(f"Line {i}: non-int class '{tokens[0]}'")
                continue
            try:
                xc, yc, w, h = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
                if w <= 0 or h <= 0:
                    errors.append(f"Line {i}: zero/negative dimension")
            except ValueError:
                errors.append(f"Line {i}: non-float coords")
    return total, errors

print("=" * 70)
print("  DATASET RE-SPLIT VERIFICATION")
print("=" * 70)

splits = ['train', 'val', 'test']
split_data = {}

# 1. Pairing check
print("\n1. IMAGE-LABEL PAIRING")
print("-" * 40)
for s in splits:
    img_dir = SPLIT_ROOT / s / 'images'
    lbl_dir = SPLIT_ROOT / s / 'labels'
    imgs = {p.stem: p for p in img_dir.glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}}
    lbls = {p.stem: p for p in lbl_dir.glob('*.txt')}
    orphan_imgs = set(imgs) - set(lbls)
    orphan_lbls = set(lbls) - set(imgs)
    split_data[s] = {'imgs': imgs, 'lbls': lbls}
    status = "[PASS]" if len(orphan_imgs) == 0 and len(orphan_lbls) == 0 else "[FAIL]"
    print(f"  {s:6s}: {len(imgs)} images, {len(lbls)} labels, orphan_img={len(orphan_imgs)}, orphan_lbl={len(orphan_lbls)} {status}")

# 2. Annotation validation
print("\n2. ANNOTATION VALIDATION")
print("-" * 40)
total_errors_all = 0
for s in splits:
    total_boxes = 0
    total_errors = 0
    empty_files = 0
    for stem, lbl in split_data[s]['lbls'].items():
        n, errs = validate_annotation(lbl)
        total_boxes += n
        total_errors += len(errs)
        if n == 0: empty_files += 1
    total_errors_all += total_errors
    status = "[PASS]" if total_errors == 0 else "[FAIL]"
    print(f"  {s:6s}: {total_boxes} boxes, {empty_files} empty files, {total_errors} errors {status}")

# 3. Class distribution
print("\n3. CLASS DISTRIBUTION")
print("-" * 40)
print(f"  {'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print("  " + "-" * 50)
split_obj_counts = {}
for s in splits:
    counts = Counter()
    for stem, lbl in split_data[s]['lbls'].items():
        counts += count_objects(lbl)
    split_obj_counts[s] = counts
for cid in range(8):
    name = CLASS_NAMES[cid]
    tr = split_obj_counts['train'].get(cid, 0)
    va = split_obj_counts['val'].get(cid, 0)
    te = split_obj_counts['test'].get(cid, 0)
    tag = " ***" if cid in EMERGENCY else ""
    print(f"  {name:<15} {tr:>8} {va:>8} {te:>8} {tr+va+te:>8}{tag}")
tr_total = sum(split_obj_counts['train'].values())
va_total = sum(split_obj_counts['val'].values())
te_total = sum(split_obj_counts['test'].values())
print("  " + "-" * 50)
print(f"  {'TOTAL':<15} {tr_total:>8} {va_total:>8} {te_total:>8} {tr_total+va_total+te_total:>8}")

# 4. Emergency image counts
print("\n4. EMERGENCY IMAGE COUNTS")
print("-" * 40)
for s in splits:
    amb_imgs = 0; pol_imgs = 0
    for stem, lbl in split_data[s]['lbls'].items():
        classes = get_classes(lbl)
        if 6 in classes: amb_imgs += 1
        if 7 in classes: pol_imgs += 1
    print(f"  {s:6s}: ambulance={amb_imgs} images, police_car={pol_imgs} images")

# 5. Cross-split leakage (filename overlap)
print("\n5. CROSS-SPLIT FILENAME LEAKAGE")
print("-" * 40)
for a, b in [('train','val'), ('train','test'), ('val','test')]:
    overlap = set(split_data[a]['imgs'].keys()) & set(split_data[b]['imgs'].keys())
    status = "[PASS]" if len(overlap) == 0 else f"[FAIL] {len(overlap)} shared"
    print(f"  {a}-{b}: {status}")

# 6. Group-level leakage (augmentation grouping)
print("\n6. AUGMENTATION GROUP LEAKAGE")
print("-" * 40)
split_groups = {}
for s in splits:
    groups = defaultdict(list)
    for stem in split_data[s]['imgs']:
        gid = extract_group_id(stem)
        groups[gid].append(stem)
    split_groups[s] = groups

for a, b in [('train','val'), ('train','test'), ('val','test')]:
    shared_groups = set(split_groups[a].keys()) & set(split_groups[b].keys())
    status = "[PASS]" if len(shared_groups) == 0 else f"[FAIL] {len(shared_groups)} shared groups"
    print(f"  {a}-{b}: {status}")
    if shared_groups:
        for gid in list(shared_groups)[:5]:
            print(f"    Group '{gid}': {a}={split_groups[a][gid][:2]}, {b}={split_groups[b][gid][:2]}")

# 7. MD5 cross-split check (sample-based for speed)
print("\n7. HASH-BASED DUPLICATE CHECK (FULL)")
print("-" * 40)
all_hashes = {}
dupes_within = Counter()
dupes_cross = []
for s in splits:
    for stem, img_path in split_data[s]['imgs'].items():
        h = md5(img_path)
        if h in all_hashes:
            prev_split, prev_stem = all_hashes[h]
            if prev_split == s:
                dupes_within[s] += 1
            else:
                dupes_cross.append((prev_split, prev_stem, s, stem))
        else:
            all_hashes[h] = (s, stem)

for s in splits:
    print(f"  Duplicates within {s}: {dupes_within[s]}")
print(f"  Cross-split duplicates: {len(dupes_cross)} {'[PASS]' if len(dupes_cross)==0 else '[FAIL]'}")
if dupes_cross:
    for ps, pn, cs, cn in dupes_cross[:5]:
        print(f"    {ps}/{pn} == {cs}/{cn}")

# 8. Summary
print("\n" + "=" * 70)
print("  VERIFICATION SUMMARY")
print("=" * 70)
total_imgs = sum(len(split_data[s]['imgs']) for s in splits)
print(f"  Total images: {total_imgs}")
print(f"  Train: {len(split_data['train']['imgs'])} ({len(split_data['train']['imgs'])/total_imgs*100:.1f}%)")
print(f"  Val:   {len(split_data['val']['imgs'])} ({len(split_data['val']['imgs'])/total_imgs*100:.1f}%)")
print(f"  Test:  {len(split_data['test']['imgs'])} ({len(split_data['test']['imgs'])/total_imgs*100:.1f}%)")
print(f"  Annotation errors: {total_errors_all}")
print(f"  Cross-split duplicates: {len(dupes_cross)}")
print("=" * 70)
