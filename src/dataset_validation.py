"""
Dataset Quality Inspection and Validation Script for YOLO Object Detection
Dataset: C:\\Users\\Asus\\Downloads\\rm task\\combined_dataset
Classes:
  0: car
  1: motorcycle
  2: autorickshaw
  3: truck
  4: bus
  5: bicycle
  6: ambulance
  7: police_car
"""

import os
import sys
import re
import glob
import hashlib
import random
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse

# Graceful Pillow import with binary magic-byte fallback
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

CLASS_NAMES = {
    0: 'car',
    1: 'motorcycle',
    2: 'autorickshaw',
    3: 'truck',
    4: 'bus',
    5: 'bicycle',
    6: 'ambulance',
    7: 'police_car'
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def compute_file_md5(filepath, chunk_size=65536):
    """Compute MD5 hash of a file efficiently."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return filepath, hasher.hexdigest(), None
    except Exception as e:
        return filepath, None, str(e)


def extract_base_name(filename):
    """
    Extract inferred original/base image identifier by stripping common augmentation suffixes.
    Handles Roboflow patterns (e.g. stem_jpg.rf.hash),
    custom augmentation tokens (_aug, _bal, _flip, _rot, _bright, etc.),
    and numeric/variant suffixes.
    """
    stem = Path(filename).stem

    # Pattern 1: Roboflow export pattern (e.g., "name_jpg.rf.abcdef123456...")
    rf_match = re.match(r'^(.*?)(?:_jpg|_png|_jpeg)?\.rf\.[a-f0-9]+$', stem, re.IGNORECASE)
    if rf_match:
        return rf_match.group(1), 'roboflow_rf'

    # Pattern 2: Explicit augmentation tokens
    # e.g., name_bal_1, name_aug2, name_flip_h, name_rot90, name_bright_0.8
    aug_keywords = r'(?:_bal|_aug|_flip|_rot|_bright|_scale|_trans|_noise|_blur|_crop|_zoom|_contrast|_hsv|_sat|_vflip|_hflip|_aff)'
    aug_match = re.search(aug_keywords, stem, re.IGNORECASE)
    if aug_match:
        base = stem[:aug_match.start()]
        if base:
            return base, 'keyword_aug'

    # Pattern 3: Trailing copy or variant markers like name(1), name_1, name_copy
    copy_match = re.match(r'^(.*?)(?:_copy|_v\d+|\(\d+\)|_\d+)$', stem, re.IGNORECASE)
    if copy_match:
        base = copy_match.group(1)
        if base:
            return base, 'numbered_variant'

    return stem, 'original'


def validate_yolo_label_file(label_path, num_classes=8):
    """
    Validate a single YOLO format label file.
    Returns file_stats dict.
    """
    stats = {
        'total_boxes': 0,
        'valid_boxes': 0,
        'invalid_token_count': 0,
        'invalid_class_id': 0,
        'invalid_coord_format': 0,
        'coord_out_of_range': 0,
        'bbox_out_of_image': 0,
        'bbox_severe_out_of_image': 0,
        'classes_in_file': set(),
        'class_counts': Counter(),
        'error_samples': []
    }

    try:
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        stats['error_samples'].append(f"Cannot read file: {e}")
        return stats

    for line_idx, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue

        stats['total_boxes'] += 1
        tokens = line.split()

        if len(tokens) != 5:
            stats['invalid_token_count'] += 1
            if len(stats['error_samples']) < 5:
                stats['error_samples'].append(f"Line {line_idx}: expected 5 tokens, got {len(tokens)} -> '{line}'")
            continue

        # Check class ID
        try:
            class_id = int(tokens[0])
            if class_id < 0 or class_id >= num_classes:
                stats['invalid_class_id'] += 1
                if len(stats['error_samples']) < 5:
                    stats['error_samples'].append(f"Line {line_idx}: invalid class ID {class_id}")
                continue
        except ValueError:
            stats['invalid_class_id'] += 1
            if len(stats['error_samples']) < 5:
                stats['error_samples'].append(f"Line {line_idx}: non-integer class ID '{tokens[0]}'")
            continue

        # Check coordinates
        try:
            xc = float(tokens[1])
            yc = float(tokens[2])
            w = float(tokens[3])
            h = float(tokens[4])
        except ValueError:
            stats['invalid_coord_format'] += 1
            if len(stats['error_samples']) < 5:
                stats['error_samples'].append(f"Line {line_idx}: non-float coords '{tokens[1:]}'")
            continue

        # Check center and dimension range [0, 1]
        coord_error = False
        if not (0.0 <= xc <= 1.0) or not (0.0 <= yc <= 1.0):
            stats['coord_out_of_range'] += 1
            coord_error = True

        if not (0.0 < w <= 1.0) or not (0.0 < h <= 1.0):
            stats['coord_out_of_range'] += 1
            coord_error = True

        # Check bbox corners
        xmin = xc - w / 2.0
        xmax = xc + w / 2.0
        ymin = yc - h / 2.0
        ymax = yc + h / 2.0

        if xmin < 0.0 or xmax > 1.0 or ymin < 0.0 or ymax > 1.0:
            stats['bbox_out_of_image'] += 1
            # Check severe (>5% outside)
            if xmin < -0.05 or xmax > 1.05 or ymin < -0.05 or ymax > 1.05:
                stats['bbox_severe_out_of_image'] += 1

        if not coord_error:
            stats['valid_boxes'] += 1
            stats['classes_in_file'].add(class_id)
            stats['class_counts'][class_id] += 1

    return stats


def test_image_readability(image_paths, sample_size=200, seed=42):
    """
    Test image readability using PIL or binary header inspection fallback.
    """
    if len(image_paths) <= sample_size:
        sample = list(image_paths)
    else:
        rng = random.Random(seed)
        sample = rng.sample(image_paths, sample_size)

    results = {
        'total_tested': len(sample),
        'corrupt': [],
        'truncated': [],
        'formats': Counter(),
        'modes': Counter(),
        'resolutions': []
    }

    for p in sample:
        if HAS_PIL:
            try:
                with Image.open(p) as img:
                    img.verify()
                with Image.open(p) as img:
                    results['formats'][img.format] += 1
                    results['modes'][img.mode] += 1
                    results['resolutions'].append(img.size)
            except Exception as e:
                results['corrupt'].append((str(p), str(e)))
        else:
            # Fallback: check file readability and binary header magic bytes
            try:
                with open(p, 'rb') as f:
                    header = f.read(32)
                if not header:
                    results['corrupt'].append((str(p), "Empty 0-byte file"))
                elif header.startswith(b'\xff\xd8\xff'):
                    results['formats']['JPEG'] += 1
                elif header.startswith(b'\x89PNG\r\n\x1a\n'):
                    results['formats']['PNG'] += 1
                elif header.startswith(b'BM'):
                    results['formats']['BMP'] += 1
                elif header.startswith(b'RIFF') and b'WEBP' in header:
                    results['formats']['WEBP'] += 1
                else:
                    results['formats']['UNKNOWN'] += 1
            except Exception as e:
                results['corrupt'].append((str(p), str(e)))

    return results


def run_dataset_inspection(dataset_dir, output_dir, sample_size=200):
    dataset_path = Path(dataset_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "inspection_report.txt"

    lines = []
    def log(msg=""):
        lines.append(msg)
        print(msg)

    log("=" * 80)
    log("     YOLO DATASET COMPREHENSIVE QUALITY INSPECTION REPORT")
    log("=" * 80)
    log(f"Dataset Location : {dataset_path}")
    log(f"Output Directory : {out_path}")
    log(f"Configured Classes: {CLASS_NAMES}")
    log("=" * 80)
    log()

    # ---------------------------------------------------------
    # 1. IMAGE-LABEL PAIRING
    # ---------------------------------------------------------
    log("1. IMAGE-LABEL PAIRING CHECK")
    log("-" * 40)

    splits = ['train', 'val']
    split_images = {}
    split_labels = {}
    pairing_stats = {}

    for split in splits:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            log(f"ERROR: Image directory not found: {img_dir}")
            return
        if not lbl_dir.exists():
            log(f"ERROR: Label directory not found: {lbl_dir}")
            return

        # Scan images
        images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        # Scan labels
        labels = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == '.txt']

        split_images[split] = images
        split_labels[split] = labels

        img_stems = {p.stem: p for p in images}
        lbl_stems = {p.stem: p for p in labels}

        orphan_images = [p for stem, p in img_stems.items() if stem not in lbl_stems]
        orphan_labels = [p for stem, p in lbl_stems.items() if stem not in img_stems]

        pairing_stats[split] = {
            'images_count': len(images),
            'labels_count': len(labels),
            'orphan_images': orphan_images,
            'orphan_labels': orphan_labels,
            'image_extensions': Counter(p.suffix.lower() for p in images)
        }

        log(f"Split: '{split}'")
        log(f"  Total Images Found : {len(images)}")
        log(f"  Total Labels Found : {len(labels)}")
        log(f"  Image Extensions   : {dict(pairing_stats[split]['image_extensions'])}")
        log(f"  Orphan Images      : {len(orphan_images)}")
        if orphan_images:
            for oi in orphan_images[:10]:
                log(f"    - Orphan Image: {oi.name}")
            if len(orphan_images) > 10:
                log(f"    ... and {len(orphan_images) - 10} more")
        log(f"  Orphan Labels      : {len(orphan_labels)}")
        if orphan_labels:
            for ol in orphan_labels[:10]:
                log(f"    - Orphan Label: {ol.name}")
            if len(orphan_labels) > 10:
                log(f"    ... and {len(orphan_labels) - 10} more")
        log()

    # ---------------------------------------------------------
    # 2. YOLO ANNOTATION VALIDATION
    # ---------------------------------------------------------
    log("2. YOLO ANNOTATION VALIDATION")
    log("-" * 40)

    annotation_stats = {}
    class_object_counts = {split: Counter() for split in splits}
    class_image_counts = {split: Counter() for split in splits}
    empty_label_files = {split: 0 for split in splits}

    for split in splits:
        labels = split_labels[split]
        total_boxes = 0
        valid_boxes = 0
        inv_tokens = 0
        inv_classes = 0
        inv_coords = 0
        out_of_range_coords = 0
        bbox_out_image = 0
        bbox_severe_out_image = 0
        error_samples_all = []

        for lbl in labels:
            res = validate_yolo_label_file(lbl, num_classes=len(CLASS_NAMES))
            total_boxes += res['total_boxes']
            valid_boxes += res['valid_boxes']
            inv_tokens += res['invalid_token_count']
            inv_classes += res['invalid_class_id']
            inv_coords += res['invalid_coord_format']
            out_of_range_coords += res['coord_out_of_range']
            bbox_out_image += res['bbox_out_of_image']
            bbox_severe_out_image += res['bbox_severe_out_of_image']

            if res['total_boxes'] == 0:
                empty_label_files[split] += 1

            for c_id, c_count in res['class_counts'].items():
                class_object_counts[split][c_id] += c_count

            for c_id in res['classes_in_file']:
                class_image_counts[split][c_id] += 1

            if res['error_samples'] and len(error_samples_all) < 15:
                for err in res['error_samples']:
                    error_samples_all.append(f"[{lbl.name}] {err}")

        annotation_stats[split] = {
            'total_boxes': total_boxes,
            'valid_boxes': valid_boxes,
            'inv_tokens': inv_tokens,
            'inv_classes': inv_classes,
            'inv_coords': inv_coords,
            'out_of_range_coords': out_of_range_coords,
            'bbox_out_image': bbox_out_image,
            'bbox_severe_out_image': bbox_severe_out_image,
            'empty_files': empty_label_files[split],
            'errors': error_samples_all
        }

        log(f"Split: '{split}'")
        log(f"  Total Label Files Inspected    : {len(labels)}")
        log(f"  Empty Label Files (Background) : {empty_label_files[split]}")
        log(f"  Total Bounding Boxes           : {total_boxes}")
        log(f"  Valid Bounding Boxes           : {valid_boxes}")
        log(f"  Invalid Token Count (!=5)      : {inv_tokens}")
        log(f"  Invalid Class IDs (<0 or >=8)  : {inv_classes}")
        log(f"  Invalid Coordinate Format      : {inv_coords}")
        log(f"  Center/Dim Out-of-Range [0,1]  : {out_of_range_coords}")
        log(f"  BBox Bounds Exceeding [0,1]    : {bbox_out_image} (Severe >5%: {bbox_severe_out_image})")
        if error_samples_all:
            log(f"  Error Samples (first {len(error_samples_all)}):")
            for err in error_samples_all[:10]:
                log(f"    - {err}")
        log()

    # ---------------------------------------------------------
    # 3. IMAGE READABILITY
    # ---------------------------------------------------------
    log("3. IMAGE READABILITY & INTEGRITY SAMPLING")
    log("-" * 40)

    for split in splits:
        img_paths = split_images[split]
        read_res = test_image_readability(img_paths, sample_size=sample_size, seed=42)
        log(f"Split: '{split}' (Sample Size: {read_res['total_tested']} of {len(img_paths)})")
        log(f"  Corrupt Images Detected : {len(read_res['corrupt'])}")
        if read_res['corrupt']:
            for img_name, err in read_res['corrupt']:
                log(f"    - {img_name}: {err}")
        log(f"  Image Formats Found     : {dict(read_res['formats'])}")
        if read_res['modes']:
            log(f"  Color Modes Found       : {dict(read_res['modes'])}")
        if read_res['resolutions']:
            widths = [w for w, h in read_res['resolutions']]
            heights = [h for w, h in read_res['resolutions']]
            log(f"  Resolution Summary      : min=({min(widths)}x{min(heights)}), max=({max(widths)}x{max(heights)}), avg=({int(sum(widths)/len(widths))}x{int(sum(heights)/len(heights))})")
        log()

    # ---------------------------------------------------------
    # 4. CLASS DISTRIBUTION
    # ---------------------------------------------------------
    log("4. CLASS DISTRIBUTION (OBJECT-LEVEL & IMAGE-LEVEL)")
    log("-" * 40)

    total_train_objs = sum(class_object_counts['train'].values())
    total_val_objs = sum(class_object_counts['val'].values())
    total_train_imgs = len(split_images['train'])
    total_val_imgs = len(split_images['val'])

    log(f"{'Class ID':<8} {'Class Name':<15} {'Train Objs':<12} {'Train %':<10} {'Val Objs':<10} {'Val %':<10} {'Train Imgs':<12} {'Val Imgs':<10}")
    log("-" * 85)

    for c_id in range(len(CLASS_NAMES)):
        c_name = CLASS_NAMES[c_id]
        tr_objs = class_object_counts['train'][c_id]
        tr_pct = (tr_objs / total_train_objs * 100) if total_train_objs else 0
        va_objs = class_object_counts['val'][c_id]
        va_pct = (va_objs / total_val_objs * 100) if total_val_objs else 0
        tr_imgs = class_image_counts['train'][c_id]
        va_imgs = class_image_counts['val'][c_id]

        log(f"{c_id:<8} {c_name:<15} {tr_objs:<12} {tr_pct:<9.2f}% {va_objs:<10} {va_pct:<9.2f}% {tr_imgs:<12} {va_imgs:<10}")

    log("-" * 85)
    log(f"{'TOTAL':<24} {total_train_objs:<12} {'100.0%':<10} {total_val_objs:<10} {'100.0%':<10} {total_train_imgs:<12} {total_val_imgs:<10}")
    log()

    # Calculate class imbalance
    if total_train_objs > 0:
        max_c = max(class_object_counts['train'].items(), key=lambda x: x[1])
        min_c = min([x for x in class_object_counts['train'].items() if x[1] > 0] or [(0, 1)], key=lambda x: x[1])
        ratio = max_c[1] / max(min_c[1], 1)
        log(f"Class Imbalance in Train (Max vs Min): {CLASS_NAMES[max_c[0]]} ({max_c[1]}) vs {CLASS_NAMES[min_c[0]]} ({min_c[1]}) -> Ratio: {ratio:.1f}:1")
    log()

    # ---------------------------------------------------------
    # 5. DUPLICATE DETECTION (MD5 HASHES)
    # ---------------------------------------------------------
    log("5. DUPLICATE DETECTION (MD5 HASHES OF ALL IMAGES)")
    log("-" * 40)
    log("Computing MD5 hashes across all train and val images...")

    all_images_with_split = []
    for split in splits:
        for p in split_images[split]:
            all_images_with_split.append((p, split))

    # Fast hash calculation
    hash_to_files = defaultdict(list)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(compute_file_md5, p) for p, _ in all_images_with_split]
        for idx, fut in enumerate(futures):
            p, h, err = fut.result()
            split = all_images_with_split[idx][1]
            if h:
                hash_to_files[h].append((p, split))
            else:
                log(f"Warning: Failed to hash {p}: {err}")

    # Analyze duplicates
    train_dupes = []
    val_dupes = []
    leakage_dupes = []

    for h, entries in hash_to_files.items():
        if len(entries) > 1:
            tr_entries = [e for e in entries if e[1] == 'train']
            va_entries = [e for e in entries if e[1] == 'val']

            if len(tr_entries) > 1:
                train_dupes.append(tr_entries)
            if len(va_entries) > 1:
                val_dupes.append(va_entries)
            if len(tr_entries) >= 1 and len(va_entries) >= 1:
                leakage_dupes.append((tr_entries, va_entries))

    log(f"Total Unique Hashes: {len(hash_to_files)}")
    log(f"Exact Duplicate Groups within Train: {len(train_dupes)}")
    total_tr_dupe_files = sum(len(group) for group in train_dupes)
    log(f"  Total duplicate files involved in Train: {total_tr_dupe_files}")
    if train_dupes:
        for group in train_dupes[:5]:
            names = [p.name for p, s in group]
            log(f"    - Duplicates ({len(names)}): {names}")

    log(f"Exact Duplicate Groups within Val: {len(val_dupes)}")
    total_va_dupe_files = sum(len(group) for group in val_dupes)
    log(f"  Total duplicate files involved in Val: {total_va_dupe_files}")
    if val_dupes:
        for group in val_dupes[:5]:
            names = [p.name for p, s in group]
            log(f"    - Duplicates ({len(names)}): {names}")

    log(f"CRITICAL: Exact Duplicates Across Train & Val (LEAKAGE): {len(leakage_dupes)} groups")
    if leakage_dupes:
        total_leak_tr = sum(len(tr) for tr, va in leakage_dupes)
        total_leak_va = sum(len(va) for tr, va in leakage_dupes)
        log(f"  Total Leaked Files: {total_leak_tr} train images match {total_leak_va} val images!")
        for idx, (tr_entries, va_entries) in enumerate(leakage_dupes[:15], 1):
            tr_names = [p.name for p, _ in tr_entries]
            va_names = [p.name for p, _ in va_entries]
            log(f"    Leak #{idx}: Train {tr_names} == Val {va_names}")
        if len(leakage_dupes) > 15:
            log(f"    ... and {len(leakage_dupes) - 15} more leaked duplicate groups")
    else:
        log("  [PASS] No exact hash duplicates found between train and val.")
    log()

    # ---------------------------------------------------------
    # 6. AUGMENTATION PATTERN CHECK
    # ---------------------------------------------------------
    log("6. AUGMENTATION PATTERN CHECK & SPLIT LEAKAGE")
    log("-" * 40)

    base_groups = defaultdict(lambda: {'train': [], 'val': [], 'patterns': set()})

    for split in splits:
        for img in split_images[split]:
            base_name, pattern_type = extract_base_name(img.name)
            base_groups[base_name][split].append(img.name)
            if pattern_type != 'original':
                base_groups[base_name]['patterns'].add(pattern_type)

    aug_groups = {k: v for k, v in base_groups.items() if (len(v['train']) + len(v['val']) > 1) or v['patterns']}
    aug_leakage_groups = {k: v for k, v in base_groups.items() if len(v['train']) > 0 and len(v['val']) > 0}

    log(f"Inferred Base Image Roots: {len(base_groups)}")
    log(f"Groups with Augmented/Multiple Variants: {len(aug_groups)}")
    all_patterns = Counter()
    for v in base_groups.values():
        for pat in v['patterns']:
            all_patterns[pat] += 1
    log(f"Augmentation Pattern Types Identified: {dict(all_patterns)}")

    log(f"CRITICAL: Augmentation Split Leakage Groups: {len(aug_leakage_groups)}")
    if aug_leakage_groups:
        total_leak_aug_tr = sum(len(v['train']) for v in aug_leakage_groups.values())
        total_leak_aug_va = sum(len(v['val']) for v in aug_leakage_groups.values())
        log(f"  Total Images Involved in Augmentation Leakage: {total_leak_aug_tr} in train, {total_leak_aug_va} in val")
        log("  Sample Augmentation Leakage Pairs (Original/Variant in Train, Variant in Val):")
        for idx, (base_name, group) in enumerate(list(aug_leakage_groups.items())[:15], 1):
            log(f"    Leak #{idx} [Base: {base_name}]:")
            log(f"      Train ({len(group['train'])}): {group['train'][:3]}")
            log(f"      Val   ({len(group['val'])}): {group['val'][:3]}")
        if len(aug_leakage_groups) > 15:
            log(f"    ... and {len(aug_leakage_groups) - 15} more leakage groups")
    else:
        log("  [PASS] No augmentation split leakage detected based on filename root matching.")
    log()

    # ---------------------------------------------------------
    # 7. EMERGENCY CLASS ANALYSIS
    # ---------------------------------------------------------
    log("7. EMERGENCY CLASS ANALYSIS (AMBULANCE & POLICE_CAR)")
    log("-" * 40)

    amb_tr_objs = class_object_counts['train'][6]
    amb_va_objs = class_object_counts['val'][6]
    amb_tr_imgs = class_image_counts['train'][6]
    amb_va_imgs = class_image_counts['val'][6]

    pol_tr_objs = class_object_counts['train'][7]
    pol_va_objs = class_object_counts['val'][7]
    pol_tr_imgs = class_image_counts['train'][7]
    pol_va_imgs = class_image_counts['val'][7]

    emerg_tr_objs = amb_tr_objs + pol_tr_objs
    emerg_va_objs = amb_va_objs + pol_va_objs
    non_emerg_tr_objs = total_train_objs - emerg_tr_objs
    non_emerg_va_objs = total_val_objs - emerg_va_objs

    log("Class 6: Ambulance")
    log(f"  Train: {amb_tr_objs} objects in {amb_tr_imgs} images ({amb_tr_imgs/total_train_imgs*100:.2f}% of train images)")
    log(f"  Val  : {amb_va_objs} objects in {amb_va_imgs} images ({amb_va_imgs/total_val_imgs*100:.2f}% of val images)")
    log(f"  Total: {amb_tr_objs + amb_va_objs} objects in {amb_tr_imgs + amb_va_imgs} images")
    log()

    log("Class 7: Police Car")
    log(f"  Train: {pol_tr_objs} objects in {pol_tr_imgs} images ({pol_tr_imgs/total_train_imgs*100:.2f}% of train images)")
    log(f"  Val  : {pol_va_objs} objects in {pol_va_imgs} images ({pol_va_imgs/total_val_imgs*100:.2f}% of val images)")
    log(f"  Total: {pol_tr_objs + pol_va_objs} objects in {pol_tr_imgs + pol_va_imgs} images")
    log()

    log("Emergency vs Non-Emergency Totals:")
    log(f"  Train Emergency Objects     : {emerg_tr_objs} ({emerg_tr_objs/total_train_objs*100:.2f}% of total)")
    log(f"  Train Non-Emergency Objects : {non_emerg_tr_objs} ({non_emerg_tr_objs/total_train_objs*100:.2f}% of total)")
    log(f"  Val Emergency Objects       : {emerg_va_objs} ({emerg_va_objs/total_val_objs*100:.2f}% of total)")
    log(f"  Val Non-Emergency Objects   : {non_emerg_va_objs} ({non_emerg_va_objs/total_val_objs*100:.2f}% of total)")
    log()

    log("Reliability Assessment for Validation Split:")
    eval_comments = []
    if amb_va_objs < 20:
        eval_comments.append(f"CRITICAL DEFICIENCY: Ambulance has only {amb_va_objs} objects ({amb_va_imgs} images) in val. Insufficient for statistically reliable mAP evaluation.")
    elif amb_va_objs < 50:
        eval_comments.append(f"MARGINAL: Ambulance has {amb_va_objs} objects in val. AP metric will have high variance (+/- 5-10% mAP).")
    else:
        eval_comments.append(f"SUFFICIENT: Ambulance has {amb_va_objs} objects in val.")

    if pol_va_objs < 20:
        eval_comments.append(f"CRITICAL DEFICIENCY: Police Car has only {pol_va_objs} objects ({pol_va_imgs} images) in val. Insufficient for statistically reliable mAP evaluation.")
    elif pol_va_objs < 50:
        eval_comments.append(f"MARGINAL: Police Car has {pol_va_objs} objects in val. AP metric will have high variance (+/- 5-10% mAP).")
    else:
        eval_comments.append(f"SUFFICIENT: Police Car has {pol_va_objs} objects in val.")

    for cmt in eval_comments:
        log(f"  * {cmt}")
    log()

    # ---------------------------------------------------------
    # 8. SUMMARY RECOMMENDATIONS
    # ---------------------------------------------------------
    log("8. SUMMARY OF ISSUES & ACTIONABLE RECOMMENDATIONS")
    log("-" * 40)
    has_issues = False

    if len(leakage_dupes) > 0:
        has_issues = True
        log(f"- DATA LEAKAGE DETECTED: {len(leakage_dupes)} exact duplicate image groups exist across train and val splits.")
        log("  RECOMMENDATION: Remove leaking duplicates from validation set or relocate them to train.")

    if len(aug_leakage_groups) > 0:
        has_issues = True
        log(f"- AUGMENTATION LEAKAGE: {len(aug_leakage_groups)} image root groups have variants split across train and val.")
        log("  RECOMMENDATION: Group images by original capture/video sequence before splitting into train/val.")

    for split in splits:
        if pairing_stats[split]['orphan_images']:
            has_issues = True
            log(f"- ORPHAN IMAGES in {split}: {len(pairing_stats[split]['orphan_images'])} images have no label file.")
        if pairing_stats[split]['orphan_labels']:
            has_issues = True
            log(f"- ORPHAN LABELS in {split}: {len(pairing_stats[split]['orphan_labels'])} labels have no image file.")

    for split in splits:
        if annotation_stats[split]['inv_tokens'] > 0 or annotation_stats[split]['inv_classes'] > 0:
            has_issues = True
            log(f"- INVALID ANNOTATIONS in {split}: Check label syntax.")

    if not has_issues:
        log("- [PASSED ALL SANITY CHECKS] Dataset is clean with no pairing or syntax errors.")

    log("=" * 80)
    log("END OF INSPECTION REPORT")
    log("=" * 80)

    # Save to file
    report_text = "\n".join(lines)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    log(f"\nReport successfully saved to: {report_file}")
    return report_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Dataset Quality Inspector")
    parser.add_argument('--dataset_dir', type=str, default=r"C:\Users\Asus\Downloads\rm task\combined_dataset")
    parser.add_argument('--output_dir', type=str, default=r"C:\Users\Asus\Downloads\rm task\Emergency_Vehicle_Project\results\dataset_report")
    parser.add_argument('--sample_size', type=int, default=200)

    args = parser.parse_args()
    run_dataset_inspection(args.dataset_dir, args.output_dir, args.sample_size)
