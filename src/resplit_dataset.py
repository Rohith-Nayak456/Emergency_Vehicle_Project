"""
Dataset Re-splitting Script

Pools all images from the existing train/val splits and creates a new
leakage-resistant 70/15/15 train/val/test split with:
  - Augmented images grouped with their originals
  - Stratified by emergency class presence
  - Reproducible random seed
  - Image-label pairs kept together

This script creates SYMLINKS (or copies) in a new directory structure.
The original combined_dataset is NOT modified.
"""
import os
import re
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\Asus\Downloads\rm task\combined_dataset")
OUTPUT_ROOT = Path(r"C:\Users\Asus\Downloads\rm task\Emergency_Vehicle_Project\data\split_dataset")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EMERGENCY_CLASSES = {6, 7}  # ambulance, police_car


def extract_group_id(filename: str) -> str:
    """
    Extract the group ID from a filename.
    Image_1001.png -> Image_1001
    Image_1001_bal556_bicycle.png -> Image_1001
    """
    stem = Path(filename).stem
    match = re.match(r'^(Image_\d+)', stem)
    if match:
        return match.group(1)
    return stem  # fallback: treat as its own group


def get_classes_in_label(label_path: Path) -> set:
    """Get the set of class IDs present in a label file."""
    classes = set()
    if not label_path.exists():
        return classes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cls_id = int(line.split()[0])
                    classes.add(cls_id)
                except (ValueError, IndexError):
                    pass
    return classes


def collect_all_image_label_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    """Collect all image-label pairs from both train and val splits."""
    pairs = []
    for split in ['train', 'val']:
        img_dir = dataset_root / split / 'images'
        lbl_dir = dataset_root / split / 'labels'
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob('*')):
            if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                lbl_path = lbl_dir / (img_path.stem + '.txt')
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path))
    return pairs


def group_pairs(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    """Group image-label pairs by their group ID (original image)."""
    groups = defaultdict(list)
    for img_path, lbl_path in pairs:
        gid = extract_group_id(img_path.name)
        groups[gid].append((img_path, lbl_path))
    return dict(groups)


def classify_group(group_pairs: list[tuple[Path, Path]]) -> str:
    """
    Classify a group as 'emergency' or 'normal' based on
    whether any image in the group contains an emergency vehicle.
    """
    for _, lbl_path in group_pairs:
        classes = get_classes_in_label(lbl_path)
        if classes & EMERGENCY_CLASSES:
            return 'emergency'
    return 'normal'


def stratified_split(groups: dict, seed: int = SEED):
    """
    Split groups into train/val/test with stratification by
    emergency class presence.
    """
    rng = random.Random(seed)

    emergency_groups = []
    normal_groups = []

    for gid, pairs in groups.items():
        category = classify_group(pairs)
        if category == 'emergency':
            emergency_groups.append((gid, pairs))
        else:
            normal_groups.append((gid, pairs))

    rng.shuffle(emergency_groups)
    rng.shuffle(normal_groups)

    def split_list(items, train_r, val_r):
        n = len(items)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        return train, val, test

    emg_train, emg_val, emg_test = split_list(
        emergency_groups, TRAIN_RATIO, VAL_RATIO
    )
    nor_train, nor_val, nor_test = split_list(
        normal_groups, TRAIN_RATIO, VAL_RATIO
    )

    train_groups = emg_train + nor_train
    val_groups = emg_val + nor_val
    test_groups = emg_test + nor_test

    return train_groups, val_groups, test_groups


def flatten_groups(group_list) -> list[tuple[Path, Path]]:
    """Flatten a list of (gid, pairs) into a list of (img_path, lbl_path)."""
    result = []
    for _, pairs in group_list:
        result.extend(pairs)
    return result


def copy_pairs_to_split(pairs, split_name, output_root):
    """Copy image-label pairs to the output directory."""
    img_dir = output_root / split_name / 'images'
    lbl_dir = output_root / split_name / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_dir / img_path.name)
        shutil.copy2(lbl_path, lbl_dir / lbl_path.name)


def count_classes_in_pairs(pairs) -> dict[int, int]:
    """Count objects per class across all label files."""
    counts = defaultdict(int)
    for _, lbl_path in pairs:
        with open(lbl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cls_id = int(line.split()[0])
                        counts[cls_id] += 1
                    except (ValueError, IndexError):
                        pass
    return dict(counts)


def count_images_with_class(pairs, cls_id) -> int:
    """Count images containing at least one object of the given class."""
    count = 0
    for _, lbl_path in pairs:
        classes = get_classes_in_label(lbl_path)
        if cls_id in classes:
            count += 1
    return count


CLASS_NAMES = {
    0: 'car', 1: 'motorcycle', 2: 'autorickshaw', 3: 'truck',
    4: 'bus', 5: 'bicycle', 6: 'ambulance', 7: 'police_car'
}


def main():
    print("=" * 70)
    print("  DATASET RE-SPLITTING: Leakage-Resistant Stratified Split")
    print("=" * 70)

    # 1. Collect all pairs
    print(f"\nCollecting all image-label pairs from: {DATASET_ROOT}")
    all_pairs = collect_all_image_label_pairs(DATASET_ROOT)
    print(f"Total pairs collected: {len(all_pairs)}")

    # 2. Group by original image
    print("\nGrouping by original image ID...")
    groups = group_pairs(all_pairs)
    print(f"Total unique groups: {len(groups)}")

    emg_count = sum(1 for gid, pairs in groups.items() if classify_group(pairs) == 'emergency')
    nor_count = len(groups) - emg_count
    print(f"Emergency groups: {emg_count}")
    print(f"Normal groups:    {nor_count}")

    # 3. Stratified split
    print(f"\nSplitting with ratio {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}, seed={SEED}")
    train_groups, val_groups, test_groups = stratified_split(groups, SEED)

    train_pairs = flatten_groups(train_groups)
    val_pairs = flatten_groups(val_groups)
    test_pairs = flatten_groups(test_groups)

    print(f"\nSplit results (groups -> images):")
    print(f"  Train: {len(train_groups):>5} groups -> {len(train_pairs):>6} images")
    print(f"  Val:   {len(val_groups):>5} groups -> {len(val_pairs):>6} images")
    print(f"  Test:  {len(test_groups):>5} groups -> {len(test_pairs):>6} images")
    print(f"  Total: {len(train_groups)+len(val_groups)+len(test_groups):>5} groups -> "
          f"{len(train_pairs)+len(val_pairs)+len(test_pairs):>6} images")

    # 4. Class distribution per split
    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION IN NEW SPLITS")
    print("-" * 70)
    for split_name, pairs in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
        counts = count_classes_in_pairs(pairs)
        print(f"\n  {split_name} ({len(pairs)} images):")
        for cls_id in range(8):
            obj_count = counts.get(cls_id, 0)
            img_count = count_images_with_class(pairs, cls_id)
            marker = " *** EMERGENCY" if cls_id in EMERGENCY_CLASSES else ""
            print(f"    {CLASS_NAMES[cls_id]:15s}: {obj_count:>6} objects in {img_count:>5} images{marker}")

    # 5. Leakage verification
    print("\n" + "-" * 70)
    print("LEAKAGE VERIFICATION")
    print("-" * 70)
    train_gids = {gid for gid, _ in train_groups}
    val_gids = {gid for gid, _ in val_groups}
    test_gids = {gid for gid, _ in test_groups}

    tv_overlap = train_gids & val_gids
    tt_overlap = train_gids & test_gids
    vt_overlap = val_gids & test_gids

    print(f"  Train-Val group overlap:  {len(tv_overlap)} {'[PASS]' if len(tv_overlap) == 0 else '[FAIL!]'}")
    print(f"  Train-Test group overlap: {len(tt_overlap)} {'[PASS]' if len(tt_overlap) == 0 else '[FAIL!]'}")
    print(f"  Val-Test group overlap:   {len(vt_overlap)} {'[PASS]' if len(vt_overlap) == 0 else '[FAIL!]'}")

    # 6. Copy files
    print(f"\nCopying files to: {OUTPUT_ROOT}")
    if OUTPUT_ROOT.exists():
        print("  Removing existing split directory...")
        shutil.rmtree(OUTPUT_ROOT)

    copy_pairs_to_split(train_pairs, 'train', OUTPUT_ROOT)
    print(f"  Train: {len(train_pairs)} pairs copied")
    copy_pairs_to_split(val_pairs, 'val', OUTPUT_ROOT)
    print(f"  Val:   {len(val_pairs)} pairs copied")
    copy_pairs_to_split(test_pairs, 'test', OUTPUT_ROOT)
    print(f"  Test:  {len(test_pairs)} pairs copied")

    # 7. Create data.yaml
    yaml_path = OUTPUT_ROOT / 'data.yaml'
    yaml_content = f"""# Emergency Vehicle Detection Dataset
# Generated by re-split script with seed={SEED}
# Split ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}

path: {OUTPUT_ROOT.as_posix()}

train: train/images
val: val/images
test: test/images

nc: 8

names:
  0: car
  1: motorcycle
  2: autorickshaw
  3: truck
  4: bus
  5: bicycle
  6: ambulance
  7: police_car
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nData YAML saved: {yaml_path}")

    # 8. Save split manifest
    manifest_path = OUTPUT_ROOT / 'split_manifest.txt'
    with open(manifest_path, 'w') as f:
        f.write(f"Seed: {SEED}\n")
        f.write(f"Split ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}\n")
        f.write(f"Total groups: {len(groups)}\n")
        f.write(f"Emergency groups: {emg_count}\n")
        f.write(f"Normal groups: {nor_count}\n\n")
        for split_name, grps in [('train', train_groups), ('val', val_groups), ('test', test_groups)]:
            f.write(f"\n--- {split_name} ({len(grps)} groups) ---\n")
            for gid, pairs in sorted(grps, key=lambda x: x[0]):
                files = [p[0].name for p in pairs]
                f.write(f"{gid}: {', '.join(files)}\n")
    print(f"Split manifest saved: {manifest_path}")

    print("\n" + "=" * 70)
    print("  RE-SPLITTING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
