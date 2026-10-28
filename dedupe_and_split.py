from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import shutil
import sys
from sklearn.model_selection import train_test_split

SRC_DIR = Path(r"C:\Users\hp\Downloads\archive")                      # original dataset
OUT_DIR = Path(r"D:\projects\Solar_visible_fault_detection\data\clean")  # nayi clean copy
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# Purane output ke upar mat likho (test set ek baar banne ke baad nahi badalna)
if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
    sys.exit(f"[!] {OUT_DIR} pehle se bhara hai. Delete karke dobara chalao ya naya path do.")


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# 1) Teeno splits ki saari images ek pool mein, hash ke hisaab se group
by_hash = defaultdict(list)          # hash -> [(class, path), ...]
total_found = 0
for s in SPLITS:
    for cls_dir in sorted((SRC_DIR / s).iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                by_hash[md5(p)].append((cls_dir.name, p))
                total_found += 1
print(f"Total images mili: {total_found}, unique hashes: {len(by_hash)}")

# 2) Dedupe: ek hash = ek image. Alag classes mein same hash = label conflict, drop
kept, conflicts = [], []
for h, items in by_hash.items():
    classes = {c for c, _ in items}
    if len(classes) > 1:
        conflicts.append((h, items))
        continue
    items = sorted(items, key=lambda x: str(x[1]))
    kept.append((items[0][0], items[0][1], h))   # (class, path, hash)

print(f"Duplicates hataye: {total_found - len(by_hash)}")
print(f"Label-conflict wale groups (drop kiye): {len(conflicts)}")
print(f"Unique images bachi: {len(kept)}")

# 3) Stratified split 70 / 15 / 15
kept.sort(key=lambda x: str(x[1]))               # fixed order -> reproducible split
labels = [k[0] for k in kept]
idx = list(range(len(kept)))
train_idx, temp_idx = train_test_split(
    idx, test_size=0.30, stratify=labels, random_state=SEED)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=[labels[i] for i in temp_idx], random_state=SEED)

split_of = {}
for name, group in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
    for i in group:
        split_of[i] = name

# 4) Nayi folder structure mein copy + manifest
counter = Counter()
rows = []
for i, (cls, path, h) in enumerate(kept):
    s = split_of[i]
    counter[(s, cls)] += 1
    new_name = f"{cls}_{counter[(s, cls)]:04d}{path.suffix.lower()}"
    dst = OUT_DIR / s / cls / new_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    rows.append([str(path), h, s, cls, str(dst)])

with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["original_path", "md5", "split", "class", "new_path"])
    w.writerows(rows)

with open(OUT_DIR / "label_conflicts.txt", "w", encoding="utf-8") as f:
    for h, items in conflicts:
        f.write(f"{h}\n")
        for cls, p in items:
            f.write(f"   {cls}: {p}\n")

# 5) Final table
classes = sorted({k[0] for k in kept})
print(f"\n{'class':<20}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'total':>8}")
for cls in classes:
    row = [counter[(s, cls)] for s in SPLITS]
    print(f"{cls:<20}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>8}")
tot = [sum(counter[(s, c)] for c in classes) for s in SPLITS]
print(f"{'TOTAL':<20}" + "".join(f"{n:>8}" for n in tot) + f"{sum(tot):>8}")