from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import random
from PIL import Image

DATA_DIR = Path(r"D:\projects\Solar_visible_fault_detection\data\clean")
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    # rglob nested folders ke andar ki images bhi ginta hai
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT]


# 1) Har split mein class folders kaunse hain
classes_per_split = {}
for s in SPLITS:
    sp = DATA_DIR / s
    if not sp.exists():
        print(f"[!] Split folder nahi mila: {sp}")
        continue
    classes_per_split[s] = sorted(d.name for d in sp.iterdir() if d.is_dir())
    print(f"{s}: {classes_per_split[s]}")

all_sets = [set(v) for v in classes_per_split.values()]
if all(a == all_sets[0] for a in all_sets):
    print("\nClass folder names teeno splits mein same hain.")
else:
    print("\n[!] Class folder names splits mein alag hain, ImageFolder ka class order bigad sakta hai.")

# 2) Split x class count table
all_classes = sorted(set().union(*all_sets))
images = {}
print(f"\n{'class':<20}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'total':>8}")
split_totals = Counter()
for cls in all_classes:
    row, total = [], 0
    for s in SPLITS:
        folder = DATA_DIR / s / cls
        files = list_images(folder) if folder.exists() else []
        images[(s, cls)] = files
        row.append(len(files))
        split_totals[s] += len(files)
        total += len(files)
    print(f"{cls:<20}" + "".join(f"{n:>8}" for n in row) + f"{total:>8}")
grand = sum(split_totals.values())
print(f"{'TOTAL':<20}" + "".join(f"{split_totals[s]:>8}" for s in SPLITS) + f"{grand:>8}")
for s in SPLITS:
    print(f"{s}: {100 * split_totals[s] / grand:.1f}%")

# 3) Image size aur mode (train se har class ke 20 random sample)
random.seed(42)
print("\nImage size / mode (train sample):")
for cls in all_classes:
    files = images[("train", cls)]
    sample = random.sample(files, min(20, len(files)))
    widths, heights, modes = [], [], Counter()
    for f in sample:
        with Image.open(f) as im:
            widths.append(im.size[0])
            heights.append(im.size[1])
            modes[im.mode] += 1
    if sample:
        print(f"{cls:<20} width {min(widths)}-{max(widths)}, "
              f"height {min(heights)}-{max(heights)}, modes {dict(modes)}")

# 4) Duplicate / leakage check (exact duplicates, MD5 hash se)
def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

hashes = defaultdict(list)
for (s, cls), files in images.items():
    for f in files:
        hashes[md5(f)].append((s, cls, f))

cross_split = [v for v in hashes.values() if len({x[0] for x in v}) > 1]
same_split_dups = [v for v in hashes.values()
                   if len(v) > 1 and len({x[0] for x in v}) == 1]
label_conflicts = [v for v in hashes.values() if len({x[1] for x in v}) > 1]

print(f"\nSplits ke beech common images (leakage): {len(cross_split)}")
print(f"Same split ke andar duplicates: {len(same_split_dups)}")
print(f"Alag classes mein same image (label conflict): {len(label_conflicts)}")
for group in cross_split[:5]:
    print("  e.g.", [(s, cls, p.name) for s, cls, p in group])