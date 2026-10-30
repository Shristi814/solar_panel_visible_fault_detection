# add_not_panel.py
import hashlib
import random
import sys
from pathlib import Path
from PIL import Image, ImageOps

RAW = Path("data/not_panel_raw")
CLEAN = Path("data/clean")
CLASS = "Not-a-panel"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_SIDE = 1600
SEED = 42
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if any((CLEAN / s / CLASS).exists() for s in SPLITS):
    sys.exit(f"[!] {CLASS} folders already exist under {CLEAN}. Delete them to redo.")


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


# 1) panel images ke hashes (accidental overlap rokne ke liye)
panel_hashes = {md5(p) for p in CLEAN.rglob("*") if p.suffix.lower() in EXT}

# 2) collect + dedupe. Subfolder = ek group, root ki file = apna group
groups, seen, dups = {}, set(), 0
for p in sorted(RAW.rglob("*")):
    if p.suffix.lower() not in EXT:
        continue
    h = md5(p)
    if h in seen or h in panel_hashes:
        dups += 1
        continue
    seen.add(h)
    rel = p.relative_to(RAW)
    key = rel.parts[0] if len(rel.parts) > 1 else f"file:{p.name}"
    groups.setdefault(key, []).append(p)

n_total = sum(len(v) for v in groups.values())
print(f"Unique images: {n_total} in {len(groups)} groups (duplicates skipped: {dups})")
if n_total < 60:
    sys.exit("[!] Too few images. Add more (aim for 150-200).")

# 3) poore groups ko splits mein baanto
rng = random.Random(SEED)
keys = list(groups)
rng.shuffle(keys)
keys.sort(key=lambda k: -len(groups[k]))       # bade groups pehle
count = {s: 0 for s in SPLITS}
assign = {}
for k in keys:
    s = max(SPLITS, key=lambda s: SPLITS[s] * n_total - count[s])
    assign[k] = s
    count[s] += len(groups[k])
print("Split sizes:", count)
if min(count.values()) < 10:
    sys.exit(f"[!] Split sizes {count} are too uneven. Add more scene folders (10+) and rerun.")

# 4) clean RGB JPEG ke roop mein save (EXIF rotation lagao, bade images chhote karo)
n = {s: 0 for s in SPLITS}
for k, files in groups.items():
    s = assign[k]
    for p in files:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > MAX_SIDE:
                im.thumbnail((MAX_SIDE, MAX_SIDE))
            n[s] += 1
            out = CLEAN / s / CLASS / f"{CLASS}_{n[s]:04d}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, quality=92)
print("Saved:", n)