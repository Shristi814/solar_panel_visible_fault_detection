# fetch_places_negatives.py
import random
import shutil
from pathlib import Path
from torchvision import datasets

CACHE = Path("data/_places_cache")
OUT = Path("data/not_panel_raw")
PER_CAT = 5
SEED = 42
WANT = [
    "living_room", "bedroom", "kitchen", "office", "corridor", "lobby",
    "street", "highway", "parking_lot", "courtyard", "patio", "balcony/exterior",
    "playground", "forest_path", "field/cultivated", "beach",
    "skyscraper", "building_facade", "roof_garden", "greenhouse/outdoor",
    "bus_station/indoor", "swimming_pool/outdoor",
]

try:
    ds = datasets.Places365(root=str(CACHE), split="val", small=True, download=True)
except RuntimeError:       # pehle se extract ho chuka hai
    ds = datasets.Places365(root=str(CACHE), split="val", small=True, download=False)

by_cls = {}
for path, idx in ds.imgs:
    by_cls.setdefault(idx, []).append(path)

rng = random.Random(SEED)
saved = 0
for w in WANT:
    idxs = [ds.class_to_idx[c] for c in ds.classes if c.endswith("/" + w)]
    if not idxs:
        print("[!] category not found:", w)
        continue
    files = by_cls[idxs[0]]
    for j, p in enumerate(rng.sample(files, min(PER_CAT, len(files))), 1):
        dst = OUT / f"places_{w.replace('/', '_')}" / f"{j:02d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)
        saved += 1
print("Saved", saved, "images in", OUT)