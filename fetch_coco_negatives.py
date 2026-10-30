# fetch_coco_negatives.py
import random
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image

URL = "http://images.cocodataset.org/zips/val2017.zip"
ZIP = Path("data/_cache/val2017.zip")
OUT = Path("data/not_panel_raw")
N, SEED = 60, 42

ZIP.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

if not ZIP.exists():
    print("Downloading val2017.zip (about 1 GB)...")

    def hook(block, block_size, total):
        if total > 0 and block % 3000 == 0:
            print(f"  {min(100, block * block_size * 100 // total)}%")

    urllib.request.urlretrieve(URL, ZIP, hook)

with zipfile.ZipFile(ZIP) as z:
    names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
    random.Random(SEED).shuffle(names)
    for i, n in enumerate(names[:N], 1):
        with Image.open(BytesIO(z.read(n))) as im:
            im.convert("RGB").save(OUT / f"coco_{i:04d}.jpg", quality=92)
print(f"Saved {N} images to {OUT}")