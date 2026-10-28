
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from model_utils import build_model, IMG_SIZE

MODELS_DIR = Path("models")

EVAL_TF = transforms.Compose([                # wahi jo training ke val/test mein tha
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_class_names(path=MODELS_DIR / "class_names.json"):
    return json.loads(Path(path).read_text())


def load_model(model_name, ckpt_path, num_classes, device):
    model = build_model(model_name, num_classes, pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model, image, class_names, device):
    image = ImageOps.exif_transpose(image).convert("RGB")
    x = EVAL_TF(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(dim=0)
    all_probs = {c: probs[i].item() for i, c in enumerate(class_names)}
    return class_names[idx.item()], conf.item(), all_probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="mobilenet_v3", choices=["mobilenet_v3", "resnet50"])
    args = ap.parse_args()

    device = torch.device("cpu")
    class_names = load_class_names()
    model = load_model(args.model, MODELS_DIR / f"{args.model}_rgb.pth",
                       len(class_names), device)

    with Image.open(args.image) as img:
        name, conf, probs = predict(model, img, class_names, device)
    print(f"Prediction: {name}  (confidence {conf*100:.1f}%)")
    for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<20}{p*100:6.1f}%")