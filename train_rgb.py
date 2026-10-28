import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (accuracy_score, f1_score,

                             classification_report, confusion_matrix)

from model_utils import build_model , IMG_SIZE

DATA_DIR = Path(r"D:\projects\Solar_visible_fault_detection\data\clean")
OUT_DIR = Path("models")
# IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 2          # Windows pe error aaye toh 0 kar do
SEED = 42


# def build_model(name, num_classes):
#     if name == "mobilenet_v3":
#         model = models.mobilenet_v3_large(
#             weights=models.MobileNet_V3_Large_Weights.DEFAULT)
#         model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
#     elif name == "resnet50":
#         model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#     else:
#         raise ValueError(name)
#     return model


def get_data():
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([       # val/test/app: sirf resize + normalize
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), norm,
    ])
    train_set = datasets.ImageFolder(DATA_DIR / "train", train_tf)
    val_set = datasets.ImageFolder(DATA_DIR / "val", eval_tf)
    test_set = datasets.ImageFolder(DATA_DIR / "test", eval_tf)
    assert train_set.classes == val_set.classes == test_set.classes
    return train_set, val_set, test_set


def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, labels_all, total_loss = [], [], 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item() * images.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(preds), np.array(labels_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3",
                    choices=["mobilenet_v3", "resnet50"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    OUT_DIR.mkdir(exist_ok=True)
    ckpt = OUT_DIR / f"{args.model}_rgb.pth"

    train_set, val_set, test_set = get_data()
    class_names = train_set.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_set, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_set, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin)

    model = build_model(args.model, num_classes).to(device)

    if not args.test_only:
        with open(OUT_DIR / "class_names.json", "w") as f:
            json.dump(class_names, f)

        counts = np.bincount(train_set.targets, minlength=num_classes)
        weights = len(train_set) / (num_classes * counts)   # kam images -> zyada weight
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_f1 = -1.0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            running = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running += loss.item() * images.size(0)

            val_loss, vp, vl = evaluate(model, val_loader, device, criterion)
            val_acc = accuracy_score(vl, vp)
            val_f1 = f1_score(vl, vp, average="macro", zero_division=0)
            print(f"Epoch {epoch}/{args.epochs} | train loss {running/len(train_set):.4f} "
                  f"| val loss {val_loss:.4f} | val acc {val_acc*100:.2f}% "
                  f"| val macro-F1 {val_f1*100:.2f}% | {time.time()-t0:.0f}s")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), ckpt)
                print("  -> best model saved")
        print(f"\nBest val macro-F1: {best_f1*100:.2f}%")
    else:
        # TEST: sirf final model pe, ek baar
        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, tp, tl = evaluate(model, test_loader, device)
        print(f"Test acc: {accuracy_score(tl, tp)*100:.2f}% | "
              f"Test macro-F1: {f1_score(tl, tp, average='macro', zero_division=0)*100:.2f}%")
        print(classification_report(tl, tp, target_names=class_names, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(tl, tp))


if __name__ == "__main__":      # Windows + num_workers ke liye zaroori
    main()