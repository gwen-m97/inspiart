import os, random, math, json
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from timm.data import resolve_model_data_config, create_transform
import timm
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "images"    # images liegt direkt neben dem Script
OUT_DIR  = SCRIPT_DIR / "models"    # models-Ordner wird ebenfalls hier angelegt
OUT_DIR.mkdir(exist_ok=True)

print("Using DATA_DIR:", DATA_DIR)
print("Using OUT_DIR:", OUT_DIR)

assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"
subdirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
assert len(subdirs) >= 2, f"Need >=2 class folders in {DATA_DIR}, found: {len(subdirs)}"

def stratified_split(dataset, val_ratio=0.15, test_ratio=0.15):
    # ImageFolder: samples = [(path, class_idx), ...]
    y = [lbl for _, lbl in dataset.samples]
    y = np.array(y)
    idxs = np.arange(len(y))

    # pro Klasse aufteilen
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        c_idx = idxs[y == c]
        rng = np.random.default_rng(SEED)
        rng.shuffle(c_idx)

        n = len(c_idx)
        n_test = int(round(n * test_ratio))
        n_val  = int(round(n * val_ratio))
        n_train = n - n_val - n_test

        train_idx += list(c_idx[:n_train])
        val_idx   += list(c_idx[n_train:n_train+n_val])
        test_idx  += list(c_idx[n_train+n_val:])
    return train_idx, val_idx, test_idx

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
# 1) Pretrained VGG16 laden (timm)

# Dataset einmal laden, nur um Klassen zu bestimmen
    tmp_ds = datasets.ImageFolder(DATA_DIR, transform=None)
    class_names = tmp_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names, "(", num_classes, ")")

# Modell mit passender Klassenzahl
    model = timm.create_model("vgg16.tv_in1k", pretrained=True, num_classes=num_classes)

# Transforms
    cfg = resolve_model_data_config(model)
    tfm_train = create_transform(**cfg, is_training=True)
    tfm_val   = create_transform(**cfg, is_training=False)

# 2) Dataset mit Transforms laden
    full_train = datasets.ImageFolder(DATA_DIR, transform=tfm_train)


    # 4) Splits (stratifiziert)
    # Für val/test andere Transforms verwenden -> separate Datasets
    train_idx, val_idx, test_idx = stratified_split(full_train, val_ratio=0.15, test_ratio=0.15)

    train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=tfm_train), train_idx)
    val_ds   = Subset(datasets.ImageFolder(DATA_DIR, transform=tfm_val),   val_idx)
    test_ds  = Subset(datasets.ImageFolder(DATA_DIR, transform=tfm_val),   test_idx)

    # 5) Dataloader
    BATCH = 8 if device == "cpu" else 16  # kleinere Batchsize auf CPU
    pin = (device == "cuda")              # pin_memory nur für CUDA sinnvoll

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                          num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                          num_workers=2, pin_memory=pin)
    # 6) Transfer-Learning: erst Backbone einfrieren
    for p in model.features.parameters():
        p.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # 7) Training (Head)
    BEST_PATH = OUT_DIR / "vgg16_genre_best.pt"
    best_val = float("inf")
    EPOCHS = 8

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss, tr_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        tr_acc = tr_correct / n

        # validate
        model.eval()
        val_loss, val_correct, m = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                m += y.size(0)
        val_acc = val_correct / m
        val_loss /= m

        print(f"[{epoch}/{EPOCHS}] train_loss={tr_loss/n:.4f} acc={tr_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
            {"state_dict": model.state_dict(), "classes": class_names},
            BEST_PATH)

            print("  ↳ saved:", BEST_PATH)

    # 8) Optional: leichtes Fine-Tuning (letzte Blöcke öffnen)
    for p in model.features[-5:].parameters():  # ein paar letzte Layer
        p.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    for epoch in range(1, 3):  # kurze FT-Phase
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"[finetune epoch {epoch}] done")

    torch.save({"state_dict": model.state_dict(), "classes": class_names},
           OUT_DIR / "vgg16_genre_final.pt")

    # 9) Test-Evaluierung
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x).cpu()
            y_true += list(y.numpy())
            y_pred += list(logits.argmax(1).numpy())

    print("\nClassification report (test):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
