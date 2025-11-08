import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from datareader import get_data_loaders, NEW_CLASS_NAMES
# prefer SimpleCNN export (DenseNet wrapper) but try DashNet name too
try:
    from model import SimpleCNN as BackboneClass
except Exception:
    from model import DashNet as BackboneClass
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"

def build_model(in_channels, num_classes):
    # try to construct a pretrained backbone if constructor supports it
    try:
        m = BackboneClass(in_channels=in_channels, num_classes=num_classes, pretrained=True)
    except TypeError:
        # fallback to constructor without pretrained
        m = BackboneClass(in_channels=in_channels, num_classes=num_classes)
    return m

def compute_pos_weight_from_dataset(train_loader):
    try:
        # many datasets expose samples list: (path, label)
        samples = train_loader.dataset.samples
        labels = [s[1] for s in samples]
        counts = np.bincount(labels)
        if len(counts) >= 2 and counts[1] > 0:
            pos_weight = float(counts[0]) / float(counts[1])
            return torch.tensor([pos_weight], dtype=torch.float)
    except Exception:
        pass
    return None

def tune_threshold(probs, targets):
    # find threshold that maximizes F1 on validation set
    best_thr = 0.5
    best_f1 = -1.0
    try:
        targets = np.asarray(targets).ravel()
        probs = np.asarray(probs).ravel()
        for thr in np.linspace(0.1, 0.9, 81):
            preds = (probs > thr).astype(np.int32)
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            if (prec + rec) > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
            else:
                f1 = 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
    except Exception:
        best_thr = 0.5
    return best_thr, best_f1

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model (pindahkan ke device)
    model = build_model(in_channels=in_channels, num_classes=num_classes).to(DEVICE)
    print(model)
    
    # 3. Loss + optimizer + scheduler + amp scaler
    pos_weight = compute_pos_weight_from_dataset(train_loader)
    if num_classes == 2:
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()

    # history
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_state = None
    patience = 8
    wait = 0
    best_threshold = 0.5

    print("\n--- Memulai Training ---")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            if num_classes == 2:
                labels = labels.view(-1).float().to(DEVICE)
            else:
                labels = labels.long().to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            if num_classes == 2:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = 100.0 * running_correct / max(1, running_total)

        # validation: collect probabilities + targets for threshold tuning
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        val_correct = 0
        probs_all = []
        targets_all = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                if num_classes == 2:
                    labels_v = labels.view(-1).float().to(DEVICE)
                else:
                    labels_v = labels.long().to(DEVICE)

                with autocast():
                    outputs = model(images)
                    vloss = criterion(outputs, labels_v)

                val_loss_sum += vloss.item() * images.size(0)
                if num_classes == 2:
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    preds = (probs > 0.5).astype(np.int32)
                    probs_all.append(probs)
                    targets_all.append(labels_v.detach().cpu().numpy())
                    val_correct += ( (torch.from_numpy(preds).to(DEVICE).view(-1).float() ) == labels_v ).sum().item()
                else:
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels_v).sum().item()
                val_total += labels_v.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        # tune threshold only if binary
        val_f1 = 0.0
        if num_classes == 2 and len(probs_all) > 0:
            probs_cat = np.concatenate(probs_all).ravel()
            targets_cat = np.concatenate(targets_all).ravel()
            thr, f1 = tune_threshold(probs_cat, targets_cat)
            best_threshold = thr
            val_f1 = f1
            val_acc = 100.0 * ((probs_cat > best_threshold).astype(np.int32) == targets_cat).mean()
        else:
            val_acc = 100.0 * val_correct / max(1, val_total)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{EPOCHS} time:{time.time()-t0:.1f}s "
              f"train_loss:{train_loss:.4f} train_acc:{train_acc:.2f}% "
              f"val_loss:{val_loss:.4f} val_acc:{val_acc:.2f}% val_f1:{val_f1:.4f} thr:{best_threshold:.2f}")

        # checkpoint and early stopping by val_f1 or val_acc
        improved = False
        if num_classes == 2:
            if val_f1 > best_val_f1 or val_acc > best_val_acc:
                improved = True
        else:
            if val_acc > best_val_acc:
                improved = True

        if improved:
            best_val_acc = max(val_acc, best_val_acc)
            best_val_f1 = max(val_f1, best_val_f1)
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "threshold": best_threshold
            }
            torch.save(best_state, CHECKPOINT_PATH)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # selesai training: load best
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        best_threshold = best_state.get("threshold", best_threshold)

    # simpan history plot & visualisasi
    plot_training_history(history["train_loss"], history["val_loss"], history["train_acc"], history["val_acc"])
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()
