#!/usr/bin/env python3
"""
1D-CNN Apnea Detection — End-to-End Training
=============================================
  直接把原始 ECG 信号喂给 CNN，模型自己学习判别呼吸暂停的波形模式。
  不需要人工设计 HRV 特征 —— 真正的"自主判断"。

  训练数据: a01–a20 (20 条确诊睡眠呼吸暂停患者)
  可选扩展: b01–b05 (临界), c01–c10 (健康对照)
"""

import subprocess, sys, os, pickle, warnings, time, logging, json
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cnn-train")

# ── 依赖检查 ──
REQUIRED = {"numpy":"numpy", "wfdb":"wfdb", "scipy":"scipy",
            "sklearn":"scikit-learn", "matplotlib":"matplotlib", "seaborn":"seaborn"}
for mod, pkg in REQUIRED.items():
    try: __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install",pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score, roc_curve)

# ═══════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════
DATA_DIR = os.environ.get("APNEA_DATA_DIR",
    r"C:\Users\林炀\Desktop\apnea-ecg-test-label\apnea-ecg-test-label\apnea-ecg")

# 训练记录: a01-a20 (基础) + b01-b05 (临界) + c01-c10 (对照) = 更多样本
TRAIN_RECORDS = [f"a{i:02d}" for i in range(1, 21)]   # a01-a20
EXTRA_RECORDS = [f"b{i:02d}" for i in range(1, 6)] + \
                [f"c{i:02d}" for i in range(1, 11)]   # b01-b05, c01-c10
# 设置 USE_ALL=True 使用全部 35 条有标签数据
USE_ALL = True
if USE_ALL:
    TRAIN_RECORDS = TRAIN_RECORDS + EXTRA_RECORDS

FS = 100
MINUTE_N = FS * 60             # 6000 samples / minute
RANDOM_SEED = 42
TEST_SPLIT = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
LR = 1e-3

OUTPUT_DIR = Path(DATA_DIR).parent
MODEL_PATH = OUTPUT_DIR / "cnn_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")
log.info(f"Training records: {len(TRAIN_RECORDS)} ({TRAIN_RECORDS[0]}..{TRAIN_RECORDS[-1]})")


# ═══════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════
def load_all_data(data_dir, records):
    """
    读取所有记录，切分为 1 分钟片段。
    返回: X (n_samples, 6000), y (n_samples,)
    """
    all_segments, all_labels = [], []
    stats = {"total_min": 0, "apnea_min": 0, "normal_min": 0}

    for rec in records:
        try:
            sig, _ = wfdb.rdsamp(os.path.join(data_dir, rec))
            ecg = sig[:, 0].astype(np.float64)

            ann = wfdb.rdann(os.path.join(data_dir, rec), "apn")
            labels = np.array([1 if s == "A" else 0 for s in ann.symbol], dtype=np.int32)

            n_minutes = min(len(labels), len(ecg) // MINUTE_N)

            for m in range(n_minutes):
                start = m * MINUTE_N
                end = start + MINUTE_N
                segment = ecg[start:end].copy()

                # z-score 标准化（每个片段独立）
                seg_std = segment.std()
                if seg_std > 1e-8:
                    segment = (segment - segment.mean()) / seg_std
                else:
                    segment = np.zeros_like(segment)

                all_segments.append(segment)
                all_labels.append(labels[m])

            a = int(labels[:n_minutes].sum())
            n = n_minutes - a
            stats["total_min"] += n_minutes
            stats["apnea_min"] += a
            stats["normal_min"] += n
            log.info(f"  {rec}: {n_minutes} min | A={a} ({100*a/max(1,n_minutes):.0f}%) | N={n}")

        except Exception as e:
            log.warning(f"  {rec}: SKIP ({e})")

    X = np.stack(all_segments, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)

    log.info(f"\nTotal: {len(y)} samples | A={stats['apnea_min']} "
             f"({100*stats['apnea_min']/stats['total_min']:.1f}%) | "
             f"N={stats['normal_min']} "
             f"({100*stats['normal_min']/stats['total_min']:.1f}%)")
    return X, y, stats


# ═══════════════════════════════════════════
# 2. Dataset & DataLoader
# ═══════════════════════════════════════════
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, 6000)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════
# 3. 1D-CNN 模型
# ═══════════════════════════════════════════
class ApneaCNN(nn.Module):
    """
    4 层 1D-CNN + 全连接分类器
    Input: (batch, 1, 6000)
    """
    def __init__(self, dropout=0.5):
        super().__init__()

        # Block 1: 6000 → ~1000
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=51, stride=3, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # → ~1000
        )
        # Block 2: ~1000 → ~250
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # → ~250
        )
        # Block 3: ~250 → ~125
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # → ~125
        )
        # Block 4: ~125 → 1 (global pooling)
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # → (256, 1)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x.squeeze(1)


# ═══════════════════════════════════════════
# 4. 训练函数
# ═══════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE).float()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += ((pred > 0.5).long() == y.long()).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE).float()
        pred = model(X)
        loss = criterion(pred, y)
        total_loss += loss.item() * X.size(0)
        probs = pred.cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(y.cpu().numpy().astype(int))
        correct += (preds == np.array(all_labels[-len(preds):])).sum()
        total += X.size(0)
    acc = correct / total
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return total_loss / total, acc, auc, np.array(all_labels), np.array(all_probs), np.array(all_preds)


# ═══════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("1D-CNN Apnea Detection — Training")
    log.info("=" * 60)

    # ── 5a. 加载数据 ──
    X, y, stats = load_all_data(DATA_DIR, TRAIN_RECORDS)

    # ── 5b. 划分训练/验证集 ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    log.info(f"Train: {len(y_train)} | Val: {len(y_val)}")

    train_ds = ECGDataset(X_train, y_train)
    val_ds = ECGDataset(X_val, y_val)

    # 加权采样处理类别不均衡
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── 5c. 构建模型 ──
    model = ApneaCNN(dropout=0.5).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {n_params:,}")

    # 类别加权损失
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(DEVICE)
    criterion = nn.BCELoss()  # 不用 pos_weight，已经用 WeightedRandomSampler
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    # ── 5d. 训练循环 ──
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, _, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dropout": 0.5, "input_len": MINUTE_N},
                "stats": stats,
                "history": history,
            }, MODEL_PATH)
            marker = " ★"
        else:
            patience_counter += 1

        log.info(f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
                 f"T_loss={train_loss:.4f} T_acc={train_acc:.3f} | "
                 f"V_loss={val_loss:.4f} V_acc={val_acc:.3f} V_auc={val_auc:.3f} | "
                 f"{elapsed:.1f}s{marker}")

        if patience_counter >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Best model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # ── 5e. 加载最佳模型做最终评估 ──
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, final_auc, y_true, y_prob, y_pred = evaluate(model, val_loader, criterion)

    log.info(f"\n{'='*40}")
    log.info(f"  Final Accuracy : {final_acc:.4f}")
    log.info(f"  Final AUC-ROC  : {final_auc:.4f}")
    log.info(f"{'='*40}")
    print("\n" + classification_report(y_true, y_pred, target_names=["Normal", "Apnea"]))

    # ═══════════════════════════════════════
    # 6. 可视化
    # ═══════════════════════════════════════

    # Fig 1: 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train", linewidth=1)
    axes[0].plot(history["val_loss"], label="Val", linewidth=1)
    axes[0].axvline(best_epoch-1, color="gray", linestyle="--", alpha=0.5, label=f"Best (ep {best_epoch})")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", linewidth=1)
    axes[1].plot(history["val_acc"], label="Val", linewidth=1)
    axes[1].axvline(best_epoch-1, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_auc"], label="Val AUC", linewidth=1, color="green")
    axes[2].axvline(best_epoch-1, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC-ROC")
    axes[2].set_title("AUC-ROC"); axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle(f"1D-CNN Training  |  Acc={final_acc:.3f}  AUC={final_auc:.3f}  Params={n_params:,}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    curve_path = OUTPUT_DIR / "step4_training_curves.png"
    fig.savefig(curve_path, dpi=150)
    log.info(f"Saved: {curve_path}")

    # Fig 2: 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Apnea"], yticklabels=["Normal", "Apnea"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  (Acc={final_acc:.3f}, AUC={final_auc:.3f})")
    fig.tight_layout()
    cm_path = OUTPUT_DIR / "step4_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    log.info(f"Saved: {cm_path}")

    # Fig 3: ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, color="tab:blue", label=f"AUC = {final_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="gray", alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color="tab:blue")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_path = OUTPUT_DIR / "step4_roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    log.info(f"Saved: {roc_path}")

    # ═══════════════════════════════════════
    # 7. 保存最终模型元数据
    # ═══════════════════════════════════════
    ckpt["accuracy"] = final_acc
    ckpt["auc"] = final_auc
    ckpt["n_params"] = n_params
    ckpt["class_counts"] = {"normal": int(class_counts[0]), "apnea": int(class_counts[1])}
    torch.save(ckpt, MODEL_PATH)
    log.info(f"Model saved → {MODEL_PATH}")

    log.info("\n✓ Training complete.")
    return model, history


if __name__ == "__main__":
    main()
