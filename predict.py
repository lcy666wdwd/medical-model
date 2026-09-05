#!/usr/bin/env python3
"""
Apnea Detection — Interactive Prediction Tool
==============================================
  双击运行，按提示输入即可。
  支持:
    - 预测单个 .dat 文件
    - 批量预测整个文件夹
    - 与真实标签对比（如果有 .apn 文件）
"""

import subprocess, sys, os, warnings, time, logging
warnings.filterwarnings("ignore")

# Windows GBK 编码兼容
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

# ── 自检依赖 ──
REQUIRED = {"numpy":"numpy", "wfdb":"wfdb", "torch":"torch", "pandas":"pandas"}
for mod, pkg in REQUIRED.items():
    try: __import__(mod)
    except ImportError:
        print(f"正在安装 {pkg} ...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from pathlib import Path

# ═══════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════
class ApneaCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 51, stride=3, padding=25), nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 25, stride=2, padding=12), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 11, padding=5), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        return self.classifier(x).squeeze(1)

FS = 100
MINUTE_N = FS * 60

# ═══════════════════════════════════════════
# 加载模型
# ═══════════════════════════════════════════
def load_model():
    candidates = [
        Path(__file__).parent / "apnea-ecg-test-label" / "cnn_model.pt",
        Path(__file__).parent / "cnn_model.pt",
        Path.cwd() / "cnn_model.pt",
    ]
    for p in candidates:
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            model = ApneaCNN(dropout=ckpt.get("config", {}).get("dropout", 0.5))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            print(f"[OK] 模型已加载: {p}")
            print(f"  训练时 Accuracy: {ckpt.get('accuracy','-'):.2%}")
            print(f"  训练时 AUC:      {ckpt.get('auc','-'):.4f}")
            return model
    print("[X] 找不到 cnn_model.pt，请确保模型文件在当前目录下")
    return None

# ═══════════════════════════════════════════
# 预测
# ═══════════════════════════════════════════
def predict(model, file_path):
    """file_path: .dat 文件的完整路径（不含扩展名）或包含扩展名"""
    file_path = str(file_path)
    # 去掉 .dat 后缀
    if file_path.endswith(".dat"):
        file_path = file_path[:-4]

    print(f"\n正在分析: {file_path} ...")
    sig, fields = wfdb.rdsamp(file_path)
    ecg = sig[:, 0].astype(np.float64)
    total_h = len(ecg) / fields["fs"] / 3600
    print(f"  信号时长: {total_h:.1f} 小时 ({len(ecg)} 采样点 @ {fields['fs']}Hz)")

    n_min = len(ecg) // MINUTE_N
    segments = []
    for m in range(n_min):
        seg = ecg[m*MINUTE_N : (m+1)*MINUTE_N].copy()
        s = seg.std()
        segments.append((seg - seg.mean()) / s if s > 1e-8 else np.zeros_like(seg))

    X = np.stack(segments).astype(np.float32)
    X_t = torch.from_numpy(X).unsqueeze(1)
    with torch.no_grad():
        probs = model(X_t).numpy()
    preds = (probs > 0.5).astype(int)

    n_apnea = int(preds.sum())
    print(f"  总分钟数: {n_min}")
    print(f"  呼吸暂停: {n_apnea} 分钟 ({n_apnea/n_min*100:.1f}%)")
    print(f"  正常呼吸: {n_min-n_apnea} 分钟 ({(n_min-n_apnea)/n_min*100:.1f}%)")

    # 尝试读取标签对比
    gt = None
    try:
        ann = wfdb.rdann(file_path, "apn")
        gt = np.array([1 if s == "A" else 0 for s in ann.symbol])[:n_min]
        acc = (preds[:len(gt)] == gt).mean()
        print(f"  与标签对比准确率: {acc:.2%}")
    except: pass

    # 保存 CSV
    out = Path(file_path).parent / (Path(file_path).name + "_cnn_prediction.csv")
    df = pd.DataFrame({"minute": np.arange(1, n_min+1),
                       "pred_label": preds,
                       "apnea_prob": np.round(probs, 4)})
    if gt is not None:
        df["true_label"] = gt
    df.to_csv(out, index=False)
    print(f"  结果已保存: {out}")
    return df


# ═══════════════════════════════════════════
# 交互菜单
# ═══════════════════════════════════════════
def main():
    print("=" * 55)
    print("  呼吸暂停检测 — CNN 模型预测工具")
    print("=" * 55)

    model = load_model()
    if model is None:
        input("\n按 Enter 退出...")
        return

    while True:
        print("\n" + "-" * 55)
        print("请选择:")
        print("  1. 预测单个 .dat 文件（输入完整路径，不含 .dat 后缀）")
        print("  2. 预测整个文件夹中所有 .dat 文件")
        print("  3. 预测当前数据集中的某条记录（如 a01, x01）")
        print("  q. 退出")
        print("-" * 55)
        choice = input("> ").strip()

        if choice.lower() == "q":
            print("再见!")
            break

        elif choice == "1":
            path = input("请输入 .dat 文件路径（可不写 .dat 后缀）: ").strip().strip('"')
            if path:
                predict(model, path)

        elif choice == "2":
            folder = input("请输入文件夹路径: ").strip().strip('"')
            if folder and os.path.isdir(folder):
                files = sorted([f for f in os.listdir(folder) if f.endswith(".dat")])
                print(f"找到 {len(files)} 个 .dat 文件")
                for f in files:
                    try:
                        predict(model, os.path.join(folder, f))
                    except Exception as e:
                        print(f"  [X] {f}: {e}")
            else:
                print("文件夹不存在!")

        elif choice == "3":
            rec = input("请输入记录名 (如 a01, x15): ").strip()
            if rec:
                data_dir = Path(__file__).parent / "apnea-ecg-test-label" / "apnea-ecg"
                rec_path = data_dir / rec
                if rec_path.with_suffix(".dat").exists():
                    try:
                        predict(model, str(rec_path))
                    except Exception as e:
                        print(f"  [X] 预测失败: {e}")
                else:
                    print(f"记录 {rec} 不存在于 {data_dir}")

        else:
            print("无效选择，请重新输入")

    input("\n按 Enter 退出...")


if __name__ == "__main__":
    main()
