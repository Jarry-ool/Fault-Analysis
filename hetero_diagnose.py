# -*- coding: utf-8 -*-
"""
hetero_diagnose.py
无监督异常诊断（无泄漏）
- 训练(正常) → 估计“正常重构误差分布 + 隐空间统计”
- 测试集：支持 test/正常 与 test/故障 拆分评估（若存在）
- 组合得分 S = α * Z(重构) + (1-α) * Z(马氏距离)，默认 α=0.6
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import hetero_config as cfg
from hetero_model import SpatialResNetVAE
from hetero_data import TransformerVibrationDataset

ALPHA = 0.6  # 重构分数权重（剩余给马氏距离）
USE_PERC = 0.975  # 正常集分位阈值（97.5%）
USE_SIGMA = False  # 若想切换到 3σ，可置 True

def _channel_weighted_l1(recon, inp, w=(0.4, 0.5, 0.1)):
    """通道加权 L1（默认 Zerone 通道权重更高）"""
    e0 = torch.mean(torch.abs(recon[:,0] - inp[:,0]), dim=[1,2])  # CWT
    e1 = torch.mean(torch.abs(recon[:,1] - inp[:,1]), dim=[1,2])  # Zerone 占位
    e2 = torch.mean(torch.abs(recon[:,2] - inp[:,2]), dim=[1,2])  # Context
    w0, w1, w2 = w
    return (w0*e0 + w1*e1 + w2*e2).detach().cpu().numpy()

def _collect_scores(model, loader, device):
    """返回：重构加权误差数组 rec_scores, 隐空间均值向量列表 mus"""
    model.eval()
    rec_scores = []
    latents = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Scoring"):
            imgs = imgs.to(device)
            recon, mu, _ = model(imgs)         # mu: [B, C=latent, 7, 7]
            rec = _channel_weighted_l1(recon, imgs, w=(0.4, 0.5, 0.1))
            rec_scores.append(rec)
            # 隐空间做全局平均池化 → [B, C]
            z = torch.mean(mu, dim=(2,3)).detach().cpu().numpy()
            latents.append(z)
    return np.concatenate(rec_scores), np.vstack(latents)

def _fit_mahalanobis(latents):
    """根据训练(正常)的隐向量拟合均值/协方差；返回 (mean, inv_cov)"""
    m = latents.mean(axis=0)
    cov = np.cov(latents.T) + 1e-6*np.eye(latents.shape[1])  # 稳定项
    inv = np.linalg.pinv(cov)
    return m, inv

def _mahalanobis(latents, mean, inv_cov):
    diff = latents - mean[None, :]
    dist2 = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    return np.sqrt(np.maximum(dist2, 0.0))

def _zscore(arr):
    mu, sd = arr.mean(), arr.std()
    sd = sd if sd > 1e-9 else 1.0
    return (arr - mu) / sd, mu, sd

def _decide_threshold(train_scores):
    if USE_SIGMA:
        mu, sd = train_scores.mean(), train_scores.std()
        thr = mu + 3*sd
    else:
        thr = np.quantile(train_scores, USE_PERC)
    return thr

def _maybe_build_loader(root: Path, only_normal: bool, mode: str):
    if not root.exists(): return None
    ds = TransformerVibrationDataset(root, mode=mode, only_normal=only_normal)
    if len(ds) == 0: return None
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

def diagnose():
    device = torch.device(cfg.DEVICE)
    print(f"使用设备: {device}")

    (cfg.CHECKPOINT_DIR/ "diagnosis" ).mkdir(exist_ok=True, parents=True)

    # 1) 加载模型（优先 best）
    model_path = cfg.CHECKPOINT_DIR / "model" / "best_model.pth"
    if not model_path.exists():
        model_path = cfg.CHECKPOINT_DIR / "model"/ "final_model.pth"
    print(f"加载模型: {model_path}")
    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2) 训练(正常)用于建立阈值与隐空间统计
    train_loader = _maybe_build_loader(Path(cfg.TRAIN_DIR), only_normal=True, mode="train")
    if train_loader is None:
        print("训练集(正常)为空，请检查 TRAIN_DIR。"); return

    print(">> 基于训练(正常)样本：提取重构误差与隐空间统计...")
    tr_rec, tr_mu = _collect_scores(model, train_loader, device)
    mean_mu, inv_cov = _fit_mahalanobis(tr_mu)
    tr_md = _mahalanobis(tr_mu, mean_mu, inv_cov)

    # 组合得分：先各自标准化，再线性融合
    z_rec, mu_rec, sd_rec = _zscore(tr_rec)
    z_md,  mu_md,  sd_md  = _zscore(tr_md)
    tr_score = ALPHA*z_rec + (1-ALPHA)*z_md
    thr = _decide_threshold(tr_score)

    print("------------------------------")
    print(f"训练(正常)样本数: {len(tr_score)}")
    print(f"重构分数: 均值={z_rec.mean():.3f}±{z_rec.std():.3f}  隐空间MD: 均值={z_md.mean():.3f}±{z_md.std():.3f}")
    print(f"异常阈值(组合分数): {thr:.3f}")
    print("------------------------------")

    # 3) 构造测试集 Loader
    test_root = Path(cfg.TEST_DIR)
    # 若存在明确子目录，则拆分评估
    test_norm_dir  = None
    test_fault_dir = None
    for sub in test_root.iterdir() if test_root.exists() else []:
        sl = str(sub).lower()
        if any(k in sl for k in ("正常", "normal")):
            test_norm_dir = sub
        if any(k in sl for k in ("故障", "异常", "fault", "abnormal")):
            test_fault_dir = sub

    # 如未拆分，则整包评估
    test_loader_all = _maybe_build_loader(test_root, only_normal=False, mode="test")

    results = []

    def eval_split(name, loader):
        if loader is None: return
        rec, mu = _collect_scores(model, loader, device)
        md = _mahalanobis(mu, mean_mu, inv_cov)
        # 用训练统计做标准化
        zrec = (rec - mu_rec) / (sd_rec if sd_rec>1e-9 else 1.0)
        zmd  = (md  - mu_md ) / (sd_md  if sd_md >1e-9 else 1.0)
        score = ALPHA*zrec + (1-ALPHA)*zmd
        pred_fault = (score < thr).astype(int)
        detected = int(pred_fault.sum())
        total = len(score)
        print(f"\r\n[{name}] 样本={total} | 判为异常={detected} ({detected/total*100:.2f}%)")
        results.append((name, score, pred_fault))

        # 直方图
        plt.figure(figsize=(8,4))
        plt.hist(score, bins=50, alpha=0.8, color='crimson', density=True)
        plt.axvline(thr, ls='--', c='k', label='Threshold')
        plt.title(f"{name} | Combined Score")
        plt.legend(); plt.tight_layout()
        out_png = cfg.CHECKPOINT_DIR / "diagnosis" / f"diagnosis_hist_{name}.png"
        plt.savefig(out_png); plt.close()

    if test_norm_dir is not None or test_fault_dir is not None:
        eval_split("test_normal", _maybe_build_loader(test_norm_dir,  only_normal=False, mode="test"))
        eval_split("test_fault",  _maybe_build_loader(test_fault_dir, only_normal=False, mode="test"))
    else:
        eval_split("test_all", test_loader_all)

    print("结果图片已保存到 outputs/ 目录。")

if __name__ == "__main__":
    diagnose()
