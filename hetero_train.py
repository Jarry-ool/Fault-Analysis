# -*- coding: utf-8 -*-
"""
hetero_train.py
第一阶段：无监督 VAE 训练（只用“正常”数据；验证也只看“正常”重构）
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import hetero_config as cfg
from hetero_data import TransformerVibrationDataset
from hetero_model import SpatialResNetVAE, loss_function

def train():
    device = torch.device(cfg.DEVICE)
    print(f"正在使用设备: {device.type}")

    train_set = TransformerVibrationDataset(cfg.TRAIN_DIR, only_normal=True, mode="train")
    val_set   = TransformerVibrationDataset(cfg.VAL_DIR,   only_normal=True, mode="val")

    print(f"Train(正常)样本组数 = {len(train_set)} | Val(正常)样本组数 = {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.LR)

    # best_path = cfg.CHECKPOINT_DIR / "model" / "best_model.pth"
    # if best_path.exists():
    #     print(f"检测到已有 best_model，正在从 {best_path} 继续训练...")
    #     state = torch.load(best_path, map_location=device)
    #     model.load_state_dict(state)
    # else:
    #     print("未检测到已有 best_model，将从随机初始化开始训练。")

    best_val = float("inf")
    for epoch in range(cfg.EPOCHS):
        # Beta 预热
        if cfg.BETA_WARMUP_EPOCHS > 0:
            beta = min(cfg.BETA_MAX, cfg.BETA_MAX * (epoch + 1) / cfg.BETA_WARMUP_EPOCHS)
        else:
            beta = cfg.BETA_MAX

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for x in pbar:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss, rec, kld = loss_function(recon, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            pbar.set_postfix(beta=f"{beta:.4f}", loss=f"{loss.item():.4f}")

        avg_train = train_loss / max(1, len(train_loader))

        # 验证（同样只看正常）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, rec, kld = loss_function(recon, x, mu, logvar, beta=beta)
                val_loss += float(loss.item())
        avg_val = val_loss / max(1, len(val_loader))
        print(f"\nEpoch {epoch+1} | Train Loss={avg_train:.4f} | Val Loss(正常)={avg_val:.4f}")

        # ========= 每 10 个 epoch 保存一次模型快照 =========
        if (epoch + 1) % 10 == 0:
            snapshot_path = cfg.CHECKPOINT_DIR / "model" / f"epoch_{epoch+1}.pth"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), snapshot_path)
            print(f"已保存快照模型: {snapshot_path}")

        # 保存最佳模型
        if avg_val < best_val:
            best_val = avg_val
            path = cfg.CHECKPOINT_DIR / "model" / "best_model.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            print(f"发现更优模型，已保存：{path}")

    torch.save(model.state_dict(), cfg.CHECKPOINT_DIR / "model" / "final_model.pth")
    print("训练完成。")

if __name__ == "__main__":
    train()
