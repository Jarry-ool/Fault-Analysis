# -*- coding: utf-8 -*-
"""
hetero_train.py
第一阶段：无监督 VAE 训练脚本
目标：学习正常数据的流形分布，不涉及故障分类。
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

# 引入自定义模块
import hetero_config as cfg
from hetero_data import TransformerVibrationDataset
from hetero_model import SpatialResNetVAE, loss_function

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def train():
    # 1. 配置设备
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    
    # 2. 准备数据集
    # 注意：这里我们传入 RAW_DATA_DIRS，不区分文件夹里的类别
    # 按照“两阶段”思想，第一阶段通常使用所有可用的历史数据（主要为正常数据）
    dataset = TransformerVibrationDataset(
        root_dirs=cfg.RAW_DATA_DIRS, 
        mode='train'
    )
    
    if len(dataset) == 0:
        print("错误：未找到数据文件，请检查 hetero_config.py 中的 RAW_DATA_DIRS 路径。")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # 3. 初始化模型
    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    
    # 4. 开始训练循环
    model.train()
    print(f">>> 开始第一阶段无监督训练 (Total Epochs: {cfg.EPOCHS})")
    ensure_dir("outputs/model")  # 确保模型输出目录存在

    for epoch in range(cfg.EPOCHS):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        # KL 散度退火策略 (Beta Annealing)
        # 前20轮 beta 从 0 线性增加到 0.01，防止 Posterior Collapse
        if epoch < cfg.BETA_WARMUP_EPOCHS:
            beta = cfg.BETA_MAX * (epoch / cfg.BETA_WARMUP_EPOCHS)
        else:
            beta = cfg.BETA_MAX
            
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        # --- 核心训练 Loop ---
        for batch_idx, images in enumerate(progress_bar):
            # images shape: [B, 3, 224, 224]
            # 注意：这里 Dataset __getitem__ 只返回了 images，没有 label
            # 从而从代码层面物理隔绝了标签信息
            
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            recon_images, mu, logvar = model(images)
            
            # 计算损失
            loss, recon_loss, kld_loss = loss_function(recon_images, images, mu, logvar, beta=beta)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录统计
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "Loss": loss.item(), 
                "Recon": recon_loss.item(), 
                "KL": kld_loss.item(),
                "Beta": beta
            })
            
        # 打印 Epoch 总结
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成. Avg Loss: {avg_loss:.4f} (Recon: {total_recon/len(dataloader):.4f}, KL: {total_kld/len(dataloader):.4f})")
        
        # 保存权重
        if (epoch + 1) % 10 == 0:
            save_path = cfg.CHECKPOINT_DIR / "model" / f"vae_stage1_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存至: {save_path}")

    print("第一阶段训练完成。")

if __name__ == "__main__":
    train()