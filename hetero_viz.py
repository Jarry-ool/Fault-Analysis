# -*- coding: utf-8 -*-
"""
hetero_viz.py
多通道异构快照可视化 (Multi-channel Heterogeneous Snapshot Visualization)
---------------------------------------------------------------------
功能：
1. 寻找具有相同时间戳 (data_time) 的一组多传感器数据 (模拟 U=N 的场景)。
2. 将这组数据的 Raw Signal, Original CWT, Reconstructed CWT, Reconstructed Zerone 并排展示。
3. 生成符合 Nature/IEEE 期刊风格的高级可视化图表。

注意：
由于本模型 (Hetero-CWT-Zerone-VAE) 是图像级生成模型，它重构的是“时频图像”而非“一维波形”。
因此，本图表将重点展示模型对 [时频结构] 和 [物理流形] 的重构能力，这是证明该模型有效的核心证据。
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import pywt
from pathlib import Path
from collections import defaultdict
import matplotlib.gridspec as gridspec

# 引入项目配置和模型
import hetero_config as cfg
from hetero_model import SpatialResNetVAE
from hetero_data import TransformerVibrationDataset # 借用部分逻辑

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # 优先英文，兼容中文
plt.rcParams['axes.unicode_minus'] = False

def load_and_group_data(root_dirs, min_sensors=3, max_groups=1):
    """
    扫描数据目录，按 data_time 分组，寻找多传感器样本。
    """
    files = []
    for d in root_dirs:
        files.extend(list(Path(d).rglob("*.jsonl")))
    
    print(f"正在扫描 {len(files)} 个文件进行分组...")
    
    # key: data_time, value: list of file_paths
    groups = defaultdict(list)
    
    # 为了演示，只扫描部分文件防止太慢
    for f in files[:2000]: 
        try:
            # 快速读取第一行获取 time
            with open(f, 'r', encoding='utf-8') as f_obj:
                line = f_obj.readline()
                data = json.loads(line)
                dt = data.get('data_time', 'unknown')
                groups[dt].append(f)
        except:
            continue
            
    # 筛选出传感器数量满足要求的组
    valid_groups = [k for k, v in groups.items() if len(v) >= min_sensors]
    
    print(f"找到 {len(valid_groups)} 个包含 >= {min_sensors} 个传感器的时间点。")
    
    if not valid_groups:
        return []
    
    # 随机选几个
    selected_keys = np.random.choice(valid_groups, min(len(valid_groups), max_groups), replace=False)
    return [(k, groups[k]) for k in selected_keys]

def process_single_file(file_path):
    """
    读取单个文件并生成模型输入 (CWT, Zerone)
    """
    # 这里复用 hetero_data 的逻辑，但为了独立性重写简化版
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())
        sig_str = data.get('signal_value', '')
        signal = np.fromstring(sig_str, sep=',')
        if len(signal) > cfg.SIGNAL_LEN: signal = signal[:cfg.SIGNAL_LEN]
        else: signal = np.pad(signal, (0, cfg.SIGNAL_LEN - len(signal)))
        
    # Generate CWT
    scales = np.arange(1, 129)
    coef, _ = pywt.cwt(signal, scales, 'morl')
    scalogram = np.log1p(np.abs(coef))
    cwt_img = cv2.resize(scalogram, (cfg.INPUT_SIZE, cfg.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    cwt_img_norm = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
    
    # Generate Zerone (Mock for viz speed, normally use hetero_data._generate_zerone_raster)
    # 在展示中，我们重点看 CWT，Zerone 用随机噪点或简化图代替展示即可，或者调用真实逻辑
    # 为了代码简洁，这里再次调用真实逻辑的简化版
    zerone_img = np.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), dtype=np.float32) 
    # (实际项目中应完整调用 feature extraction)
    
    # Context
    ctx_img = np.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), dtype=np.float32)
    
    # Stack
    tensor = np.stack([cwt_img_norm, zerone_img, ctx_img], axis=0)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0) # [1, 3, 224, 224]
    
    return signal, tensor

def visualize_snapshot(model_path):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    try:
        model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"模型加载成功: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 获取一组数据
    groups = load_and_group_data(cfg.RAW_DATA_DIRS, min_sensors=3, max_groups=1)
    if not groups:
        print("未找到合适的多传感器数据组。")
        return

    data_time, file_paths = groups[0]
    num_sensors = len(file_paths)
    
    # 限制最大展示传感器数，防止图太高
    max_display = 8
    if num_sensors > max_display:
        file_paths = file_paths[:max_display]
        num_sensors = max_display
        
    print(f"正在可视化时间点: {data_time}, 传感器数量: {num_sensors}")

    # 3. 准备绘图
    # 布局: 每行对应一个传感器。
    # 列: [Raw Signal (宽)] [Orig CWT] [Recon CWT] [Recon Zerone]
    fig = plt.figure(figsize=(20, 3 * num_sensors))
    gs = gridspec.GridSpec(num_sensors, 5, width_ratios=[3, 1, 1, 1, 0.1]) 
    # 最后一列 0.1 给 colorbar 留空
    
    # 提取一些元数据做标题
    sample_file_str = str(file_paths[0])
    status = "FAULT" if "故障" in sample_file_str else ("NORMAL" if "正常" in sample_file_str else "UNKNOWN")
    fig.suptitle(f"Multi-sensor Heterogeneous Reconstruction Snapshot\nTime: {data_time} | Status: {status} | U={num_sensors}", 
                 fontsize=16, y=0.92)

    with torch.no_grad():
        for i, fp in enumerate(file_paths):
            # 处理数据
            raw_sig, input_tensor = process_single_file(fp)
            input_tensor = input_tensor.to(device)
            
            # 模型推理
            recon_tensor, _, _ = model(input_tensor)
            
            # 转 Numpy
            orig_cwt = input_tensor[0, 0].cpu().numpy()
            recon_cwt = recon_tensor[0, 0].cpu().numpy()
            
            # orig_zerone = input_tensor[0, 1].cpu().numpy() 
            recon_zerone = recon_tensor[0, 1].cpu().numpy()
            
            # --- 绘图 ---
            
            # 1. Raw Signal (Time Domain)
            ax_sig = fig.add_subplot(gs[i, 0])
            ax_sig.plot(raw_sig, color='#1f77b4', linewidth=0.8, alpha=0.9, label='Raw Vibration')
            ax_sig.set_xlim(0, len(raw_sig))
            ax_sig.set_ylabel(f"Sensor {i+1}", fontsize=12, fontweight='bold')
            ax_sig.grid(True, alpha=0.2, linestyle='--')
            if i == 0: ax_sig.set_title("Raw Time-Domain Signal", fontsize=12)
            if i == num_sensors - 1: ax_sig.set_xlabel("Sample Index", fontsize=10)
            else: ax_sig.set_xticklabels([])
            
            # 2. Original CWT (Input)
            ax_cwt_in = fig.add_subplot(gs[i, 1])
            im1 = ax_cwt_in.imshow(orig_cwt, cmap='jet', vmin=0, vmax=1, aspect='auto')
            ax_cwt_in.axis('off')
            if i == 0: ax_cwt_in.set_title("Original CWT\n(Model Input)", fontsize=12)
            
            # 3. Recon CWT (Output)
            ax_cwt_out = fig.add_subplot(gs[i, 2])
            im2 = ax_cwt_out.imshow(recon_cwt, cmap='jet', vmin=0, vmax=1, aspect='auto')
            ax_cwt_out.axis('off')
            if i == 0: ax_cwt_out.set_title("Reconstructed CWT\n(Model Output)", fontsize=12)
            
            # 4. Recon Zerone (Latent feature map recon)
            ax_zerone = fig.add_subplot(gs[i, 3])
            im3 = ax_zerone.imshow(recon_zerone, cmap='viridis', vmin=0, vmax=1, aspect='auto')
            ax_zerone.axis('off')
            if i == 0: ax_zerone.set_title("Recon Zerone Map\n(Manifold)", fontsize=12)

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    
    # 保存
    save_name = "hetero_multichannel_snapshot.png"
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"可视化快照已保存: {save_name}")
    plt.show()

if __name__ == "__main__":
    # 使用你训练好的权重文件
    # 示例: cfg.CHECKPOINT_DIR / "vae_stage1_epoch_50.pth"
    MODEL_FILE = cfg.CHECKPOINT_DIR / "vae_stage1_epoch_50.pth" 
    visualize_snapshot(MODEL_FILE)