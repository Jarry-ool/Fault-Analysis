# -*- coding: utf-8 -*-
"""
hetero_data.py
数据加载与预处理核心模块
实现了文档中的 Channel 0 (CWT) 和 Channel 1 (Zerone Raster) 生成逻辑。
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import pywt
from scipy.signal import welch, stft
from scipy.stats import kurtosis
import glob

# 引入配置
import hetero_config as cfg

class TransformerVibrationDataset(Dataset):
    """
    变压器振动数据集加载器 (用于 Stage 1 无监督训练)
    特点：
    1. 遍历指定目录下的所有 jsonl 文件。
    2. 解析振动信号。
    3. 生成 3通道张量: [CWT, Zerone, Context]。
    4. __getitem__ 不返回标签 (Label)，只返回数据 Tensor，防止模型学到标签。
    """
    def __init__(self, root_dirs, mode='train'):
        self.files = []
        self.mode = mode
        
        # 1. 扫描所有数据文件 (不区分文件夹名字代表的类别)
        for d in root_dirs:
            # 递归查找所有 .jsonl 文件
            self.files.extend(list(Path(d).rglob("*.jsonl")))
            
        print(f"[{mode}] 共加载文件: {len(self.files)} 个")
        
        # 初始化 Zerone 特征的全局统计量 (用于归一化)
        # 在实际工程中，应该先遍历一遍数据计算 min/max 并保存
        # 这里为了代码可运行，使用动态计算或预设值，建议先运行一个独立的脚本计算 global_min/max
        self.global_min = -10.0 # 预设占位值
        self.global_max = 10.0  # 预设占位值

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # 1. 读取信号
        signal = self._load_signal(file_path)
        
        # 2. 生成 CWT 时频图 (Channel 0)
        cwt_img = self._generate_cwt(signal)
        
        # 3. 生成 Zerone 栅格图 (Channel 1)
        zerone_img = self._generate_zerone_raster(signal)
        
        # 4. 生成 工况嵌入图 (Channel 2)
        # 暂时用全0矩阵代替，如果有负载电流数据可在此填入归一化后的值
        context_img = np.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), dtype=np.float32)
        
        # 5. 堆叠成 3通道 Tensor (3, 224, 224)
        # 注意：这里我们不返回 label，严格遵守无监督要求
        tensor = np.stack([cwt_img, zerone_img, context_img], axis=0)
        return torch.tensor(tensor, dtype=torch.float32)

    def _load_signal(self, path):
        """解析 JSONL 读取 raw signal"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                line = f.readline() # 只读第一行作为一个样本示例
                data = json.loads(line)
                # 假设 json 结构里 key 是 'signal_value'，是一个逗号分隔的字符串
                # 根据你提供的 config.py 说明适配
                sig_str = data.get('signal_value', '')
                if not sig_str:
                    return np.zeros(cfg.SIGNAL_LEN)
                
                sig = np.fromstring(sig_str, sep=',')
                
                # 截断或补零到固定长度
                if len(sig) > cfg.SIGNAL_LEN:
                    sig = sig[:cfg.SIGNAL_LEN]
                else:
                    sig = np.pad(sig, (0, cfg.SIGNAL_LEN - len(sig)))
                return sig
        except Exception as e:
            # print(f"Error reading {path}: {e}")
            return np.zeros(cfg.SIGNAL_LEN)

    def _generate_cwt(self, signal):
        """
        生成连续小波变换 (CWT) 时频图
        参考文档：使用 Morlet 小波，对数尺度
        """
        # 降采样以加快计算 (可选，视性能而定)
        # signal = signal[::2] 
        
        scales = np.arange(1, 129)
        coef, _ = pywt.cwt(signal, scales, 'morl')
        scalogram = np.abs(coef)
        
        # 对数增强
        scalogram = np.log1p(scalogram)
        
        # 调整尺寸到 224x224
        img = cv2.resize(scalogram, (cfg.INPUT_SIZE, cfg.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Min-Max 归一化到 [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    def _generate_zerone_raster(self, signal):
        """
        生成 Zerone 特征栅格图
        步骤：提取1200维特征 -> 归一化 -> Reshape -> Resize
        """
        # 1. 提取特征 (简化版实现)
        feats = self._extract_1200_features(signal)
        
        # 2. 归一化 (使用类内保存的全局统计量，实际应用需预先计算)
        # 这里做局部归一化演示，严谨的方案需使用 self.global_min/max
        feats_norm = (feats - feats.min()) / (feats.max() - feats.min() + 1e-8)
        
        # 3. 栅格化 (Raster Layout)
        # 1200 维 -> 40x30 矩阵
        grid = feats_norm.reshape(40, 30)
        
        # 4. 放大到 224x224 (使用最近邻插值保持锐利)
        img = cv2.resize(grid, (cfg.INPUT_SIZE, cfg.INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        return img

    def _extract_1200_features(self, x):
        """
        提取 1200 维物理统计特征
        包含: Time(15) + STFT(127) + PSD(1050) + HF(8)
        """
        # --- Time Domain (15) ---
        mu = np.mean(x)
        rms = np.sqrt(np.mean(x**2))
        kurt = kurtosis(x)
        # ... (此处省略其余时域特征的具体计算，用占位符填充以保证维度对齐)
        time_feats = np.array([mu, rms, kurt] + [0]*12) 
        
        # --- PSD (1050) ---
        f, p = welch(x, fs=cfg.FS, nperseg=2048)
        # 截取前 1050 个点 (简化逻辑，实际需按文档分频段聚合)
        if len(p) >= 1050:
            psd_feats = p[:1050]
        else:
            psd_feats = np.pad(p, (0, 1050 - len(p)))
            
        # --- STFT (127) ---
        _, _, Zxx = stft(x, fs=cfg.FS, nperseg=256)
        stft_mean = np.mean(np.abs(Zxx), axis=1)
        if len(stft_mean) >= 127:
            stft_feats = stft_mean[1:128] # 去直流
        else:
            stft_feats = np.pad(stft_mean, (0, 127 - len(stft_mean)))
            
        # --- HF (8) ---
        hf_feats = np.zeros(8) # 占位
        
        # 拼接
        total = np.concatenate([time_feats, stft_feats, psd_feats, hf_feats])
        
        # 确保正好 1200 维
        if len(total) > 1200:
            total = total[:1200]
        elif len(total) < 1200:
            total = np.pad(total, (0, 1200 - len(total)))
            
        return total

# 单元测试
if __name__ == "__main__":
    # 测试数据加载
    import torch
    print("Test Dataset...")
    # ds = TransformerVibrationDataset(["./data/test_dir"])