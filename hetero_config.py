# -*- coding: utf-8 -*-
"""
hetero_config.py
项目配置文件
"""
from pathlib import Path

# ================= 路径配置 =================
# 假设数据根目录，请根据实际情况修改
# 这里模拟从 config.py 读取路径的逻辑，但为了演示，指向相对路径
DATA_ROOT = Path(r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016\train")

# 假设的原始数据路径结构（参考了你提供的 config.py）
# 在 Stage 1 (无监督) 中，我们通常会混合所有数据，或者只使用训练集目录
# 这里我们定义一个列表，包含所有需要读取的 jsonl 文件所在的文件夹
RAW_DATA_DIRS = [
    DATA_ROOT # DATA_ROOT / "val", # 如果想利用验证集做无监督预训练也可以加入
]

# 模型保存路径
CHECKPOINT_DIR = Path("./outputs")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ================= 物理/信号参数 =================
FS = 8192              # 采样率 (Hz)
SIGNAL_LEN = 8192      # 信号长度 (1秒)
INPUT_SIZE = 224       # 模型输入图像尺寸 (224x224)

# ================= VAE模型参数 =================
LATENT_CHANNELS = 64   # 空间隐变量的通道数 (Spatial Latent Channels)
BETA_INIT = 0.0        # KL散度权重的初始值
BETA_MAX = 0.01        # KL散度权重的最大值
BETA_WARMUP_EPOCHS = 20 # Beta退火的预热轮数

# ================= 训练超参数 =================
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda"        # 或 "cpu"
SEED = 42

# ================= Zerone特征参数 =================
# 定义 Zerone 特征的维度 (参考文档中的1200维)
FEAT_DIM_TIME = 15
FEAT_DIM_STFT = 127
FEAT_DIM_PSD = 1050
FEAT_DIM_HF = 8
TOTAL_ZERONE_DIM = FEAT_DIM_TIME + FEAT_DIM_STFT + FEAT_DIM_PSD + FEAT_DIM_HF  # 1200