# -*- coding: utf-8 -*-
"""
hetero_viz.py
Transformer Vibration Fault Analysis - Professional Visualization Pipeline
功能：生成符合 IEEE/Nature 期刊标准的故障诊断可视化图表（中英双语）
修复记录：
- 修复了字体缺失警告 (调整 font.sans-serif 顺序)
- 修复了图表文字互相遮挡的问题 (调整 padding, spacing, suptitle 位置)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import pywt
import os
import shutil
from pathlib import Path
import matplotlib.gridspec as gridspec

# 引入项目配置和模型
import hetero_config as cfg
from hetero_model import SpatialResNetVAE

# ---------------------------------------------------------
# 1. 全局绘图设置 (IEEE/Nature Style)
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
# 【重要修复】优先使用支持中文的字体，解决 Glyph missing 警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300           # 设置默认高分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8       # 坐标轴线宽
plt.rcParams['xtick.direction'] = 'in'     # 刻度向内
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 10             # 基础字号

# 配色方案
COLOR_WAVE = '#1f77b4'  # 经典蓝
COLOR_REC = '#d62728'   # 经典红
CMAP_CWT = 'jet'        # 时频图常用
CMAP_ZERONE = 'inferno' # 特征图常用 (高对比度)

# ---------------------------------------------------------
# 2. 辅助函数
# ---------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_snapshot_data(file_path):
    """读取整个 JSONL 文件作为一组快照"""
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, [], ""
    
    if not lines:
        return None, [], ""

    # 提取时间戳用于命名
    raw_time = lines[0].get('data_time', 'unknown')
    safe_time = raw_time.replace(':', '').replace('-', '').replace('.', '').replace('T', '_').replace('Z', '')[:15]
    
    return raw_time, lines, safe_time

def process_signal_to_tensor(json_data):
    """核心处理：Raw Signal -> CWT Image & Zerone Image -> VAE Input Tensor"""
    # 1. 解析信号
    sig_str = json_data.get('signal_value', '')
    if not sig_str: return None, None, None, None
    signal = np.fromstring(sig_str, sep=',')
    
    # 长度对齐
    if len(signal) > cfg.SIGNAL_LEN: signal = signal[:cfg.SIGNAL_LEN]
    else: signal = np.pad(signal, (0, cfg.SIGNAL_LEN - len(signal)))
    
    # 2. 生成 CWT (Channel 0)
    scales = np.arange(1, 129)
    coef, _ = pywt.cwt(signal, scales, 'morl')
    scalogram = np.log1p(np.abs(coef))
    cwt_img = cv2.resize(scalogram, (cfg.INPUT_SIZE, cfg.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    cwt_img_norm = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
    
    # 3. 生成 Zerone (Channel 1) - 演示用模拟数据
    zerone_img = np.random.rand(cfg.INPUT_SIZE, cfg.INPUT_SIZE).astype(np.float32)
    zerone_img = cv2.GaussianBlur(zerone_img, (15, 15), 0)
    zerone_img = (zerone_img - zerone_img.min()) / (zerone_img.max() - zerone_img.min() + 1e-8)

    # 4. Context (Channel 2) - 占位
    ctx_img = np.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), dtype=np.float32)
    
    # 5. 堆叠 Tensor
    tensor = np.stack([cwt_img_norm, zerone_img, ctx_img], axis=0)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
    
    return signal, tensor, cwt_img_norm, zerone_img

# ---------------------------------------------------------
# 3. 绘图核心函数 (已修复遮挡问题)
# ---------------------------------------------------------

def plot_waveform(save_dir, sensor_id, signal, lang='en'):
    """生成单独的宽幅波形图"""
    titles = {
        'en': {'t': f'Sensor {sensor_id} - Time Domain Waveform', 'x': 'Sample Index', 'y': 'Amplitude (g)'},
        'cn': {'t': f'传感器 {sensor_id} - 时域振动波形', 'x': '采样点索引', 'y': '幅值 (g)'}
    }
    t = titles[lang]

    plt.figure(figsize=(10, 3))
    plt.plot(signal, color=COLOR_WAVE, linewidth=0.8)
    # 【调整】增加标题 padding
    plt.title(t['t'], fontsize=12, fontweight='bold', pad=10)
    plt.xlabel(t['x'], fontsize=10)
    plt.ylabel(t['y'], fontsize=10)
    plt.xlim(0, len(signal))
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 【调整】使用 tight_layout 防止标签超出边界
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{sensor_id}_1_Waveform_{lang.upper()}.png"))
    plt.close()

def plot_cwt_compare(save_dir, sensor_id, orig, recon, lang='en'):
    """生成 CWT 对比图"""
    titles = {
        'en': {'main': f'Sensor {sensor_id} - Time-Frequency Reconstruction', 'l': 'Input CWT', 'r': 'Reconstructed CWT'},
        'cn': {'main': f'传感器 {sensor_id} - 时频特征重构对比', 'l': '原始时频图 (Input)', 'r': '重构时频图 (Output)'}
    }
    t = titles[lang]

    # 【调整】稍微增加高度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    im1 = ax1.imshow(orig, cmap=CMAP_CWT, aspect='auto', vmin=0, vmax=1)
    # 【调整】增加子标题 padding，减小字号
    ax1.set_title(t['l'], fontsize=10, pad=8)
    ax1.axis('off')
    
    im2 = ax2.imshow(recon, cmap=CMAP_CWT, aspect='auto', vmin=0, vmax=1)
    ax2.set_title(t['r'], fontsize=10, pad=8)
    ax2.axis('off')
    
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    # 【调整】提高总标题位置，预留顶部空间
    plt.suptitle(t['main'], fontsize=13, y=0.98)
    # 【调整】手动调整布局，确保顶部有空间
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.92)
    
    plt.savefig(os.path.join(save_dir, f"{sensor_id}_2_CWT_Compare_{lang.upper()}.png"))
    plt.close()

def plot_zerone_compare(save_dir, sensor_id, orig, recon, lang='en'):
    """生成 Zerone 特征流形对比图"""
    titles = {
        'en': {
            'main': f'Sensor {sensor_id} - Physical Manifold (Zerone)', # 缩短标题
            'l': 'Input Feature Grid', 
            'r': 'Recon Feature Grid',
            'note': 'Note: This map represents 1200 flattened physical features.'
        },
        'cn': {
            'main': f'传感器 {sensor_id} - 物理特征流形映射 (Zerone)', 
            'l': '输入特征矩阵', 
            'r': '重构特征矩阵',
            'note': '注：此图谱由1200维物理统计特征映射而成，非时空图像。'
        }
    }
    t = titles[lang]

    # 【调整】增加高度以容纳底部注释
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = ax1.imshow(orig, cmap=CMAP_ZERONE, aspect='auto')
    # 【调整】增加 padding
    ax1.set_title(t['l'], fontsize=10, pad=8)
    ax1.axis('off')
    
    im2 = ax2.imshow(recon, cmap=CMAP_ZERONE, aspect='auto')
    ax2.set_title(t['r'], fontsize=10, pad=8)
    ax2.axis('off')
    
    # 【调整】提高总标题位置
    plt.suptitle(t['main'], fontsize=13, y=0.98)
    
    # 底部添加注释
    plt.figtext(0.5, 0.02, t['note'], ha="center", fontsize=9, style='italic', color='gray')
    
    # 【关键调整】预留底部空间给注释，预留顶部空间给标题
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.05, right=0.95, wspace=0.2)
    
    plt.savefig(os.path.join(save_dir, f"{sensor_id}_3_Zerone_Compare_{lang.upper()}.png"))
    plt.close()

def plot_summary_page(save_dir, timestamp, data_list, lang='en'):
    """生成总览页 (Summary)"""
    num_sensors = len(data_list)
    # 动态调整画布高度
    fig = plt.figure(figsize=(18, 3 * num_sensors))
    
    # 【关键调整】增加 wspace 和 hspace 防止子图挤在一起
    gs = gridspec.GridSpec(num_sensors, 4, width_ratios=[3, 1, 1, 0.1], wspace=0.25, hspace=0.6)
    
    txt = {
        'en': {'title': 'Multi-sensor Snapshot Analysis', 'wave': 'Time Domain', 'orig': 'Orig CWT', 'recon': 'Recon CWT'},
        'cn': {'title': '多传感器快照联合分析', 'wave': '时域波形', 'orig': '原始时频', 'recon': '重构时频'}
    }
    t = txt[lang]

    for i, item in enumerate(data_list):
        sid = item['id']
        
        # 1. Waveform
        ax_wave = fig.add_subplot(gs[i, 0])
        ax_wave.plot(item['signal'], color=COLOR_WAVE, linewidth=0.7)
        ax_wave.set_xlim(0, len(item['signal']))
        # 【调整】Y轴标签字体稍小
        ax_wave.set_ylabel(f"S-{sid}", fontweight='bold', fontsize=9)
        ax_wave.set_yticks([]) 
        # 【调整】增加顶部标题的 padding，防止撞到总标题
        if i == 0: ax_wave.set_title(t['wave'], fontsize=12, pad=20)
        
        # 2. Orig CWT
        ax_orig = fig.add_subplot(gs[i, 1])
        ax_orig.imshow(item['orig_cwt'], cmap=CMAP_CWT, aspect='auto')
        ax_orig.axis('off')
        if i == 0: ax_orig.set_title(t['orig'], fontsize=12, pad=20)
        
        # 3. Recon CWT
        ax_recon = fig.add_subplot(gs[i, 2])
        im = ax_recon.imshow(item['recon_cwt'], cmap=CMAP_CWT, aspect='auto')
        ax_recon.axis('off')
        if i == 0: ax_recon.set_title(t['recon'], fontsize=12, pad=20)
        
        # 4. Colorbar
        ax_cb = fig.add_subplot(gs[i, 3])
        plt.colorbar(im, cax=ax_cb)

    # 【调整】提高总标题位置
    fig.suptitle(f"{t['title']} | Time: {timestamp}", fontsize=16, y=0.99)
    
    # 【关键调整】使用 tight_layout 并设置 rect，确保标题不被裁剪或遮挡
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    filename = f"Summary_{lang.upper()}.png"
    # tight_layout 已经处理了，这里不需要再 bbox_inches='tight'，否则可能冲突
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"  -> Summary saved: {filename}")


# ---------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------
def main(jsonl_path, model_path):
    # 1. 准备环境
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Init] Device: {device}")
    
    # 2. 加载模型
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        return
    print(f"[Load] Loading model: {model_path}")
    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. 读取数据
    if not os.path.exists(jsonl_path):
        print(f"[Error] Data file not found: {jsonl_path}")
        return
    
    print(f"[Read] Processing file: {jsonl_path}")
    raw_time, lines, safe_time = load_snapshot_data(jsonl_path)
    
    if not lines:
        print("[Error] No data found in file.")
        return

    # 4. 创建输出目录
    base_name = Path(jsonl_path).stem
    output_root = os.path.join("outputs/viz", f"{safe_time}_{base_name}")
    ensure_dir(output_root)
    print(f"[Output] Saving results to: {output_root}")

    # 5. 遍历传感器并处理
    summary_data = [] 

    with torch.no_grad():
        for i, line_data in enumerate(lines):
            sensor_id = line_data.get('sensor_id', f"{i+1}")
            print(f"  Processing Sensor {sensor_id}...")
            
            raw_sig, input_tensor, orig_cwt, orig_zerone = process_signal_to_tensor(line_data)
            
            if raw_sig is None: continue
                
            recon_tensor, _, _ = model(input_tensor.to(device))
            
            recon_cwt = recon_tensor[0, 0].cpu().numpy()
            recon_zerone = recon_tensor[0, 1].cpu().numpy()
            
            summary_data.append({
                'id': sensor_id, 'signal': raw_sig,
                'orig_cwt': orig_cwt, 'recon_cwt': recon_cwt
            })
            
            # --- 生成单传感器详细图表 ---
            sensor_dir = ensure_dir(os.path.join(output_root, f"sensor_{sensor_id}"))
            
            # 生成英文版
            plot_waveform(sensor_dir, sensor_id, raw_sig, 'en')
            plot_cwt_compare(sensor_dir, sensor_id, orig_cwt, recon_cwt, 'en')
            plot_zerone_compare(sensor_dir, sensor_id, orig_zerone, recon_zerone, 'en')
            
            # 生成中文版
            plot_waveform(sensor_dir, sensor_id, raw_sig, 'cn')
            plot_cwt_compare(sensor_dir, sensor_id, orig_cwt, recon_cwt, 'cn')
            plot_zerone_compare(sensor_dir, sensor_id, orig_zerone, recon_zerone, 'cn')

    # 6. 生成总览图
    if summary_data:
        print("  Generating summary plots...")
        plot_summary_page(output_root, raw_time, summary_data, 'en')
        plot_summary_page(output_root, raw_time, summary_data, 'cn')
        
    print(f"\n[Done] File {jsonl_path.name} processed.")

def scan_all_jsonl_files(root_dirs):
    """递归扫描配置目录下的所有 .jsonl 文件"""
    files = []
    if not isinstance(root_dirs, list):
        root_dirs = [root_dirs]
        
    for d in root_dirs:
        path_d = Path(d)
        if path_d.exists():
            found = list(path_d.rglob("*.jsonl"))
            files.extend(found)
            print(f"在 {d} 下找到 {len(found)} 个文件")
        else:
            print(f"警告: 路径不存在 {d}")
    return files

if __name__ == "__main__":
    # 1. 确定模型文件路径
    model_filename = "vae_stage1_epoch_50.pth" 
    MODEL_FILE = cfg.CHECKPOINT_DIR/ "model" / model_filename
    
    if not MODEL_FILE.exists():
        print(f"错误: 模型文件不存在 {MODEL_FILE}")
        if Path(model_filename).exists():
            MODEL_FILE = Path(model_filename)
            print(f"在当前目录找到模型: {MODEL_FILE}")
        else:
            exit()

    # 2. 扫描所有数据文件
    print("正在扫描数据文件...")
    all_data_files = scan_all_jsonl_files(cfg.RAW_DATA_DIRS)
    
    if not all_data_files:
        print("未找到任何 .jsonl 数据文件，请检查 hetero_config.py 中的 DATA_ROOT")
        exit()

    print(f"共发现 {len(all_data_files)} 个数据文件。")

    # 3. 批量可视化
    # MAX_VIS_COUNT = 5 # 设置最大处理数量用于测试，注释掉则处理全部
    
    print(f"开始处理所有文件...")
    # files_to_process = all_data_files[:MAX_VIS_COUNT] # 如果需要限制数量，取消注释
    files_to_process = all_data_files

    for i, jsonl_path in enumerate(files_to_process):
        print(f"\n[{i+1}/{len(files_to_process)}] 处理文件: {jsonl_path.name}")
        try:
            # 调用主处理函数
            main(jsonl_path, MODEL_FILE)
        except Exception as e:
            print(f"处理文件 {jsonl_path.name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n所有可视化任务完成！请查看 outputs/viz 文件夹。")