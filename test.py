import os
import csv
from pathlib import Path
import mimetypes
import argparse
from pytorch.inference import audio_tagging
import multiprocessing

def is_audio_file(file_path):
    """判断文件是否为音频文件"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('audio/')

def process_audio_files(root_dir, output_csv, num_processes=4):
    """递归处理目录下的音频文件并保存结果"""
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开CSV文件准备写入
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['File Path', 'Status', 'Result'])
        
        # 递归遍历目录
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 检查是否为音频文件
                if is_audio_file(file_path):
                    print(f"Processing: {file_path}")
                    
                    # 创建参数对象
                    args = argparse.Namespace()
                    args.sample_rate = 32000
                    args.window_size = 1024
                    args.hop_size = 320
                    args.mel_bins = 64
                    args.fmin = 50
                    args.fmax = 14000
                    args.model_type = 'Cnn14'
                    args.checkpoint_path = 'files/Cnn14_mAP=0.431.pth'
                    args.audio_path = file_path
                    args.cuda = False
                    
                    try:
                        # 处理文件 - 只传递args参数
                        result, labels = audio_tagging(args)
                        status = 'Success'
                    except Exception as e:
                        success = False
                        result = str(e)
                        status = 'Failed'
                    
                    # 写入结果到CSV
                    writer.writerow([file_path, status, result])

if __name__ == "__main__":
    # 设置要处理的根目录和输出CSV文件路径
    root_directory = "D:\声纹测试库-amp-amp"
    output_csv_file = "results.csv"
    
    # 获取 CPU 核心数
    cpu_count = multiprocessing.cpu_count()
    
    # 使用核心数的一半
    process_audio_files(root_directory, output_csv_file, num_processes=cpu_count // 2)