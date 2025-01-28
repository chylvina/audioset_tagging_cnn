import os
import csv
from pathlib import Path
import mimetypes
import argparse
from pytorch.inference import audio_tagging
import multiprocessing
from functools import partial
import logging
from multiprocessing import Queue, Process

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_audio_file(file_path):
    """判断文件是否为音频文件"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('audio/')

def process_single_file(file_path, args_template, result_queue):
    """处理单个音频文件并将结果放入队列"""
    try:
        logger.info(f"Processing: {file_path}")
        
        args = argparse.Namespace(**vars(args_template))
        args.audio_path = file_path
        
        result, labels = audio_tagging(args)
        status = 'Success'
        
        logger.info(f"Successfully processed: {file_path}")
        result_queue.put([file_path, status, result])
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        result_queue.put([file_path, 'Failed', str(e)])

def writer_process(result_queue, output_csv, total_files):
    """专门的写入进程，实时写入结果"""
    try:
        files_processed = 0
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['File Path', 'Status', 'Result'])
            
            # 持续从队列接收结果直到所有文件都处理完
            while files_processed < total_files:
                result = result_queue.get()
                writer.writerow(result)
                f.flush()  # 立即写入磁盘
                files_processed += 1
                logger.info(f"Progress: {files_processed}/{total_files}")
                
        logger.info("All results have been written to CSV")
        
    except Exception as e:
        logger.error(f"Error in writer process: {str(e)}")
        raise

def worker(file_paths, args_template, result_queue):
    """工作进程函数"""
    for file_path in file_paths:
        process_single_file(file_path, args_template, result_queue)

def process_audio_files(root_dir, output_csv, num_processes=4):
    """使用多进程处理音频文件并实时保存结果"""
    try:
        # 使用绝对路径
        abs_root_dir = os.path.abspath(root_dir)
        abs_output_csv = os.path.abspath(output_csv)
        
        logger.info(f"Processing files in directory: {abs_root_dir}")
        logger.info(f"Output will be saved to: {abs_output_csv}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(abs_output_csv), exist_ok=True)
        
        # 创建基础args模板
        args_template = argparse.Namespace()
        args_template.sample_rate = 32000
        args_template.window_size = 1024
        args_template.hop_size = 320
        args_template.mel_bins = 64
        args_template.fmin = 50
        args_template.fmax = 14000
        args_template.model_type = 'Cnn14'
        args_template.checkpoint_path = './20000_iterations.pth'
        args_template.cuda = True
        
        # 收集所有音频文件路径
        audio_files = []
        for root, _, files in os.walk(abs_root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if is_audio_file(file_path):
                    audio_files.append(file_path)
        
        if not audio_files:
            logger.warning(f"No audio files found in {abs_root_dir}")
            return
        
        total_files = len(audio_files)
        logger.info(f"Found {total_files} audio files to process")
        
        # 创建结果队列
        result_queue = Queue()
        
        # 启动写入进程
        writer = Process(target=writer_process, args=(result_queue, abs_output_csv, total_files))
        writer.start()
        
        # 将文件平均分配给处理进程
        files_per_process = total_files // num_processes
        processes = []
        
        for i in range(num_processes):
            start_idx = i * files_per_process
            end_idx = start_idx + files_per_process if i < num_processes - 1 else total_files
            process_files = audio_files[start_idx:end_idx]
            
            if process_files:  # 只有当有文件要处理时才创建进程
                p = Process(target=worker, args=(process_files, args_template, result_queue))
                processes.append(p)
                p.start()
        
        # 等待所有处理进程完成
        for p in processes:
            p.join()
            
        # 等待写入进程完成
        writer.join()
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in process_audio_files: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 获取当前脚本所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置输入和输出路径
        root_directory = os.path.join(current_dir, "datasets/audios/audios/eval_segments")
        output_csv_file = os.path.join(current_dir, "results.csv")
        
        # 获取 CPU 核心数
        cpu_count = multiprocessing.cpu_count()
        num_processes = max(cpu_count, 1)
        
        logger.info(f"Starting processing with {num_processes} processes")
        
        # 并行处理文件
        process_audio_files(root_directory, output_csv_file, num_processes)
        
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")