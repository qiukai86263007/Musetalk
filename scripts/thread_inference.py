import multiprocessing
import subprocess
import argparse
import time
from datetime import datetime
import os

def print_with_time(message):
    """打印带时间戳的消息"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def run_inference_on_gpu(gpu_id, video_path, audio_path, result_dir):
    """
    在指定的GPU上运行inference_test.py脚本。
    
    :param gpu_id: 要使用的GPU的ID
    :param video_path: 视频文件路径
    :param audio_path: 音频文件路径
    :param result_dir: 结果保存目录
    :return: 脚本的返回码
    """
    print_with_time(f"GPU {gpu_id} - 开始处理任务")
    print_with_time(f"GPU {gpu_id} - 视频: {video_path}")
    print_with_time(f"GPU {gpu_id} - 音频: {audio_path}")
    print_with_time(f"GPU {gpu_id} - 结果目录: {result_dir}")
    
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m scripts.inference_test --video_path {video_path} --audio_path {audio_path} --result_dir {result_dir}"
    print_with_time(f"GPU {gpu_id} - 执行命令: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    
    if result.returncode == 0:
        print_with_time(f"GPU {gpu_id} - 任务成功完成，耗时: {end_time - start_time:.2f}秒")
    else:
        print_with_time(f"GPU {gpu_id} - 任务失败，返回码: {result.returncode}")
        print_with_time(f"GPU {gpu_id} - 错误输出: {result.stderr.decode('utf-8', errors='replace')}")
    
    return result.returncode

def worker_process(gpu_id, video_path, audio_path, result_dir, return_dict):
    """工作进程函数，用于在multiprocessing中运行并返回结果"""
    try:
        return_code = run_inference_on_gpu(gpu_id, video_path, audio_path, result_dir)
        return_dict[gpu_id] = return_code
        print_with_time(f"GPU {gpu_id} - 工作进程完成，返回码: {return_code}")
    except Exception as e:
        print_with_time(f"GPU {gpu_id} - 工作进程异常: {str(e)}")
        return_dict[gpu_id] = -999  # 使用特殊代码表示异常

def main(args):
    print_with_time("主函数开始执行")
    print_with_time(f"视频路径列表: {args.video_path}")
    print_with_time(f"音频路径列表: {args.audio_path}")
    print_with_time(f"结果目录: {args.result_dir}")
    
    num_gpus = 8
    num_tasks = len(args.video_path)
    
    print_with_time(f"任务总数: {num_tasks}, 可用GPU数: {num_gpus}")
    
    if num_tasks > num_gpus:
        print_with_time(f"错误: 任务数({num_tasks})超过可用GPU数({num_gpus})")
        return -1
    
    if len(args.video_path) != len(args.audio_path):
        print_with_time(f"错误: 视频路径数({len(args.video_path)})与音频路径数({len(args.audio_path)})不匹配")
        return -1
    
    # 使用Manager来共享返回值
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    
    print_with_time("开始创建并启动工作进程...")
    
    # 创建并启动每个GPU的进程
    for gpu_id in range(num_tasks):
        p = multiprocessing.Process(
            target=worker_process,
            args=(gpu_id, args.video_path[gpu_id], args.audio_path[gpu_id], args.result_dir, return_dict)
        )
        print_with_time(f"创建进程 GPU {gpu_id} - 视频: {args.video_path[gpu_id]}, 音频: {args.audio_path[gpu_id]}")
        p.start()
        processes.append(p)
    
    print_with_time(f"已启动 {len(processes)} 个工作进程，等待完成...")
    
    # 等待所有进程完成
    for i, p in enumerate(processes):
        p.join()
        print_with_time(f"进程 {i} 已结束")
    
    print_with_time("所有进程已完成")
    
    # 收集结果
    results = [return_dict.get(i, -888) for i in range(num_tasks)]  # -888表示进程没有返回结果
    print_with_time(f"结果列表: {results}")
    
    # 检查是否有任何进程失败
    if any(result != 0 for result in results):
        failed_count = sum(1 for result in results if result != 0)
        print_with_time(f"警告: {failed_count}个任务失败")
        return 1
    
    print_with_time("所有任务成功完成")
    return 0

if __name__ == "__main__":
    print_with_time("程序开始执行")
    
    parser = argparse.ArgumentParser(description="多GPU并行运行MuseTalk推理")
    parser.add_argument("--video_path", type=str, nargs='+', required=True, help="视频文件路径列表")
    parser.add_argument("--audio_path", type=str, nargs='+', required=True, help="音频文件路径列表")
    parser.add_argument("--result_dir", default='./results', help="结果输出目录")
    
    args = parser.parse_args()
    print_with_time(f"解析的命令行参数: {args}")
    
    start_time = time.time()
    exit_code = main(args)
    end_time = time.time()
    
    total_time = end_time - start_time
    print_with_time(f"程序执行完成，总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print_with_time(f"退出码: {exit_code}")
    
    import sys
    sys.exit(exit_code)