#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuseTalk推理脚本调用工具

该脚本用于在多GPU环境中自动选择合适的GPU来运行MuseTalk的推理任务。
它支持标准推理模式和实时推理模式，并提供多种参数用于控制推理过程。

作者: Claude
日期: 2025-03-06
"""

import argparse
import subprocess
import sys
import os
import time


def get_gpu_info(gpu_id):
    """
    获取指定GPU的内存和利用率信息

    Args:
        gpu_id (int): GPU编号

    Returns:
        tuple: (可用内存(MB), GPU利用率(%))
    """
    # 获取可用内存
    memory_cmd = f"nvidia-smi -i {gpu_id} --query-gpu=memory.free --format=csv,noheader,nounits"
    memory_result = subprocess.run(memory_cmd, shell=True, capture_output=True, text=True)
    free_memory = int(memory_result.stdout.strip()) if memory_result.returncode == 0 else 0

    # 获取GPU利用率
    util_cmd = f"nvidia-smi -i {gpu_id} --query-gpu=utilization.gpu --format=csv,noheader,nounits"
    util_result = subprocess.run(util_cmd, shell=True, capture_output=True, text=True)
    utilization = int(util_result.stdout.strip()) if util_result.returncode == 0 else 100

    return free_memory, utilization


def acquire_gpu_lock(gpu_id, lock_dir='/tmp/gpu_locks'):
    """
    尝试获取GPU的锁，防止多个进程同时使用同一个GPU

    Args:
        gpu_id (int): GPU编号
        lock_dir (str): 锁文件目录

    Returns:
        bool: 是否成功获取锁
    """
    # 确保锁目录存在
    os.makedirs(lock_dir, exist_ok=True)

    lock_file = os.path.join(lock_dir, f"gpu_{gpu_id}.lock")

    # 检查锁文件是否存在且进程仍在运行
    if os.path.exists(lock_file):
        try:
            # 读取PID
            with open(lock_file, 'r') as f:
                pid = int(f.read().strip())

            # 检查进程是否存在
            os.kill(pid, 0)  # 不发送信号，只检查进程是否存在
            return False  # 进程存在，锁有效
        except (ProcessLookupError, ValueError):
            # 进程不存在或PID无效，可以覆盖锁
            pass

    # 创建新锁
    try:
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except:
        return False


def release_gpu_lock(gpu_id, lock_dir='/tmp/gpu_locks'):
    """释放GPU锁"""
    lock_file = os.path.join(lock_dir, f"gpu_{gpu_id}.lock")
    if os.path.exists(lock_file):
        try:
            with open(lock_file, 'r') as f:
                pid = int(f.read().strip())

            # 只有创建锁的进程才能释放锁
            if pid == os.getpid():
                os.remove(lock_file)
                return True
        except:
            pass
    return False


def run_musetalk():
    """
    执行MuseTalk推理脚本的主函数

    Returns:
        int: 进程的退出状态码
    """
    parser = argparse.ArgumentParser(description='运行MuseTalk推理')
    parser.add_argument('-v', '--video-path', required=True, help='输入视频路径')
    parser.add_argument('-a', '--audio-path', required=True, help='输入音频路径')
    parser.add_argument('-r', '--result-dir', default='./results', help='结果保存目录')
    parser.add_argument('-b', '--bbox-shift', type=int, default=0, help='bbox_shift参数，控制嘴部开合程度')
    parser.add_argument('-m', '--memory-threshold', type=int, default=12000, help='最小所需GPU内存(MB)')
    parser.add_argument('-t', '--mode', choices=['standard', 'realtime'], default='standard',
                        help='推理模式: standard或realtime')
    parser.add_argument('--timeout', type=int, default=3600, help='脚本超时时间(秒)')
    parser.add_argument('--utilization-threshold', type=int, default=20, help='最大GPU利用率阈值(%)')
    parser.add_argument('--max-gpu-id', type=int, default=7, help='最大GPU ID范围')
    parser.add_argument('--memory-safety-margin', type=int, default=2000,
                        help='内存安全边际(MB)，防止内存估计不准确')
    parser.add_argument('--lock-dir', default='/tmp/gpu_locks', help='GPU锁文件目录')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在: {args.video_path}", file=sys.stderr)
        return 1

    if not os.path.exists(args.audio_path):
        print(f"错误: 音频文件不存在: {args.audio_path}", file=sys.stderr)
        return 1

    # 确保结果目录存在
    os.makedirs(args.result_dir, exist_ok=True)

    # 创建锁目录
    os.makedirs(args.lock_dir, exist_ok=True)

    # 真实所需内存 = 请求内存 + 安全边际
    real_memory_threshold = args.memory_threshold + args.memory_safety_margin

    # 查找合适的GPU并获取锁
    selected_gpu = None
    print("正在查找合适的GPU...")

    for gpu_id in range(args.max_gpu_id):  # 检查0到max_gpu_id-1号GPU
        free_memory, utilization = get_gpu_info(gpu_id)
        print(f"GPU {gpu_id}: 可用内存 {free_memory}MB, 利用率 {utilization}%")

        if (free_memory >= real_memory_threshold and
                utilization <= args.utilization_threshold):

            print(f"尝试获取GPU {gpu_id}的锁...")
            if acquire_gpu_lock(gpu_id, args.lock_dir):
                selected_gpu = gpu_id
                print(f"成功获取GPU {selected_gpu}的锁")
                break
            else:
                print(f"GPU {gpu_id}已被其他进程锁定")

    if selected_gpu is None:
        print("错误: 无可用GPU (内存 >= {}MB, 利用率 <= {}%)".format(
            real_memory_threshold, args.utilization_threshold), file=sys.stderr)
        return 255

    try:
        # 构建命令
        if args.mode == 'standard':
            cmd = [
                'python', '-m', 'scripts.inference_test',
                '--video_path', args.video_path,
                '--audio_path', args.audio_path,
                '--result_dir', args.result_dir,
                '--bbox_shift', str(args.bbox_shift)
            ]
        else:  # realtime模式
            # 为realtime模式创建临时配置文件
            config_dir = os.path.join(args.result_dir, 'configs')
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, 'temp_realtime_config.yaml')

            # 从视频路径获取任务ID
            task_id = os.path.splitext(os.path.basename(args.video_path))[0]

            with open(config_path, 'w') as f:
                f.write(f"""
{task_id}:
  preparation: True
  video_path: {args.video_path}
  bbox_shift: {args.bbox_shift}
  audio_clips:
    output: {args.audio_path}
""")

            cmd = [
                'python', '-m', 'scripts.realtime_inference',
                '--inference_config', config_path,
                '--batch_size', '4'
            ]

        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)

        print(f"执行命令: {' '.join(cmd)}")

        # 运行命令，设置超时
        try:
            process = subprocess.run(
                cmd,
                env=env,
                timeout=args.timeout
            )
            return process.returncode
        except subprocess.TimeoutExpired:
            print(f"错误: 推理任务超时 (>{args.timeout}秒)", file=sys.stderr)
            return 124  # 标准timeout退出码
    finally:
        # 无论如何，最后释放GPU锁
        if selected_gpu is not None:
            release_gpu_lock(selected_gpu, args.lock_dir)
            print(f"已释放GPU {selected_gpu}的锁")
        print(f"已释放GPU {selected_gpu}的锁")


if __name__ == "__main__":
    # 如果不带参数执行，则使用默认值
    if len(sys.argv) == 1:
        sys.argv.extend([
            '-v', '/opt/Musetalk/data/video/sun.mp4',
            '-a', '/opt/Musetalk/data/audio/sun.wav',
            '-t', 'standard'
        ])
        print("使用默认参数:")
        print("  视频: /opt/Musetalk/data/video/sun.mp4")
        print("  音频: /opt/Musetalk/data/audio/sun.wav")
        print("  模式: standard")

    start_time = time.time()
    print("=" * 50)
    print("MuseTalk推理调用工具 - 开始执行")
    print("=" * 50)

    exit_code = run_musetalk()

    elapsed_time = time.time() - start_time
    print("=" * 50)
    print(f"任务完成，耗时: {elapsed_time:.2f}秒, 退出状态: {exit_code}")
    print("=" * 50)

    sys.exit(exit_code)