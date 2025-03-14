"""
GPU锁状态检查工具 (带脚本实例锁)

该脚本检查系统中GPU的锁状态，判断哪些GPU是空闲的（未被锁定）。
同时使用文件锁确保同一时间只有一个脚本实例在运行。

结合了GPU锁以及单例模式

"""

import os
import subprocess
import argparse
import sys
import fcntl
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))


def check_gpu_lock(gpu_id, lock_dir='/tmp/gpu_locks'):
    """
    检查GPU是否被锁定

    Args:
        gpu_id (int): GPU编号
        lock_dir (str): 锁文件目录

    Returns:
        bool: 是否被锁定
    """
    lock_file = os.path.join(lock_dir, f"gpu_{gpu_id}.lock")

    # 如果锁文件不存在，GPU未被锁定
    if not os.path.exists(lock_file):
        return False

    try:
        # 读取PID
        with open(lock_file, 'r') as f:
            pid = int(f.read().strip())

        # 检查进程是否存在
        try:
            os.kill(pid, 0)  # 不发送信号，只检查进程是否存在
            return True  # 进程存在，GPU被锁定
        except ProcessLookupError:
            # 进程不存在，锁无效，GPU未被锁定
            return False
    except (ValueError, IOError):
        # 文件格式错误或读取问题，认为锁无效
        return False


def acquire_gpu_lock(gpu_id, lock_dir='/tmp/gpu_locks'):
    """
    尝试获取GPU的锁

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
    """
    释放GPU锁

    Args:
        gpu_id (int): GPU编号
        lock_dir (str): 锁文件目录

    Returns:
        bool: 是否成功释放锁
    """
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


def get_free_gpus(max_gpu_id=7, lock_dir='/tmp/gpu_locks'):
    """
    获取未被锁定的GPU列表

    Args:
        max_gpu_id (int): 最大GPU ID范围
        lock_dir (str): GPU锁文件目录

    Returns:
        list: 未锁定GPU的ID列表
    """
    # 确保锁目录存在
    os.makedirs(lock_dir, exist_ok=True)

    # 检查每个GPU是否被锁定
    free_gpus = []
    for gpu_id in range(max_gpu_id + 1):  # 包含max_gpu_id
        if not check_gpu_lock(gpu_id, lock_dir):
            free_gpus.append(gpu_id)

    return free_gpus


def select_and_lock_gpu(max_gpu_id=7, lock_dir='/tmp/gpu_locks'):
    """
    选择一个空闲GPU并锁定

    Args:
        max_gpu_id (int): 最大GPU ID范围
        lock_dir (str): GPU锁文件目录

    Returns:
        int or None: 选中并锁定的GPU ID，如果没有可用GPU则返回None
    """
    free_gpus = get_free_gpus(max_gpu_id, lock_dir)

    for gpu_id in free_gpus:
        if acquire_gpu_lock(gpu_id, lock_dir):
            logging.info(f"成功获取并锁定GPU {gpu_id}")
            return gpu_id

    return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GPU锁状态检查和锁定工具')
    parser.add_argument('--max-gpu-id', type=int, default=7, help='最大GPU ID范围')
    parser.add_argument('--lock-dir', default='/tmp/gpu_locks', help='GPU锁文件目录')
    parser.add_argument('--mode', choices=['check', 'lock', 'release'], default='check',
                        help='操作模式: check=只检查空闲GPU, lock=锁定一个空闲GPU, release=释放指定的GPU')
    parser.add_argument('--gpu-id', type=int, help='指定GPU ID (用于release模式)')

    args = parser.parse_args()

    if args.mode == 'check':
        # 获取未锁定的GPU列表
        free_gpus = get_free_gpus(args.max_gpu_id, args.lock_dir)
        print(f"空闲GPU: {' '.join(map(str, free_gpus))}")

        # 如果没有空闲GPU，返回非零退出码
        if not free_gpus:
            sys.exit(-1)

    elif args.mode == 'lock':
        # 选择并锁定一个GPU
        selected_gpu = select_and_lock_gpu(args.max_gpu_id, args.lock_dir)
        if selected_gpu is not None:
            print(f"已锁定GPU: {selected_gpu}")
            sys.exit(0)
        else:
            print("无法锁定任何GPU")
            sys.exit(-1)

    elif args.mode == 'release':
        # 检查是否提供了GPU ID
        if args.gpu_id is None:
            print("错误: release模式需要指定--gpu-id参数")
            sys.exit(-1)

        # 释放指定的GPU锁
        if release_gpu_lock(args.gpu_id, args.lock_dir):
            print(f"已释放GPU {args.gpu_id}的锁")
            sys.exit(0)
        else:
            print(f"无法释放GPU {args.gpu_id}的锁")
            sys.exit(-1)


if __name__ == "__main__":
    # 创建脚本实例锁文件路径
    script_lock_path = os.path.join(current_dir, 'gpu_check_script.lock')

    try:
        # 打开（或创建）锁文件
        with open(script_lock_path, 'w') as script_lock:
            try:
                # 尝试获取独占锁
                fcntl.flock(script_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 成功获取锁，执行主函数
                main()
            except IOError:
                # 无法获取锁，说明另一个脚本实例正在运行
                logging.error("另一个GPU检查工具实例正在运行")
                sys.exit(1)
    except Exception as e:
        logging.error(f"发生错误: {e}")
        sys.exit(1)