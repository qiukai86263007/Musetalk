#!/bin/bash

# 设置Conda的路径，根据您的安装位置
CONDA_PATH="/root/miniconda3"

# 设置您的MuseTalk环境名称
CONDA_ENV="musetalk"

# 设置MuseTalk代码路径
MUSETALK_PATH="/opt/Musetalk"

# 使用Conda初始化shell
source "$CONDA_PATH/etc/profile.d/conda.sh"

# 激活MuseTalk环境
conda activate $CONDA_ENV

echo "已激活 $CONDA_ENV 环境"

# 切换到MuseTalk目录
cd $MUSETALK_PATH

# 获取传递给脚本的所有参数
VIDEO_PATH="$1"
AUDIO_PATH="$2"

if [ -z "$VIDEO_PATH" ] || [ -z "$AUDIO_PATH" ]; then
    # 如果没有提供参数，使用默认值
    VIDEO_PATH="data/video/sun.mp4"
    AUDIO_PATH="data/audio/sun.wav"
    echo "使用默认参数："
    echo "视频路径: $VIDEO_PATH"
    echo "音频路径: $AUDIO_PATH"
fi

# 执行命令
echo "执行命令: python -m scripts.parallel_inference --video_path $VIDEO_PATH --audio_path $AUDIO_PATH"
python -m scripts.thread_inference --video_path "$VIDEO_PATH" --audio_path "$AUDIO_PATH"

# 执行完毕后停用环境
conda deactivate

echo "MuseTalk命令执行完毕，已停用 $CONDA_ENV 环境"