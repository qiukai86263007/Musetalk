#!/bin/bash

# 设置Conda的路径
CONDA_PATH="/root/miniconda3"

# 设置MuseTalk环境名称
CONDA_ENV="musetalk"

# 设置MuseTalk代码路径
MUSETALK_PATH="/opt/Musetalk"

# 设置默认值数组
DEFAULT_VIDEOS=("$MUSETALK_PATH/data/video/sun.mp4" "$MUSETALK_PATH/data/video/sun1.mp4" "$MUSETALK_PATH/data/video/sun2.mp4" "$MUSETALK_PATH/data/video/sun3.mp4" "$MUSETALK_PATH/data/video/sun4.mp4" "$MUSETALK_PATH/data/video/sun5.mp4" "$MUSETALK_PATH/data/video/sun6.mp4")
DEFAULT_AUDIOS=("$MUSETALK_PATH/data/audio/sun.wav" "$MUSETALK_PATH/data/audio/sun1.wav" "$MUSETALK_PATH/data/audio/sun2.wav" "$MUSETALK_PATH/data/audio/sun3.wav" "$MUSETALK_PATH/data/audio/sun4.wav" "$MUSETALK_PATH/data/audio/sun5.wav" "$MUSETALK_PATH/data/video/sun6.mp4")

# 使用Conda初始化shell
source "$CONDA_PATH/etc/profile.d/conda.sh"

# 激活MuseTalk环境
conda activate $CONDA_ENV

echo "已激活 $CONDA_ENV 环境"

# 切换到MuseTalk目录
cd $MUSETALK_PATH

# 默认结果目录
RESULT_DIR="./results"

# 初始化空数组
VIDEO_PATHS=()
AUDIO_PATHS=()
PARSING_VIDEOS=0
PARSING_AUDIOS=0

# 先打印一下
echo "开始解析参数..."

# 如果没有参数，使用默认值
if [ "$#" -eq 0 ]; then
    echo "未提供参数，使用默认值"
    VIDEO_PATHS=("${DEFAULT_VIDEOS[@]}")
    AUDIO_PATHS=("${DEFAULT_AUDIOS[@]}")
else
    # 解析传入的参数
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --video_path)
                PARSING_VIDEOS=1
                PARSING_AUDIOS=0
                shift
                # 处理后续的视频路径，直到遇到下一个标志
                while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                    VIDEO_PATHS+=("$1")
                    shift
                done
                ;;
            --audio_path)
                PARSING_VIDEOS=0
                PARSING_AUDIOS=1
                shift
                # 处理后续的音频路径，直到遇到下一个标志
                while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                    AUDIO_PATHS+=("$1")
                    shift
                done
                ;;
            --result_dir)
                PARSING_VIDEOS=0
                PARSING_AUDIOS=0
                shift
                if [ "$#" -gt 0 ]; then
                    RESULT_DIR="$1"
                    shift
                fi
                ;;
            *)
                echo "未知参数: $1"
                shift
                ;;
        esac
    done
fi

# 确保路径数组非空
if [ ${#VIDEO_PATHS[@]} -eq 0 ]; then
    echo "未提供视频路径，使用默认值"
    VIDEO_PATHS=("${DEFAULT_VIDEOS[@]}")
fi

if [ ${#AUDIO_PATHS[@]} -eq 0 ]; then
    echo "未提供音频路径，使用默认值"
    AUDIO_PATHS=("${DEFAULT_AUDIOS[@]}")
fi

# 检查视频和音频路径数量是否匹配
if [ ${#VIDEO_PATHS[@]} -ne ${#AUDIO_PATHS[@]} ]; then
    echo "错误: 视频路径数量(${#VIDEO_PATHS[@]})与音频路径数量(${#AUDIO_PATHS[@]})不匹配"
    exit 1
fi

echo "视频路径: ${VIDEO_PATHS[*]}"
echo "音频路径: ${AUDIO_PATHS[*]}"
echo "结果目录: $RESULT_DIR"

# 构建命令参数
VIDEO_ARGS=""
for video in "${VIDEO_PATHS[@]}"; do
    VIDEO_ARGS+="$video "
done

AUDIO_ARGS=""
for audio in "${AUDIO_PATHS[@]}"; do
    AUDIO_ARGS+="$audio "
done

# 执行Python脚本
echo "执行命令: python -m scripts.thread_inference --video_path $VIDEO_ARGS --audio_path $AUDIO_ARGS --result_dir $RESULT_DIR"
python -m scripts.thread_inference --video_path $VIDEO_ARGS --audio_path $AUDIO_ARGS --result_dir "$RESULT_DIR"

# 保存退出状态码
EXIT_CODE=$?

# 执行完毕后停用环境
conda deactivate

echo "MuseTalk命令执行完毕，已停用 $CONDA_ENV 环境"
echo "退出状态码: $EXIT_CODE"

# 返回Python脚本的退出状态码
exit $EXIT_CODE