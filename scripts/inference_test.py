import argparse
import fileinput
import os
from omegaconf import OmegaConf
import numpy as np
import cv2

import torch
import glob
import pickle
from tqdm import tqdm
import copy

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil
# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
timesteps = torch.tensor([0],device=device)

@torch.no_grad()
def main(args):
    global pe
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    # inference_config = OmegaConf.load(args.inference_config)

    # print(inference_config)
    # for task_id in inference_config:
    # video_path = inference_config[task_id]["video_path"]
    # audio_path = inference_config[task_id]["audio_path"]
    # bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
    video_path = args.video_path
    audio_path = args.audio_path
    bbox_shift = args.bbox_shift
    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename  = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
    os.makedirs(result_img_save_path,exist_ok =True)

    if args.output_vid_name is None:
        output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path)=="video":
        # 如果输入是视频文件：

        # 设置保存提取帧的目录路径
        # args.result_dir 是结果保存的根目录，input_basename 是视频文件的基本名称（不含扩展名）
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full,exist_ok = True)
        # 使用 ffmpeg 从视频中提取帧
        # -v fatal：只显示致命错误信息
        # -i {video_path}：指定输入视频路径
        # -start_number 0：帧编号从 0 开始
        # {save_dir_full}/%08d.png：将帧保存为 8 位数字编号的 PNG 文件（例如 00000000.png, 00000001.png, ...）
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        os.system(cmd)
        # 获取保存的帧文件列表，按文件名排序
        # 使用 glob 匹配目录中所有支持的图片格式（如 jpg, png, jpeg, JPG, PNG 等）
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    elif get_file_type(video_path)=="image":
        # 将图片路径放入列表中（因为后续代码可能期望一个列表）
        input_img_list = [video_path, ]
        fps = args.fps
    elif os.path.isdir(video_path):  # input img folder
        # 如果输入是一个图片文件夹：
        # 获取文件夹中所有支持的图片文件（如 jpg, png, jpeg, JPG, PNG 等）
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        # 按文件名排序（假设文件名是数字，例如 0.jpg, 1.jpg, 2.jpg, ...）
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    else:
        # 啥也不是则抛出ValueError异常
        raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

    #print(input_img_list)
    ############################################## extract audio feature ##############################################
    # 使用了whisper推理，（可尝试对whisper推理过程加速）
    # 使用音频处理器将音频文件转换为特征表示
    whisper_feature = audio_processor.audio2feat(audio_path)
    # 将提取的音频特征按时间分块
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    ############################################## preprocess input image  ##############################################
    # 预处理图片 方便后续的UNET进行处理，其中使用了VAE（可尝试VAE加速）
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)

    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
            # 解包边界框的坐标值
        x1, y1, x2, y2 = bbox
        # 根据边界框坐标裁剪帧
        crop_frame = frame[y1:y2, x1:x2]
        # 调整裁剪后图像大小为256x256像素，使用Lanczos插值算法以获得高质量缩放效果
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        # 使用变分自动编码器(VAE)将裁剪并调整大小后的帧转换为潜在变量，供UNET模型使用
        latents = vae.get_latents_for_unet(crop_frame)
        # 将处理得到的潜在变量添加到输入潜在变量列表中
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    ############################################## inference batch by batch ##############################################
    # 真正开始推理的过程
    print("start inference")
    # 获取音频chunks的数量
    video_num = len(whisper_chunks)
    # 获取批处理大小
    batch_size = args.batch_size
    # 使用datagen函数生成批次数据迭代器，输入为whisper音频特征块和循环潜在变量列表
    gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    # 展示重构后的帧/模型输出的帧
    res_frame_list = []
    # 遍历由datagen生成的数据批次，tqdm用于显示进度条
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        # 将whisper音频特征块转换为torch张量，并将设备 和 类型 都与unet同步
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                     dtype=unet.model.dtype) # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
#                 print(bbox)
            continue

        combine_frame = get_image(ori_frame,res_frame,bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)

    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)

    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)

    os.remove("temp.mp4")
    shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results', help="path to output")

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord",
                        action="store_true",
                        help='use saved coordinate to save time')
    parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )

    # new param
    parser

    args = parser.parse_args()
    print(args)
    main(args)
