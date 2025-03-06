#!/bin/bash

video_path=""
audio_path=""
result_dir="./results"

while getopts "v:a:r:" opt; do
  case $opt in
    v)
      video_path="$OPTARG"
      ;;
    a)
      audio_path="$OPTARG"
      ;;
    r)
      result_dir="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 254
      ;;
  esac
done

for ((i=0; i<7; i++)); do
  # Query the free memory of the current GPU
  free_memory=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits)

  # Check if the free memory is greater than or equal to 24000MB
  if [ "$free_memory" -ge 24000 ]; then
    # Run the Python script using the current GPU
    CUDA_VISIBLE_DEVICES=$i python -m scripts.inference_test --video_path "$video_path" --audio_path "$audio_path" --result_dir "$result_dir"
    # Check if the Python script ran successfully
    exit $?
  fi
done

exit 255