#!/bin/bash
models=(
    "X-Humanoid/XR-1-Stage1-UVMC"
    "X-Humanoid/XR-1-Stage2"
)

mkdir -p ../pretrained

for model in "${models[@]}";
do
  echo "Downloading ${model}..."
  huggingface-cli download --resume-download --local-dir-use-symlinks False "${model}" \
  --local-dir ../pretrained/$(basename "${model}")
done

