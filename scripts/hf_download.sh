#!/bin/bash
models=(
    "google/siglip-so400m-patch14-224"
    "google/paligemma-3b-pt-224"
)

mkdir -p ../pretrained

for model in "${models[@]}";
do
  echo "Downloading ${model}..."
  huggingface-cli download --resume-download --local-dir-use-symlinks False "${model}" \
  --local-dir ../pretrained/$(basename "${model}")
done

