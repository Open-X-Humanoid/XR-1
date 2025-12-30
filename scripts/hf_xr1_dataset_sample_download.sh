#!/bin/bash

# configure parameters
REPO_ID="X-Humanoid/XR-1-Dataset-Sample"
LOCAL_DIR="../dataset/XR-1-Dataset-Sample"
HF_TOKEN="" # if private dataset, please fill the token here

echo "Start downloading dataset: $REPO_ID to $LOCAL_DIR ..."

# use huggingface-cli to download the dataset
# --repo-type dataset: type of the dataset
# --local-dir: local save path
# --local-dir-use-symlinks False: do not use symlinks, directly download the files to the target directory
huggingface-cli download $REPO_ID \
    --repo-type dataset \
    --local-dir $LOCAL_DIR \
    --local-dir-use-symlinks False \
    --token "$HF_TOKEN"

echo "Download completed!"