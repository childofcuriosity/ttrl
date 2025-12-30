#!/bin/bash
set -e

# 基础路径（你需要确保目录存在）
BASE_DIR="."
DATA_LOCAL_DIR="${BASE_DIR}/data"
LLM_DIR="${BASE_DIR}/llms"

mkdir -p $DATA_LOCAL_DIR
mkdir -p $LLM_DIR

echo ">>> Downloading MATH dataset"
git clone https://github.com/hendrycks/math.git ${DATA_LOCAL_DIR}/MATH
echo "MATH dataset downloaded to: ${DATA_LOCAL_DIR}/MATH"

echo ">>> Downloading Qwen2.5-Math-1.5B model"
cd $LLM_DIR
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-Math-1.5B
echo "Model downloaded to: ${LLM_DIR}/Qwen2.5-Math-1.5B"

echo "✅ All downloads completed!"
