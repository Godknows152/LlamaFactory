#!/bin/bash
# ============================================================================
# 单机4卡 DDP 训练脚本：Qwen3-VL Stop 工具 SFT 补训
# ============================================================================
# 目的：让模型学会在图像质量足够好时调用 stop 工具
# 基底模型: Train3/Step36 (已具备工具调用能力)
# 训练数据: 四种退化各200张，GT为 stop 工具调用
# 使用方法: bash train_restore_stop_4gpu.sh
# 或指定GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_restore_stop_4gpu.sh
# ============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Qwen3-VL Stop 工具 SFT 补训启动器 ===${NC}\n"

# 1. 验证必要文件
CONFIG_FILE="examples/train_lora/qwen3vl_lora_sft_stop_4gpu.yaml"
DATA_TRAIN="data/restoration_sft_stop_train.jsonl"
DATA_VAL="data/restoration_sft_stop_val.jsonl"
BASE_MODEL="/home/LXJ/Python_Projects/verl/checkpoints/merged/Train3/Step36"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -f "$DATA_TRAIN" ] || [ ! -f "$DATA_VAL" ]; then
    echo -e "${YELLOW}⚠ 数据文件不存在，正在生成...${NC}"
    python3 scripts/generate_stop_sft_data.py
    if [ ! -f "$DATA_TRAIN" ] || [ ! -f "$DATA_VAL" ]; then
        echo -e "${RED}✗ 数据生成失败!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 数据文件生成完成${NC}"
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo -e "${RED}✗ 基底模型不存在: $BASE_MODEL${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件验证通过${NC}"
echo -e "${GREEN}✓ 数据文件验证通过${NC}"
echo -e "${GREEN}✓ 基底模型验证通过: $BASE_MODEL${NC}\n"

# 2. 统计训练数据
TRAIN_COUNT=$(wc -l < "$DATA_TRAIN")
VAL_COUNT=$(wc -l < "$DATA_VAL")
echo -e "${YELLOW}训练数据: ${TRAIN_COUNT} 条${NC}"
echo -e "${YELLOW}验证数据: ${VAL_COUNT} 条${NC}\n"

# 3. 生成时间戳格式的实验名
TIMESTAMP=$(date +"Stop_SFT_%m月%d日%H时%M分")

echo -e "${YELLOW}时间戳实验名: ${BLUE}${TIMESTAMP}${NC}"

# 4. 生成临时配置（注入实验名）
TEMP_CONFIG="/tmp/qwen3vl_stop_sft_4gpu_${TIMESTAMP// /}.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# 使用 sed 修改 YAML 中的 swanlab_run_name 和 output_dir
sed -i "s|^swanlab_run_name:.*|swanlab_run_name: ${TIMESTAMP}|" "$TEMP_CONFIG"
sed -i "s|^output_dir:.*|output_dir: saves/stop_sft/${TIMESTAMP}/|" "$TEMP_CONFIG"

echo -e "${GREEN}✓ 配置已更新: swanlab_run_name = ${TIMESTAMP}${NC}"
echo -e "${GREEN}✓ 配置已更新: output_dir = saves/stop_sft/${TIMESTAMP}/${NC}"

echo ""

# 5. 检查 GPU 数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)

if [ "$GPU_COUNT" -lt 4 ]; then
    echo -e "${YELLOW}⚠ 警告: 检测到 ${GPU_COUNT} 张 GPU，配置为 4 卡训练${NC}"
    echo -e "${YELLOW}   建议: CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_restore_stop_4gpu.sh${NC}"
else
    echo -e "${GREEN}✓ GPU 数量验证: ${GPU_COUNT} 张${NC}"
fi

echo ""

# 6. 执行单机分布式训练
echo -e "${BLUE}启动单机 4 卡 DDP 训练...${NC}\n"
echo -e "${YELLOW}注意: 这是 Stop 工具补充训练，基底模型为 Train3/Step36${NC}\n"

# 使用 torchrun 启动单机 DDP
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    --standalone \
    src/train.py "$TEMP_CONFIG"

# 7. 清理临时文件
rm -f "$TEMP_CONFIG"

echo -e "\n${GREEN}=== Stop SFT 补训完成 ===${NC}"
echo -e "${GREEN}✓ 模型已保存到: saves/stop_sft/${TIMESTAMP}/${NC}"
echo -e "${GREEN}✓ SwanLab 实验: LLaMA_Factory / ${TIMESTAMP}${NC}"