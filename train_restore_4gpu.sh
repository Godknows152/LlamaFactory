#!/bin/bash
# ============================================================================
# 单机4卡 DDP 训练脚本：Qwen3-VL 图像还原工具调用 SFT
# ============================================================================
# 使用方法: bash train_restore_4gpu.sh
# 或指定GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_restore_4gpu.sh
# ============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Qwen3-VL 单机4卡 DDP 训练启动器 ===${NC}\n"

# 1. 验证必要文件
CONFIG_FILE="examples/train_lora/qwen3vl_lora_sft_restore_4gpu.yaml"
DATA_TRAIN="data/restoration_sft_train.jsonl"
DATA_VAL="data/restoration_sft_val.jsonl"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -f "$DATA_TRAIN" ] || [ ! -f "$DATA_VAL" ]; then
    echo -e "${RED}✗ 数据文件不存在!${NC}"
    echo "  需要: $DATA_TRAIN"
    echo "  需要: $DATA_VAL"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件验证通过${NC}"
echo -e "${GREEN}✓ 数据文件验证通过${NC}\n"

# 2. 生成时间戳格式的实验名
TIMESTAMP=$(python3 << 'PYEOF'
from datetime import datetime
now = datetime.now()
exp_name = now.strftime("SFT_%-m月%-d日%-H时%-M分")
print(exp_name)
PYEOF
)

echo -e "${YELLOW}时间戳实验名: ${BLUE}${TIMESTAMP}${NC}"

# 3. 生成临时配置（注入实验名）
TEMP_CONFIG="/tmp/qwen3vl_restore_4gpu_${TIMESTAMP// /}.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# 使用 Python 修改 YAML 中的 swanlab_run_name
python3 << PYEOF
import yaml

config_file = "$TEMP_CONFIG"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['swanlab_run_name'] = "$TIMESTAMP"

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"✓ 配置已更新: swanlab_run_name = $TIMESTAMP")
PYEOF

echo ""

# 4. 检查 GPU 数量
GPU_COUNT=$(python3 << 'PYEOF'
import torch
count = torch.cuda.device_count()
print(count)
PYEOF
)

if [ "$GPU_COUNT" -lt 4 ]; then
    echo -e "${YELLOW}⚠ 警告: 检测到 ${GPU_COUNT} 张 GPU，配置为 4 卡训练${NC}"
    echo -e "${YELLOW}   建议: CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_restore_4gpu.sh${NC}"
else
    echo -e "${GREEN}✓ GPU 数量验证: ${GPU_COUNT} 张${NC}"
fi

echo ""

# 5. 执行单机分布式训练
echo -e "${BLUE}启动单机 4 卡 DDP 训练...${NC}\n"
echo -e "${YELLOW}注意: 这是单机4卡训练，使用 --standalone 模式${NC}\n"

# 使用 torchrun 启动单机 DDP
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    --standalone \
    src/train.py "$TEMP_CONFIG"

# 6. 清理临时文件
rm -f "$TEMP_CONFIG"

echo -e "\n${GREEN}=== 训练完成 ===${NC}"
echo -e "${GREEN}✓ 模型已保存到: $(grep 'output_dir:' $CONFIG_FILE | awk '{print $2}')${NC}"
echo -e "${GREEN}✓ SwanLab 实验: LLaMA_Factory / ${TIMESTAMP}${NC}"
