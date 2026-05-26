#!/usr/bin/env python3
"""
生成 Stop SFT 训练数据：让模型学会在图像质量足够好时调用 stop 工具。

从 a_tmp_results 目录下选择四种退化各自200张图片，
GT 为 stop 工具调用，系统提示词与用户提示词与 val 数据一致。
"""

import json
import os
import random
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================
IMG_BASE_DIR = "/home/LXJ/Python_Projects/JarvisIR/data/inference/results/a_tmp_results"
OUTPUT_DIR = "/home/LXJ/Python_Projects/LLaMA_Factory/LlamaFactory/data"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "restoration_sft_stop_train.jsonl")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "restoration_sft_stop_val.jsonl")

NUM_PER_DEGRADATION = 200  # 每种退化类型选取的图片数
VAL_RATIO = 0.05           # 验证集比例 (5% = 40张)
SEED = 42

DEGRADATION_TYPES = ["fog", "night", "rain", "snow"]

# ============================================================================
# 10种用户提示词模板（与 restoration_sft_val.jsonl 完全一致）
# ============================================================================
USER_PROMPT_TEMPLATES = [
    # Template 1
    "Analyze the image degradation in the following image<image> and select the optimal solution from these image restoration algorithms:\n[General Restoration] real_esrgan - Powerful multi-purpose restoration model\n[Denoising] scunet - CNN and Transformer hybrid denoising\n[Low-light Series] retinexformer_fivek, hvicidnet, lightdiff\n[Deraining Series] turbo_rain, s2former, idt\n[Dehazing Series] ridcp, kanet\n[Desnowing Series] turbo_snow, snowmaster",

    # Template 2
    "This image<image> contains one or more types of degradation. Please identify the degradation type and recommend the most suitable image restoration operation. Available restoration operations:\n1. real_esrgan: A powerful general-purpose model for image super-resolution, deblurring, denoising, and removing compression artifacts.\n2. scunet: A hybrid model combining CNN and Transformer, specialized for efficient, high-quality image denoising.\n3. retinexformer_fivek: Based on Retinex theory and Transformer, used to enhance brightness and details in low-light images.\n4. hvicidnet: A lightweight Transformer model for low-light enhancement and exposure correction.\n5. lightdiff: A diffusion model-based framework for unsupervised low-light image enhancement.\n6. img2img-turbo-rain (turbo_rain): An efficient SD-Turbo-based model specifically for fast removal of rain from real-world images.\n7. s2former: An uncertainty-aware Transformer model for rain streak removal.\n8. idt: A Transformer-based model that can remove both rain and raindrops from images.\n9. ridcp: An efficient dehazing model utilizing high-quality codebook priors.\n10. kanet: An efficient dehazing network using a \"locate-and-remove\" pipeline.\n11. img2img-turbo-snow (turbo_snow): An efficient SD-Turbo-based model for removing snow artifacts while preserving scene details.\n12. snowmaster: A real-world image desnowing model optimized with multimodal large language model (MLLM) feedback.",

    # Template 3
    "This is an image<image> with degradation issues. Please identify the degradation type and decide which restoration method to use:\nSuper-resolution/Deblurring/Denoising: real_esrgan\nImage Denoising: scunet\nLow-light Enhancement: retinexformer_fivek / hvicidnet / lightdiff\nDeraining: turbo_rain / s2former / idt\nDehazing: ridcp / kanet\nDesnowing: turbo_snow / snowmaster",

    # Template 4
    "Observe this degraded image<image>, identify its degradation type and recommend the best restoration solution. Available options include:\n1. real_esrgan - Super-resolution/deblurring/denoising/compression artifact removal\n2. scunet - High-quality denoising\n3. retinexformer_fivek - Low-light enhancement\n4. hvicidnet - Low-light/exposure correction\n5. lightdiff - Low-light enhancement (diffusion model)\n6. turbo_rain - Fast deraining\n7. s2former - Rain streak removal\n8. idt - Deraining/raindrop removal\n9. ridcp - Dehazing\n10. kanet - Dehazing\n11. turbo_snow - Desnowing\n12. snowmaster - Real-world desnowing",

    # Template 5
    "For this image<image> containing degradation, please complete the following tasks:\n1. Identify the degradation type in the image\n2. Select the most suitable restoration solution from the following algorithms\nAvailable algorithms: real_esrgan (general restoration), scunet (denoising), retinexformer_fivek/hvicidnet/lightdiff (low-light), turbo_rain/s2former/idt (deraining), ridcp/kanet (dehazing), turbo_snow/snowmaster (desnowing)",

    # Template 6
    "Please analyze the degradation issues in this image<image> and select the most appropriate restoration method from the following:\n- real_esrgan: General image enhancement for super-resolution, deblurring, denoising, and compression artifact removal\n- scunet: Efficient image denoising (CNN+Transformer hybrid architecture)\n- retinexformer_fivek: Low-light image enhancement (based on Retinex theory)\n- hvicidnet: Lightweight low-light enhancement and exposure correction\n- lightdiff: Unsupervised low-light enhancement (diffusion model)\n- turbo_rain: Fast deraining (based on SD-Turbo)\n- s2former: Rain streak removal (uncertainty-aware Transformer)\n- idt: Deraining and raindrop removal (Transformer architecture)\n- ridcp: Efficient dehazing (high-quality codebook priors)\n- kanet: Dehazing network (locate-and-remove pipeline)\n- turbo_snow: Desnowing (based on SD-Turbo)\n- snowmaster: Real-world desnowing (MLLM optimized)",

    # Template 7
    "Please diagnose the issues in this degraded image<image> and formulate a restoration plan. Available restoration tools:\n\u2022 real_esrgan: General restoration (super-resolution/deblurring/denoising/artifact removal)\n\u2022 scunet: Professional denoising\n\u2022 retinexformer_fivek/hvicidnet/lightdiff: Low-light processing\n\u2022 turbo_rain/s2former/idt: Deraining related\n\u2022 ridcp/kanet: Dehazing processing\n\u2022 turbo_snow/snowmaster: Desnowing processing",

    # Template 8
    "This image<image> has image quality degradation issues. Please identify the degradation type and select an appropriate restoration tool:\n[Super-resolution & General Restoration] real_esrgan\n[Denoising] scunet\n[Low-light Enhancement] retinexformer_fivek, hvicidnet, lightdiff\n[Deraining] turbo_rain, s2former, idt\n[Dehazing] ridcp, kanet\n[Desnowing] turbo_snow, snowmaster\nPlease provide your analysis and restoration decision.",

    # Template 9
    "Please examine this degraded image<image>, analyze its quality issues, and select an appropriate method from the restoration toolbox:\n\u25c6 General Enhancement: real_esrgan (super-resolution/deblurring/denoising/compression artifact removal)\n\u25c6 Denoising: scunet (efficient denoising)\n\u25c6 Low-light Enhancement: retinexformer_fivek / hvicidnet / lightdiff\n\u25c6 Rainy Weather Restoration: turbo_rain / s2former / idt\n\u25c6 Foggy Weather Restoration: ridcp / kanet\n\u25c6 Snowy Weather Restoration: turbo_snow / snowmaster",

    # Template 10
    "Review this image<image>, determine the existing degradation issues, and recommend the best image restoration strategy. Restoration options:\n1. real_esrgan - Multi-purpose restoration (super-resolution, deblurring, denoising, compression artifact removal)\n2. scunet - High-quality image denoising\n3. retinexformer_fivek - Low-light enhancement (Retinex+Transformer)\n4. hvicidnet - Lightweight low-light/exposure correction\n5. lightdiff - Diffusion model low-light enhancement\n6. turbo_rain - SD-Turbo fast deraining\n7. s2former - Uncertainty-aware deraining\n8. idt - Transformer deraining/raindrop removal\n9. ridcp - Codebook prior dehazing\n10. kanet - Locate-and-remove dehazing\n11. turbo_snow - SD-Turbo desnowing\n12. snowmaster - MLLM optimized desnowing",
]

# ============================================================================
# 固定字段
# ============================================================================
SYSTEM_PROMPT = "You are an expert image restoration assistant. Your task is to analyze a degraded image and apply appropriate restoration operations one step at a time using the available tools. After each restoration step you will receive the restored image and quality feedback. Select the most suitable restoration action for the observed degradation type, and call the 'restore_image' tool with the chosen action. Stop when the image quality is satisfactory."

TOOLS_JSON = json.dumps([
    {
        "type": "function",
        "function": {
            "name": "restore_image",
            "description": "Restore degraded image by selecting one action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "real_esrgan", "scunet",
                            "retinexformer_fivek", "hvicidnet", "lightdiff",
                            "turbo_rain", "s2former", "idt",
                            "ridcp", "kanet",
                            "turbo_snow", "snowmaster",
                            "stop"
                        ]
                    }
                },
                "required": ["action"]
            }
        }
    }
])

ASSISTANT_GT = (
    "Observation: After restoration, the image quality is good enough "
    "without noticeable degradation, so I can choose to stop.\n"
    "\u2291{\"name\": \"restore_image\", \"arguments\": {\"action\": \"stop\"}}\u2292"
)


def collect_images():
    """从每种退化类型目录中随机选取指定数量的图片"""
    random.seed(SEED)
    all_samples = []

    for deg_type in DEGRADATION_TYPES:
        deg_dir = os.path.join(IMG_BASE_DIR, deg_type)
        # 获取所有 png 图片
        images = sorted([
            f for f in os.listdir(deg_dir)
            if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')
        ])
        print(f"  {deg_type}: 共 {len(images)} 张图片，选取 {NUM_PER_DEGRADATION} 张")

        # 随机选取
        selected = random.sample(images, NUM_PER_DEGRADATION)

        for img_file in selected:
            img_path = os.path.join(deg_dir, img_file)
            # 随机选择一个用户提示词模板
            template_idx = random.randint(0, len(USER_PROMPT_TEMPLATES) - 1)
            user_prompt = USER_PROMPT_TEMPLATES[template_idx]

            sample = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "user", "value": user_prompt},
                    {"from": "assistant", "value": ASSISTANT_GT},
                ],
                "images": [img_path],
                "tools": TOOLS_JSON,
            }
            all_samples.append((deg_type, template_idx, sample))

    return all_samples


def main():
    print("=" * 60)
    print("生成 Stop SFT 训练数据")
    print("=" * 60)
    print(f"\n图片源目录: {IMG_BASE_DIR}")
    print(f"每种退化选取: {NUM_PER_DEGRADATION} 张")
    print(f"退化类型: {DEGRADATION_TYPES}")
    print(f"验证集比例: {VAL_RATIO}")
    print()

    # 收集图片并生成样本
    all_samples = collect_images()
    total = len(all_samples)
    print(f"\n总样本数: {total}")

    # 分割训练集和验证集
    val_count = int(total * VAL_RATIO)
    train_count = total - val_count
    print(f"训练集: {train_count} 张")
    print(f"验证集: {val_count} 张")

    # 随机打乱后分割
    random.seed(SEED)
    random.shuffle(all_samples)
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    # 写入训练集
    with open(TRAIN_OUTPUT, 'w') as f:
        for _, _, sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\n✓ 训练数据已写入: {TRAIN_OUTPUT} ({train_count} 条)")

    # 写入验证集
    with open(VAL_OUTPUT, 'w') as f:
        for _, _, sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"✓ 验证数据已写入: {VAL_OUTPUT} ({val_count} 条)")

    # 统计训练集退化分布
    from collections import Counter
    train_deg_counter = Counter(deg for deg, _, _ in train_samples)
    val_deg_counter = Counter(deg for deg, _, _ in val_samples)

    print("\n=== 训练集退化分布 ===")
    for deg, count in sorted(train_deg_counter.items()):
        print(f"  {deg}: {count} 张")

    print("\n=== 验证集退化分布 ===")
    for deg, count in sorted(val_deg_counter.items()):
        print(f"  {deg}: {count} 张")

    # 统计提示词模板分布
    train_template_counter = Counter(t for _, t, _ in train_samples)
    print("\n=== 训练集提示词模板分布 ===")
    for t, count in sorted(train_template_counter.items()):
        print(f"  Template {t+1}: {count} 次")

    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()