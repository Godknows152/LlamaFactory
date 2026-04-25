"""
测试 Qwen3-VL 图像还原工具调用 SFT 模型
使用 vLLM 框架加载合并后的模型，测试工具调用格式是否正确
"""

import json
import re
import base64
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

MODEL_PATH = "/home/LXJ/Python_Projects/LLaMA_Factory/LlamaFactory/saves/qwen3-vl-8b-restore/merged"

SYSTEM_PROMPT = (
    "You are an expert image restoration assistant. Your task is to analyze a degraded image "
    "and apply appropriate restoration operations one step at a time using the available tools. "
    "After each restoration step, evaluate the result and decide if further processing is needed."
)

USER_PROMPT = (
    "Please diagnose the issues in this degraded image and formulate a restoration plan. "
    "Available restoration tools:\n"
    "• real_esrgan: General restoration (super-resolution/deblurring/denoising/artifact removal)\n"
    "• scunet: Professional denoising\n"
    "• retinexformer_fivek/hvicidnet/lightdiff: Low-light processing (fivek=natural, hvicidnet=extreme, lightdiff=diffusion)\n"
    "• kanet: Dehazing (fog/haze removal)\n"
    "• turbo_rain/s2former: Rain removal (turbo_rain=fast/heavy, s2former=light/streaks)\n"
    "• turbo_snow/snowmaster: Snow removal (turbo_snow=fast/heavy, snowmaster=precise)\n"
    "• nafnet: Low-noise deblurring\n"
    "• seesr: Semantic super-resolution\n\n"
    "Analyze the image and call restore_image with the most appropriate action."
)

TOOLS = [
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
                            "real_esrgan", "scunet", "retinexformer_fivek",
                            "hvicidnet", "lightdiff", "kanet",
                            "turbo_rain", "s2former", "turbo_snow",
                            "snowmaster", "nafnet", "seesr"
                        ],
                        "description": "The restoration action to apply"
                    }
                },
                "required": ["action"]
            }
        }
    }
]

TEST_CASES = [
    {
        "name": "夜间低光照",
        "image": "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/train/images/night_series/night/010148.png",
        "expected_action": "lightdiff",  # GT中的action
        "expected_category": ["retinexformer_fivek", "hvicidnet", "lightdiff"],
    },
    {
        "name": "雾霾天气",
        "image": "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/train/images/fog_series/fog/013486.png",
        "expected_action": "kanet",
        "expected_category": ["kanet"],
    },
    {
        "name": "雪景",
        "image": "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/train/images/snow_series/snow/012240.png",
        "expected_action": "turbo_snow",
        "expected_category": ["turbo_snow", "snowmaster"],
    },
    {
        "name": "雨天",
        "image": "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/train/images/rain_series/rain/all/rain_drop_003789.png",
        "expected_action": "s2former",
        "expected_category": ["turbo_rain", "s2former"],
    },
]


def build_messages(image_path: str) -> list:
    """构造带图片的消息列表"""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    return messages


def parse_tool_call(output_text: str) -> dict | None:
    """解析输出中的 <tool_call>...</tool_call>"""
    match = re.search(r"<tool_call>(.*?)</tool_call>", output_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


def main():
    print("=" * 60)
    print("加载模型:", MODEL_PATH)
    print("=" * 60)

    llm = LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        dtype="bfloat16",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # 贪心解码，结果确定
        max_tokens=256,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )

    results = []
    print()

    for i, case in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] 测试: {case['name']}")
        print(f"  图片: {Path(case['image']).name}")

        messages = build_messages(case["image"])

        # vLLM 使用 apply_chat_template 处理多模态
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # 构造 prompt
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=TOOLS,
        )

        from PIL import Image
        pil_image = Image.open(case["image"]).convert("RGB")

        outputs = llm.generate(
            {
                "prompt": text_prompt,
                "multi_modal_data": {"image": pil_image},
            },
            sampling_params=sampling_params,
        )

        output_text = outputs[0].outputs[0].text
        tool_call = parse_tool_call(output_text)

        print(f"  原始输出: {output_text[:200]}")

        if tool_call:
            action = tool_call.get("arguments", {}).get("action", "MISSING")
            match_exact = action == case["expected_action"]
            match_category = action in case["expected_category"]
            status = "✅ 精确匹配" if match_exact else ("✅ 类别正确" if match_category else "❌ 错误")
            print(f"  工具调用: {tool_call}")
            print(f"  预测动作: {action}  期望: {case['expected_category']}  {status}")
        else:
            print(f"  ❌ 未找到有效的 <tool_call> 输出")
            action = None
            match_category = False

        results.append({
            "name": case["name"],
            "predicted": action,
            "expected": case["expected_category"],
            "correct": match_category,
        })
        print()

    # 汇总
    correct = sum(r["correct"] for r in results)
    print("=" * 60)
    print(f"测试结果: {correct}/{len(results)} 正确")
    print("=" * 60)
    for r in results:
        status = "✅" if r["correct"] else "❌"
        print(f"  {status} {r['name']}: 预测={r['predicted']}  期望={r['expected']}")


if __name__ == "__main__":
    main()
