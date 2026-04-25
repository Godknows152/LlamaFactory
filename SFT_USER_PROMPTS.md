# User Prompts（10 种随机模板，RL/SFT 通用）

> 数据来源：`data/restoration/train.parquet` 中 `prompt` 列表里 `role="user"` 的 `content` 字段。
> 共 10 种，数据集会随机选用其中一种呈现给模型。

---

## 1. 模板 A

```
Analyze the image degradation in the following image<image> and select the optimal solution from these image restoration algorithms:
[General Restoration] real_esrgan - Powerful multi-purpose restoration model
[Denoising] scunet - CNN and Transformer hybrid denoising
[Low-light Series] retinexformer_fivek, hvicidnet, lightdiff
[Deraining Series] turbo_rain, s2former, idt
[Dehazing Series] ridcp, kanet
[Desnowing Series] turbo_snow, snowmaster
```

---

## 2. 模板 B

```
For this image<image> containing degradation, please complete the following tasks:
1. Identify the degradation type in the image
2. Select the most suitable restoration solution from the following algorithms
Available algorithms: real_esrgan (general restoration), scunet (denoising), retinexformer_fivek/hvicidnet/lightdiff (low-light), turbo_rain/s2former/idt (deraining), ridcp/kanet (dehazing), turbo_snow/snowmaster (desnowing)
```

---

## 3. 模板 C

```
Observe this degraded image<image>, identify its degradation type and recommend the best restoration solution. Available options include:
1. real_esrgan - Super-resolution/deblurring/denoising/compression artifact removal
2. scunet - High-quality denoising
3. retinexformer_fivek - Low-light enhancement
4. hvicidnet - Low-light/exposure correction
5. lightdiff - Low-light enhancement (diffusion model)
6. turbo_rain - Fast deraining
7. s2former - Rain streak removal
8. idt - Deraining/raindrop removal
9. ridcp - Dehazing
10. kanet - Dehazing
11. turbo_snow - Desnowing
12. snowmaster - Real-world desnowing
```

---

## 4. 模板 D

```
Please analyze the degradation issues in this image<image> and select the most appropriate restoration method from the following:
- real_esrgan: General image enhancement for super-resolution, deblurring, denoising, and compression artifact removal
- scunet: Efficient image denoising (CNN+Transformer hybrid architecture)
- retinexformer_fivek: Low-light image enhancement (based on Retinex theory)
- hvicidnet: Lightweight low-light enhancement and exposure correction
- lightdiff: Unsupervised low-light enhancement (diffusion model)
- turbo_rain: Fast deraining (based on SD-Turbo)
- s2former: Rain streak removal (uncertainty-aware Transformer)
- idt: Deraining and raindrop removal (Transformer architecture)
- ridcp: Efficient dehazing (high-quality codebook priors)
- kanet: Dehazing network (locate-and-remove pipeline)
- turbo_snow: Desnowing (based on SD-Turbo)
- snowmaster: Real-world desnowing (MLLM optimized)
```

---

## 5. 模板 E

```
Please diagnose the issues in this degraded image<image> and formulate a restoration plan. Available restoration tools:
• real_esrgan: General restoration (super-resolution/deblurring/denoising/artifact removal)
• scunet: Professional denoising
• retinexformer_fivek/hvicidnet/lightdiff: Low-light processing
• turbo_rain/s2former/idt: Deraining related
• ridcp/kanet: Dehazing processing
• turbo_snow/snowmaster: Desnowing processing
```

---

## 6. 模板 F

```
Please examine this degraded image<image>, analyze its quality issues, and select an appropriate method from the restoration toolbox:
◆ General Enhancement: real_esrgan (super-resolution/deblurring/denoising/compression artifact removal)
◆ Denoising: scunet (efficient denoising)
◆ Low-light Enhancement: retinexformer_fivek / hvicidnet / lightdiff
◆ Rainy Weather Restoration: turbo_rain / s2former / idt
◆ Foggy Weather Restoration: ridcp / kanet
◆ Snowy Weather Restoration: turbo_snow / snowmaster
```

---

## 7. 模板 G

```
Review this image<image>, determine the existing degradation issues, and recommend the best image restoration strategy. Restoration options:
1. real_esrgan - Multi-purpose restoration (super-resolution, deblurring, denoising, compression artifact removal)
2. scunet - High-quality image denoising
3. retinexformer_fivek - Low-light enhancement (Retinex+Transformer)
4. hvicidnet - Lightweight low-light/exposure correction
5. lightdiff - Diffusion model low-light enhancement
6. turbo_rain - SD-Turbo fast deraining
7. s2former - Uncertainty-aware deraining
8. idt - Transformer deraining/raindrop removal
9. ridcp - Codebook prior dehazing
10. kanet - Locate-and-remove dehazing
11. turbo_snow - SD-Turbo desnowing
12. snowmaster - MLLM optimized desnowing
```

---

## 8. 模板 H

```
This image<image> contains one or more types of degradation. Please identify the degradation type and recommend the most suitable image restoration operation. Available restoration operations:
1. real_esrgan: A powerful general-purpose model for image super-resolution, deblurring, denoising, and removing compression artifacts.
2. scunet: A hybrid model combining CNN and Transformer, specialized for efficient, high-quality image denoising.
3. retinexformer_fivek: Based on Retinex theory and Transformer, used to enhance brightness and details in low-light images.
4. hvicidnet: A lightweight Transformer model for low-light enhancement and exposure correction.
5. lightdiff: A diffusion model-based framework for unsupervised low-light image enhancement.
6. img2img-turbo-rain (turbo_rain): An efficient SD-Turbo-based model specifically for fast removal of rain from real-world images.
7. s2former: An uncertainty-aware Transformer model for rain streak removal.
8. idt: A Transformer-based model that can remove both rain and raindrops from images.
9. ridcp: An efficient dehazing model utilizing high-quality codebook priors.
10. kanet: An efficient dehazing network using a "locate-and-remove" pipeline.
11. img2img-turbo-snow (turbo_snow): An efficient SD-Turbo-based model for removing snow artifacts while preserving scene details.
12. snowmaster: A real-world image desnowing model optimized with multimodal large language model (MLLM) feedback.
```

---

## 9. 模板 I

```
This image<image> has image quality degradation issues. Please identify the degradation type and select an appropriate restoration tool:
[Super-resolution & General Restoration] real_esrgan
[Denoising] scunet
[Low-light Enhancement] retinexformer_fivek, hvicidnet, lightdiff
[Deraining] turbo_rain, s2former, idt
[Dehazing] ridcp, kanet
[Desnowing] turbo_snow, snowmaster
Please provide your analysis and restoration decision.
```

---

## 10. 模板 J

```
This is an image<image> with degradation issues. Please identify the degradation type and decide which restoration method to use:
Super-resolution/Deblurring/Denoising: real_esrgan
Image Denoising: scunet
Low-light Enhancement: retinexformer_fivek / hvicidnet / lightdiff
Deraining: turbo_rain / s2former / idt
Dehazing: ridcp / kanet
Desnowing: turbo_snow / snowmaster
```

---

## 使用说明

- SFT 训练时，用户提示词从以上 10 种中随机选取（与 RL 训练一致）。
- 每条 SFT 样本格式：
  ```json
  {
    "conversations": [
      {"from": "system", "value": "..."},
      {"from": "user", "value": "模板 A/B/C...之一"}
    ]
  }
  ```
- `<image>` 占位符在训练/推理时由框架替换为实际图像。
