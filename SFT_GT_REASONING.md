# SFT GT 推理文本（每种退化 10 条）

> 格式：`简洁推理句 → <tool_call>{"name": "restore_image", "arguments": {"action": "工具名"}}`
> 说明：推理句建立"观察 → 选工具"的因果映射，不含复杂思考链。

---

## 一、night（低光退化）

### 1.1
观察：图像整体偏暗，细节模糊，低光问题明显。
<tool_call>{"name": "restore_image", "arguments": {"action": "hvicidnet"}}</tool_call>

### 1.2
观察：暗光条件下拍摄，画面亮度严重不足，需要曝光校正。
<tool_call>{"name": "restore_image", "arguments": {"action": "hvicidnet"}}</tool_call>

### 1.3
观察：低光照导致图像模糊、颜色暗淡，适合用低光增强工具。
<tool_call>{"name": "restore_image", "arguments": {"action": "lightdiff"}}</tool_call>

### 1.4
观察：光线不足造成画面昏暗，低光增强可以改善亮度。
<tool_call>{"name": "restore_image", "arguments": {"action": "retinexformer_fivek"}}</tool_call>

### 1.5
观察：夜间场景图像偏暗，retinex 方法能有效增强暗光细节。
<tool_call>{"name": "restore_image", "arguments": {"action": "retinexformer_fivek"}}</tool_call>

### 1.6
观察：曝光严重不足，物体轮廓难以辨认，需要快速曝光校正。
<tool_call>{"name": "restore_image", "arguments": {"action": "hvicidnet"}}</tool_call>

### 1.7
观察：低光图像噪声明显，先做曝光校正再观察后续处理需求。
<tool_call>{"name": "restore_image", "arguments": {"action": "hvicidnet"}}</tool_call>

### 1.8
观察：暗光条件下颜色饱和度低，需要低光增强工具改善视觉效果。
<tool_call>{"name": "restore_image", "arguments": {"action": "lightdiff"}}</tool_call>

### 1.9
观察：整体昏暗的夜间照片，使用 retinex 增强可以更好还原真实色彩。
<tool_call>{"name": "restore_image", "arguments": {"action": "retinexformer_fivek"}}</tool_call>

### 1.10
观察：暗光场景中物体边缘不清晰，需要提升整体亮度。
<tool_call>{"name": "restore_image", "arguments": {"action": "hvicidnet"}}</tool_call>

---

## 二、fog（雾/霾退化）

### 2.1
观察：图像被雾气笼罩，远处景物模糊对比度低，需要去雾处理。
<tool_call>{"name": "restore_image", "arguments": {"action": "ridcp"}}</tool_call>

### 2.2
观察：空气中的雾霾导致画面灰蒙蒙的，kanet 可以有效去雾。
<tool_call>{"name": "restore_image", "arguments": {"action": "kanet"}}</tool_call>

### 2.3
观察：雾气覆盖了远处背景，图像景深受限，需要去雾增强。
<tool_call>{"name": "restore_image", "arguments": {"action": "ridcp"}}</tool_call>

### 2.4
观察：雾霾天气下拍摄的照片能见度低，kanet 的定位移除机制适合。
<tool_call>{"name": "restore_image", "arguments": {"action": "kanet"}}</tool_call>

### 2.5
观察：画面整体发灰，雾气造成细节丢失，ridcp 能恢复清晰度。
<tool_call>{"name": "restore_image", "arguments": {"action": "ridcp"}}</tool_call>

### 2.6
观察：浓雾遮挡了大部分场景，需要强效去雾工具。
<tool_call>{"name": "restore_image", "arguments": {"action": "ridcp"}}</tool_call>

### 2.7
观察：雾气使图像对比度严重下降，kanet 可以改善局部细节。
<tool_call>{"name": "restore_image", "arguments": {"action": "kanet"}}</tool_call>

### 2.8
观察：薄雾笼罩的场景看起来不清晰，去雾能恢复真实面貌。
<tool_call>{"name": "restore_image", "arguments": {"action": "ridcp"}}</tool_call>

### 2.9
观察：雾化效果使物体边缘模糊，代码本先验去雾方法有效。
<tool_call>{"name": "restore_image", "arguments": {"action": "ridcp"}}</tool_call>

### 2.10
观察：远景被雾气遮挡看不清，kanet 能有效提升清晰度。
<tool_call>{"name": "restore_image", "arguments": {"action": "kanet"}}</tool_call>

---

## 三、snow（雪/降雪退化）

### 3.1
观察：图像被雪花覆盖，场景细节被遮挡，需要除雪处理。
<tool_call>{"name": "restore_image", "arguments": {"action": "snowmaster"}}</tool_call>

### 3.2
观察：雪粒在画面中形成噪点和模糊，turbo_snow 可以快速去除。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_snow"}}</tool_call>

### 3.3
观察：积雪覆盖了部分场景，snowmaster 针对真实场景除雪优化。
<tool_call>{"name": "restore_image", "arguments": {"action": "snowmaster"}}</tool_call>

### 3.4
观察：雪花造成的雪花状噪声明显，需要除雪工具恢复细节。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_snow"}}</tool_call>

### 3.5
观察：雪天拍摄的图像有雪花残留和亮度提升，snowmaster 效果较好。
<tool_call>{"name": "restore_image", "arguments": {"action": "snowmaster"}}</tool_call>

### 3.6
观察：雪花遮挡了背景物体，除雪可以还原清晰场景。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_snow"}}</tool_call>

### 3.7
观察：雪粒形成的噪点降低了图像质量，snowmaster 适合精细处理。
<tool_call>{"name": "restore_image", "arguments": {"action": "snowmaster"}}</tool_call>

### 3.8
观察：降雪场景中雪花密集，turbo_snow 能快速还原场景。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_snow"}}</tool_call>

### 3.9
观察：雪天图像整体偏白，需要除雪恢复自然色彩和细节。
<tool_call>{"name": "restore_image", "arguments": {"action": "snowmaster"}}</tool_call>

### 3.10
观察：雪花在镜头前飘落形成模糊点，turbo_snow 基于 SD-Turbo 快速处理。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_snow"}}</tool_call>

---

## 四、rain（雨（通用）退化）

### 4.1
观察：图像中有雨条纹和模糊，需要除雨处理。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_rain"}}</tool_call>

### 4.2
观察：雨水在画面中形成条纹状遮挡，s2former 可以去除雨纹。
<tool_call>{"name": "restore_image", "arguments": {"action": "s2former"}}</tool_call>

### 4.3
观察：雨滴残留在镜头或玻璃上造成斑点，idt 可以去除。
<tool_call>{"name": "restore_image", "arguments": {"action": "idt"}}</tool_call>

### 4.4
观察：细雨形成的雨纹降低了图像清晰度，turbo_rain 适合快速处理。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_rain"}}</tool_call>

### 4.5
观察：雨季拍摄的画面有多种雨干扰，idt 变换器能处理复杂情况。
<tool_call>{"name": "restore_image", "arguments": {"action": "idt"}}</tool_call>

### 4.6
观察：雨水造成的条纹状遮挡明显，s2former 不确定性感知机制有效。
<tool_call>{"name": "restore_image", "arguments": {"action": "s2former"}}</tool_call>

### 4.7
观察：雨滴和水渍混合干扰，turbo_rain 可以快速还原场景。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_rain"}}</tool_call>

### 4.8
观察：雨后场景有残留雨纹，idt 适合去除复杂雨形变。
<tool_call>{"name": "restore_image", "arguments": {"action": "idt"}}</tool_call>

### 4.9
观察：连续雨天照片有均匀雨纹，s2former 可以针对性处理。
<tool_call>{"name": "restore_image", "arguments": {"action": "s2former"}}</tool_call>

### 4.10
观察：雨天图像清晰度下降，turbo_rain 基于 SD-Turbo 能快速去雨。
<tool_call>{"name": "restore_image", "arguments": {"action": "turbo_rain"}}</tool_call>

---

## 使用说明

1. **每条格式**：`观察：... → <tool_call>{"name": "restore_image", "arguments": {"action": "..."}}</tool_call>`
2. **GT 输出结构**：在 SFT 数据中 assistant 回复为"推理句 + 工具调用"，中间换行隔开
