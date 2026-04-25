# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 命令

```bash
# 代码风格（自动修复）
make style

# 代码质量检查（不修改文件）
make quality

# 运行所有测试（v0 + v1）
make test

# 运行单个测试文件
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/path/to/test_file.py

# 运行匹配特定模式的测试
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ -k "test_name"

# 许可证头检查
make license

# 构建包
make build

# 安装可选依赖
pip install -e ".[metrics]"        # 指标（如 BLEU、ROUGE）
pip install -e ".[deepspeed]"      # DeepSpeed ZeRO
pip install -e ".[liger-kernel]"   # Liger Kernel 优化
pip install -e ".[vllm]"           # vLLM 推理后端
pip install -e ".[galore]"         # GaLore 优化器
# 详见 requirements/ 目录下的所有选项
```

项目使用 `uv` 作为首选包管理器。Makefile 目标在 `uv` 可用时使用 `uv run` / `uvx`，否则回退到 `python` / `pytest`。

CLI 命令：`llamafactory-cli <子命令>` 或快捷方式 `lmf <子命令>`。

## 架构

LlamaFactory 有两个并行架构，由 `USE_V1` 环境变量控制：

- **v0（默认）：** `api, webui > chat, eval, train > data, model > hparams > extras`
- **v1（实验性，`USE_V1=1`）：** `trainers > core(base_trainer, data_engine, model_engine) > accelerator, plugins, config > utils`

v1 是下一代重写版本，拥有更清晰的插件架构。v0 仍是主要的生产路径。

### 入口点

`src/llamafactory/cli.py:main()` 是 CLI 入口点。它检查 `USE_V1` 并分派到：
- `src/llamafactory/launcher.py`（v0）
- `src/llamafactory/v1/launcher.py`（v1）

两个启动器都处理分布式训练自动检测——如果有多个 GPU 可用（且未使用 Ray/KTransformers），它们会通过 `torchrun` 重新启动。它们将 `sys.argv[1]` 作为子命令弹出，然后相应地路由。

可用子命令（v0）：`train`、`chat`、`api`、`export`、`webchat`、`webui`、`env`、`version`、`help`
可用子命令（v1）：`sft`、`train`、`dpo`（进行中）、`rm`（进行中）、`chat`、`merge`、`help`

### 训练流程（v0）

```
run_exp() [tuner.py]
  → read_args() → 解析 YAML/JSON 配置（OmegaConf + CLI 覆盖）
  → get_train_args() → 验证参数，生成类型化参数数据类
  → 路由到：run_sft / run_dpo / run_ppo / run_rm / run_pt / run_kto
  → 可选：export_model()
```

训练使用 YAML 配置：`llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml`

### 配置系统

`src/llamafactory/hparams/parser.py` 中的参数解析生成类型化数据类：
- `ModelArguments` — 模型/分词器选择、量化、推理后端
- `DataArguments` — 数据集、模板、截断长度、打包、流式处理
- `FinetuningArguments` — LoRA 配置、训练阶段（sft/dpo/ppo/rm/pt/kto）、优化器插件
- `TrainingArguments` — 扩展 HuggingFace 的 Seq2SeqTrainingArguments

配置文件支持 YAML/JSON。CLI 覆盖使用 OmegaConf 合并。关键环境变量：`ALLOW_EXTRA_ARGS=1` 忽略未识别的参数。

### 关键环境变量

完整参考见 `.env.local`。重要的变量：
- `USE_V1=1` — 使用 v1 架构
- `FORCE_TORCHRUN=1` — 始终使用 torchrun 进行分布式训练
- `USE_RAY=1` — 启用 Ray 集群支持
- `USE_KT=1` — 启用 KTransformers 后端
- `USE_MCA=1` — 启用 Megatron-Core 适配器
- `WANDB_DISABLED=true` — 禁用 W&B 日志记录（测试时必须设置）
- `LLAMAFACTORY_VERBOSITY=DEBUG` — 详细日志
- `NPROC_PER_NODE`、`MASTER_ADDR`、`MASTER_PORT` — 分布式配置

### 关键模块（v0）

| 模块 | 用途 |
|--------|---------|
| `src/llamafactory/model/loader.py` | 加载模型和分词器；应用量化、LoRA、补丁 |
| `src/llamafactory/model/patcher.py` | 模型特定兼容性补丁 |
| `src/llamafactory/data/template.py` | 提示模板；`TEMPLATES` 字典映射模型家族到格式 |
| `src/llamafactory/data/mm_plugin.py` | 多模态（图像/视频/音频）数据处理 |
| `src/llamafactory/data/processor/` | 各阶段数据处理器（监督、成对、预训练、打包、反馈、无监督） |
| `src/llamafactory/data/loader.py` | 数据集加载、合并、预处理流水线 |
| `src/llamafactory/data/formatter.py` | 不同数据类型的字符串/格式化器/工具格式化器 |
| `src/llamafactory/train/sft/` | SFT 训练器；其他阶段（dpo、kto、ppo、pt、rm）遵循相同模式 |
| `src/llamafactory/chat/` | 推理引擎：`hf_engine`、`vllm_engine`、`sglang_engine`、`kt_engine` |
| `src/llamafactory/extras/constants.py` | 项目中使用的枚举和常量 |
| `src/llamafactory/extras/packages.py` | 可选依赖版本检查和可用性标志 |

### v1 架构（`src/llamafactory/v1/`）

v1 使用基于插件的架构，关注点分离清晰：

```
v1/
  launcher.py              — CLI 入口、torchrun 分派、子命令路由
  config/                  — 类型化配置数据类（model_args、data_args、training_args）
  core/
    base_trainer.py        — 抽象训练器，包含 init/train 循环阶段
    data_engine.py         — 数据集加载、索引、格式转换
    model_engine.py        — 模型加载和设置
    utils/batching.py      — 批次生成
    utils/checkpoint.py    — 检查点保存/加载协调
    utils/rendering.py     — 对话模板渲染
  accelerator/             — 分布式训练抽象（DDP、FSDP、DeepSpeed）
  plugins/
    data_plugins/          — 数据处理插件
    model_plugins/         — 模型插件（PEFT/LoRA、内核、并行化、模板）
    sampler_plugins/       — 采样插件
    trainer_plugins/       — 训练器分派（SFT、DPO、RM）+ 分布式训练器
  trainers/                — 具体训练器（目前为 SFT）
  samplers/                — CLI/聊天采样器
  utils/                   — 日志、回调、类型定义、辅助函数
```

### 为新模型添加支持

1. 在 `src/llamafactory/data/template.py` 的 `TEMPLATES` 字典中添加提示模板
2. 在 `src/llamafactory/model/patcher.py` 中添加必要的模型补丁
3. 如需要，在 `src/llamafactory/data/mm_plugin.py` 中添加多模态支持
4. 如需要，在适当的工厂/自动类中注册模型

### 数据集注册

数据集在 `data/dataset_info.json` 中注册。该文件将数据集名称映射到其文件路径、子集和列。添加新数据集时更新此文件，以便训练配置可以通过名称引用（如 `dataset: identity,alpaca_en_demo`）。JSON 结构支持多种数据源：HF Hub、ModelScope Hub、本地文件和脚本。

### 分布式训练

多 GPU 自动使用 `torchrun`。其他后端：
- **Ray：** 可选的 Ray 集群支持（`USE_RAY=1`、`src/llamafactory/train/tuner.py:_ray_training_function`）
- **HyperParallel FSDP2：** `src/llamafactory/train/hyper_parallel/`
- **Megatron-core：** `src/llamafactory/train/mca/`（`USE_MCA=1`）
- **DeepSpeed ZeRO-3：** 通过 DeepSpeed 配置文件配置，详见 `examples/deepspeed/`
- **弹性启动：** 设置 `RDZV_ID` 环境变量以实现容错多节点训练

### 测试

- `tests/` — v0 测试；`tests_v1/` — v1 测试
- 大多数训练测试需要 GPU 硬件
- pytest 标记：`@pytest.mark.slow`、`@pytest.mark.runs_on(['cuda'])`
- 运行测试时必须设置 `WANDB_DISABLED=true`
- 测试配置：`tests/conftest.py`（v0）、`tests_v1/conftest.py`（v1）

### 代码风格

- Ruff 用于代码检查和格式化（行长度 119，Google 风格文档字符串）
- Python 3.11+ 语法
- 字符串使用双引号
- 所有新文件必须包含 Apache 2.0 许可证头（由 `make license` 检查）
