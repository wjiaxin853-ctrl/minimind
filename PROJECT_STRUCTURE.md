# MiniMind 项目文件结构说明

本文件以树状结构详细介绍了 MiniMind 项目中各个目录和文件的具体作用，帮助开发者快速理解项目架构。

```text
minimind/
├── dataset/                    # 数据处理与加载模块
│   ├── __init__.py
│   ├── dataset.md              # 数据集介绍文档
│   └── lm_dataset.py           # 数据加载器 (PyTorch Dataset)，支持 Pretrain 和 SFT 数据的读取
├── images/                     # 项目可视化资产
│   ├── logo.png / logo2.png    # 项目图标
│   ├── minimind-3.gif          # 动态效果展示
│   ├── LLM-structure.jpg       # Dense 模型架构图
│   ├── LLM-structure-moe.jpg   # MoE 模型架构图
│   ├── rl-structure.jpg        # 强化学习架构图
│   ├── pretrain_loss.jpg       # 预训练阶段 Loss 曲线示例
│   ├── sft_loss.jpg            # SFT 阶段 Loss 曲线示例
│   ├── ppo_loss.jpg            # PPO 训练 Loss 曲线
│   ├── grpo_loss.jpg           # GRPO 训练 Loss 曲线
│   ├── agent_rl_loss.jpg       # Agent RL 训练 Loss 曲线
│   ├── benchmark_radar.jpg     # 模型评测雷达图
│   ├── agent_webui.jpg         # Web 界面效果截图
│   ├── dataset.jpg             # 数据分布/处理流程图
│   ├── with_huggingface.png    # 社区链接图标
│   └── with_modelscope.png     # 社区链接图标
├── minimind-3/                 # 官方发布的成品模型目录 (Transformers 格式)
│   ├── config.json             # 模型超参数配置文件 (层数、维度等)
│   ├── model.safetensors       # 核心模型权重文件 (安全张量格式)
│   ├── tokenizer.json          # 分词器词表文件
│   ├── tokenizer_config.json   # 分词器配置 (含 Chat Template)
│   ├── generation_config.json  # 生成策略配置 (温度、Top-P 等)
│   ├── chat_template.jinja     # 对话模板定义
│   └── minimind.modelfile      # Ollama 导入使用的配置文件
├── model/                      # 模型核心架构 (源代码)
│   ├── __init__.py
│   ├── model_lora.py           # LoRA (Low-Rank Adaptation) 的原生实现
│   ├── model_minimind.py       # MiniMind 模型核心代码 (包含 Transformer 和 MoE 架构)
│   ├── tokenizer.json          # 源代码目录下的分词器配置文件
│   └── tokenizer_config.json   # 源代码目录下的分词器配置
├── scripts/                    # 工具与演示脚本
│   ├── chat_api.py             # 简易对话推理示例
│   ├── convert_model.py        # 权重转换工具 (如 LoRA 权重合并)
│   ├── eval_toolcall.py        # 工具调用 (Tool Call) 能力评测脚本
│   ├── serve_openai_api.py     # 兼容 OpenAI 协议的 FastAPI 服务端
│   └── web_demo.py             # 基于 Streamlit 的极简网页对话 Demo
├── trainer/                    # 训练核心脚本 (按阶段划分)
│   ├── rollout_engine.py       # 强化学习 (RL) 的采样生成引擎
│   ├── train_agent.py          # Agentic RL 训练脚本
│   ├── train_distillation.py   # 模型蒸馏训练脚本
│   ├── train_dpo.py            # RLHF - DPO 训练脚本
│   ├── train_full_sft.py       # 指令微调 (SFT) 全参数训练脚本
│   ├── train_grpo.py           # RLAIF - GRPO 训练实现
│   ├── train_lora.py           # 基于 LoRA 的微调脚本
│   ├── train_ppo.py            # RLAIF - PPO 训练脚本
│   ├── train_pretrain.py       # 基础预训练脚本
│   ├── train_tokenizer.py      # 分词器训练示例
│   └── trainer_utils.py        # 训练过程工具函数
├── .gitignore                  # Git 忽略配置
├── CODE_OF_CONDUCT.md          # 行为准则
├── LICENSE                     # 项目许可证 (Apache-2.0)
├── README.md                   # 中文主页文档
├── README_en.md                # 英文主页文档
├── eval_llm.py                 # 项目主要的 CLI 推理与对话测试脚本
├── pyproject.toml              # 项目管理配置文件
├── requirements.txt            # Python 环境依赖列表
└── uv.lock                     # uv 包管理器锁文件
```

## 核心文件作用简述

1. **[minimind-3/](file:///Users/wjx/Documents/wjx/minimind/minimind-3)**: 你通过 `modelscope` 下载的官方成品模型文件夹。它包含了可以直接被 `transformers` 库加载的所有必要文件。
2. **[eval_llm.py](file:///Users/wjx/Documents/wjx/minimind/eval_llm.py)**: 最常用的交互入口，支持加载 `.pth` 权重或 `transformers` 格式模型进行对话测试。
3. **[model_minimind.py](file:///Users/wjx/Documents/wjx/minimind/model/model_minimind.py)**: 项目的“灵魂”，实现了类 Qwen3 的 Transformer 结构，支持混合专家 (MoE) 模式。
4. **[train_pretrain.py](file:///Users/wjx/Documents/wjx/minimind/trainer/train_pretrain.py)**: 用于第一阶段的知识学习，让模型学会“认字”和“说话”。
5. **[train_full_sft.py](file:///Users/wjx/Documents/wjx/minimind/trainer/train_full_sft.py)**: 用于第二阶段的指令遵循，让模型学会“听话”和“当助手”。
6. **[web_demo.py](file:///Users/wjx/Documents/wjx/minimind/scripts/web_demo.py)**: 提供可视化的聊天界面，方便直观展示模型的对话能力。
