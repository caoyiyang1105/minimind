# MiniMind 微调实验

本仓库基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 二次开发,记录我在该项目上完成的预训练复现、SFT、LoRA 等微调实验。原项目的代码结构、训练脚本与使用方式请参阅上游 README。

本说明只覆盖**我新增的代码**及**实验结果存放位置**。

---

## 📁 新增代码概览

所有新增代码都集中在 [`experiments/`](experiments/) 目录下,职责如下:

| 文件 | 作用 |
| --- | --- |
| [experiments/build_hust_finetune_data.py](experiments/build_hust_finetune_data.py) | 把《华中科技大学 2025 研究生手册》PDF 解析、清洗并切分为可用于 SFT 的指令数据;用 `pdftotext -layout` 抽取文本,再按"第 X 条"等正则切段,输出 jsonl。 |
| [experiments/plot_loss.py](experiments/plot_loss.py) | 解析训练日志(`Epoch:[…], loss: …` 格式),把 pretrain / SFT / LoRA 阶段的 loss 落盘成 csv,并画出 loss 曲线 png。 |
| [experiments/eval_compare.py](experiments/eval_compare.py) | 在同一组 prompt 下批量对比多个权重(MiniMind 自训权重 vs HuggingFace 官方权重等)的生成结果,输出 jsonl + Markdown 对照表。 |
| [experiments/eval_hust_lora.py](experiments/eval_hust_lora.py) | 针对 LoRA 微调权重的评测脚本,自动加载 base + LoRA 适配器,与 base 模型同 prompt 对比,输出 jsonl + Markdown。 |
| [experiments/prompts.jsonl](experiments/prompts.jsonl) | 通用领域评测 prompt 集合(实验 1 使用)。 |
| [experiments/hust_prompts.jsonl](experiments/hust_prompts.jsonl) | 围绕《2025 研究生手册》的领域 prompt 集合(实验 2、3 使用)。 |

> 训练脚本本身仍然复用上游的 `trainer/train_pretrain.py`、`trainer/train_full_sft.py`、`trainer/train_lora.py`,不在新增代码之列。

---

## 🧪 实验结果与分析

所有实验产物(训练日志、loss 曲线、生成结果对照表)都在以下两个目录:

- 训练日志:[`experiments/logs/`](experiments/logs/)
- 评测结果与曲线:[`experiments/results/`](experiments/results/)
- 实验数据来源(原始 PDF + 抽取后的纯文本):[`references/hust/`](references/hust/)

> 模型权重(`*.pth` / `*.safetensors`)和训练数据集因体积较大,**未纳入版本控制**(已在 `.gitignore` 中排除)。

### 实验 1:MiniMind 预训练 + SFT 基线复现

在原项目数据集上跑通 Pretrain → SFT 全流程,验证训练链路无误,并与上游公开权重做对照。

| 内容 | 路径 |
| --- | --- |
| 训练日志 | [`experiments/logs/pretrain_mini.log`](experiments/logs/pretrain_mini.log)、[`experiments/logs/full_sft_mini.log`](experiments/logs/full_sft_mini.log) |
| Loss 数据 | [`experiments/results/pretrain_loss.csv`](experiments/results/pretrain_loss.csv)、[`experiments/results/sft_loss.csv`](experiments/results/sft_loss.csv)、[`experiments/results/loss_records.csv`](experiments/results/loss_records.csv) |
| Loss 曲线 | [`experiments/results/loss_curve.png`](experiments/results/loss_curve.png) |
| 自训 vs 官方权重对比 | [`experiments/results/model_comparison.md`](experiments/results/model_comparison.md)、[`experiments/results/model_comparison.jsonl`](experiments/results/model_comparison.jsonl) |

### 实验 2:基于《2025 研究生手册》的领域微调(Full SFT vs LoRA)

在实验 1 得到的 SFT 权重(`base_self`)基础上,用 [`build_hust_finetune_data.py`](experiments/build_hust_finetune_data.py) 构造的领域数据继续微调,分别尝试 **全参数 SFT** 与 **LoRA**,并在三种解码策略(默认采样、贪心、低温)下评估输出质量。

| 内容 | 路径 |
| --- | --- |
| Full SFT 训练日志 / loss / 曲线 | [`experiments/logs/full_sft_hust_handbook.log`](experiments/logs/full_sft_hust_handbook.log)、[`experiments/results/hust_full_sft_loss.csv`](experiments/results/hust_full_sft_loss.csv)、[`experiments/results/hust_full_sft_loss_curve.png`](experiments/results/hust_full_sft_loss_curve.png) |
| Full SFT 输出对比 | [`experiments/results/hust_full_sft_comparison.md`](experiments/results/hust_full_sft_comparison.md)(默认采样)、[`experiments/results/hust_full_sft_comparison_greedy.md`](experiments/results/hust_full_sft_comparison_greedy.md)(贪心)、[`experiments/results/hust_full_sft_comparison_lowtemp.md`](experiments/results/hust_full_sft_comparison_lowtemp.md)(低温) |
| LoRA 训练日志 / loss / 曲线 | [`experiments/logs/lora_hust_handbook.log`](experiments/logs/lora_hust_handbook.log)、[`experiments/results/hust_lora_loss.csv`](experiments/results/hust_lora_loss.csv)、[`experiments/results/hust_lora_loss_curve.png`](experiments/results/hust_lora_loss_curve.png) |
| LoRA 输出对比 | [`experiments/results/hust_lora_comparison.md`](experiments/results/hust_lora_comparison.md) |

### 实验 3:QA 风格数据微调

把研究生手册重新组织为问答(QA)格式后再做 Full SFT,观察问答指令模板对回答结构与事实性的影响。

| 内容 | 路径 |
| --- | --- |
| 训练日志 / loss / 曲线 | [`experiments/logs/full_sft_hust_qa.log`](experiments/logs/full_sft_hust_qa.log)、[`experiments/results/hust_qa_sft_loss.csv`](experiments/results/hust_qa_sft_loss.csv)、[`experiments/results/hust_qa_sft_loss_curve.png`](experiments/results/hust_qa_sft_loss_curve.png) |
| 输出对比 | [`experiments/results/hust_qa_sft_comparison.md`](experiments/results/hust_qa_sft_comparison.md) |

---

## 🔗 上游项目

- 原始仓库:<https://github.com/jingyaogong/minimind>
- 关于模型架构、训练流程、依赖安装、数据准备等内容,以原仓库 README 为准。
