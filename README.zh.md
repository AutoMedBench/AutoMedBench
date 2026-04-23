# [AutoMedBench](https://automedbench.github.io/)

[![Website](https://img.shields.io/badge/Website-automedbench-76B900?style=for-the-badge)](https://automedbench.github.io/)
[![Sandbox](https://img.shields.io/badge/Sandbox-coming_soon-FFD21E?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-2B2B25?style=for-the-badge)](LICENSE)

[English](README.md) · **中文**

> 迈向 *医学自动研究* <br>
> — 面向医学 AI 任务的基础模型智能体基准。

---

## 1. 简介

**AutoMedBench** 评估基础模型在完整医学 AI 研究流程中的自主能力：阅读任务 → 选择方法 → 配置环境 → 验证 → 推理 → 提交。

与其他智能体基准不同，我们评估的是**基础模型本身的智能体能力**，而不是外部框架。每次运行跨五个阶段独立打分（**S1 规划 · S2 环境 · S3 验证 · S4 推理 · S5 提交**），不仅看最终结果:

```
Overall = 0.5 × Agentic (S1–S5 过程评分) + 0.5 × Task (任务指标)
```

## 2. 快速开始

沙箱容器与数据集将托管在 HuggingFace。

<p align="left">
  <a href="#"><img src="https://img.shields.io/badge/Sandbox-coming_soon-FFD21E?style=for-the-badge" alt="Sandbox — coming soon"></a>
</p>

```bash
# 1. 克隆想测试的领域分支
git clone --branch eval_seg --single-branch \
    https://github.com/AutoMedBench/AutoMedBench.git
cd AutoMedBench

# 2. 拉取沙箱容器
docker pull <registry>/automedbench-seg:latest

# 3. 运行一个任务
python eval_seg/docker/orchestrator.py \
    --agent claude-opus-4-6 \
    --task kidney-seg-task \
    --tier lite
```

各分支 README 提供完整的运行参数与数据准备说明。

## 3. 工作流

```
  ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐
  │ S1 规划 │ ─▶ │ S2 环境 │ ─▶ │ S3 验证 │ ─▶ │ S4 推理 │ ─▶ │ S5 提交 │
  └─────────┘    └──────────┘    └─────────────┘    └──────────────┘    └───────────┘
```

- **S1–S3** 由 LLM judge 依据 `plan.md` 与工具调用历史按二元评分规则打分。
- **S4–S5** 由评分器依据输出完整性与格式有效性确定性打分。
- 违反沙箱（例如读取 `/data/private/`）则清零所有阶段。

完整评分规则见各分支的 `SCORING_RUBRICS.md`。

## 4. 更多文档

- **[任务库](docs/task-gallery.md)** — 所有任务、指标与对应分支一览
- **[数据集](docs/dataset-collection.md)** — 使用的数据集与公/私数据拆分方案
- **[难度分级](docs/task-difficulty-tiers.md)** — Lite / Standard / Pro 定义与各级度量目标

## 5. 实时榜单

实时榜单维护于 **[automedbench.github.io/#leaderboard](https://automedbench.github.io/#leaderboard)**。

目前已公开：segmentation 与 image enhancement。VQA 与 report generation 打分进行中。

## 6. 贡献

欢迎临床医生、研究员与工程师参与 — 无需熟悉内部框架。

- **有任务想法？** 开 issue 描述你希望 agent 处理的医学问题：输入形态、真值格式、"完成" 定义。我们负责接入。
- **想新增领域？** Segmentation、VQA、report generation 只是起点 — 任何具有确定性真值的医学 AI 任务都可纳入。
- **在新模型上跑过基准？** 分享结果，我们会把它加入实时榜单。

---

<p align="center">
  <a href="https://www.ucsc.edu/"><img src="assets/ucsc-logo.svg" alt="UC Santa Cruz" height="32"></a>
  &nbsp;&nbsp;×&nbsp;&nbsp;
  <a href="https://www.nvidia.com/"><img src="assets/nvidia-logo.svg" alt="NVIDIA" height="32"></a>
</p>
