# FireWorldBench

![Task](https://img.shields.io/badge/Task-Fire--Physics--VQA-red)
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red)
![Dataset](https://img.shields.io/badge/Dataset-FireWorldBench-blue)

<font size=5><div align='center'>[[📊 数据集](https://huggingface.co/datasets/Guaogua/FireWorldBench)] [[📖 论文](论文链接待补)] [[🏆 排行榜](排行榜链接待补)]</div></font>

> 基准的背景、设计说明与分析见论文（由论文作者补充）。本仓库只负责让你**快速跑起来**。

## 🚀 快速开始（评测你的模型）

完整流程：**下载数据 → 配置模型接口 → 跑题 → 得到六项指标。**

```shell
# 1. 安装依赖
pip install -e .

# 2. 从 Hugging Face 下载数据（约 660MB；国内加 --mirror 走镜像）
cd scripts
bash download.sh --mirror
cd ..

# 3. 配置你的 OpenAI 兼容模型接口
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-model"

# 4.（可选）先抽 2 题快速验证接口/图像支持
python scripts/run_eval.py --split main_synthetic --max-requests 2

# 5. 跑完整测试集并计算六项指标
python scripts/run_eval.py --split main_synthetic
```

完成后，六项指标结果写入 `results/main_synthetic-physical.json`（默认按物理轴 P1-P5 × 2 轨 × 2 题型 = 20 格报告五层能力；可用 `--group-by fire` 看火灾轴五层、`--group-by task` 看细粒度 36 格）。

两个正式测试集：

| 子集 | 说明 | 题数 |
|---|---|---|
| `main_synthetic/full_test_A` | 合成火灾事件（S 轨文本 + I 轨视觉） | 8,360 |
| `mmodalfire_c06/full_test_A` | 真实事件（C06） | 714 |

## 数据

- **数据**（题目 / Gold / 图像）：**CC BY 4.0**，见 `DATA_LICENSE.md`
- **代码**：**Apache-2.0**，见 `LICENSE`

数据发布在 Hugging Face：**[`Guaogua/FireWorldBench`](https://huggingface.co/datasets/Guaogua/FireWorldBench)**

```shell
cd scripts && bash download.sh          # 国内加 --mirror 走镜像
```

每道题带**双轴五层能力标签**：`physical_axis`（物理轴 P1-P5）与 `fire_axis`（火灾轴 T1-T5）。

### 五层双轴划分

物理轴 P：

| P | 英文名 | 中文 | 任务 |
|---|---|---|---|
| P1 | Temporal Evolution Forecasting | 时间预测 | L3-1, L3-2 |
| P2 | Physical Field Perception and Grounding | 场感知与落地 | L1-1, L1-2 |
| P3 | Cross-Field Coupling Understanding | 跨场耦合理解 | L1-3, L2-1, L2-2 |
| P4 | Counterfactual Intervention Reasoning | 反事实干预推理 | L3-3 |
| P5 | Causal Mechanism Attribution | 机制归因 | L2-3 |

火灾轴 T：

| T | 英文名 | 中文 | 任务 |
|---|---|---|---|
| T1 | Fire Evolution Prediction | 演化预测 | L3-1, L3-2, L1-2 |
| T2 | Fire Early Warning | 早期预警 | L1-1 |
| T3 | Fire State Assessment | 状态评估 | L1-3, L2-1, L2-2 |
| T4 | Fire Intervention Decision-Making | 干预决策 | L3-3 |
| T5 | Fire Mechanism Diagnosis | 机制诊断 | L2-3 |

## 安装

```shell
pip install -e .
```

（不用 pip 也可以：`pip install -r scripts/requirements.txt`。）

## 六项指标

| 指标 | 方向 | 适用题型 |
|---|---|---|
| Completion Accuracy (ACC) | 越高越好（主指标） | choice / open |
| Macro-F1 | 越高越好 | choice / open |
| Evidence-F1 | 越高越好 | open |
| Mechanism Alignment | 越高越好 | open |
| Brier Score | 越低越好 | choice / open |
| Gold-linked Support | 越高越好 | open |

- **Completion Accuracy**：choice 为选项集合与 Gold 的 Jaccard 相似度；open 为必填字段预测正确的比例。
- 用 `--group-by` 分组报告，均不合并隐藏单格失败：`physical`（物理轴 P1-P5 × 2 轨 × 2 题型 = 20 格，默认）、
  `fire`（火灾轴 T1-T5 × 2 轨 × 2 题型 = 20 格）、`task`（细粒度，36 格）。
- 全部确定性计算，无模型判分。

## 脚本

| 脚本 | 作用 |
|---|---|
| `scripts/download.sh` | 从 Hugging Face 下载数据 |
| `scripts/run_eval.py` | 一键流程：跑题 + 计算六项指标 |
| `scripts/run_api.py` | 多模态 API 运行器（可断点续跑，**绝不读 Gold**） |
| `scripts/score_six_metrics.py` | 确定性六指标打分器 |
| `scripts/score_fg9_*.py` | 打分器内部依赖 |
| `scripts/contracts/` | （可选）冻结任务说明 |

## 引用

（论文 bibtex 由论文作者补充）
