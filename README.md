# FireWorldBench

面向火灾物理世界理解的多模态 benchmark 的**运行与打分入口**。
题目数据（正式测试集）发布在 Hugging Face：**`Guaogua/FireWorldBench`**。

> 本仓库只提供「下载数据 → 运行你的模型 → 计算六项指标」的完整流程。
> 基准的设计动机与详细背景见论文（由论文作者补充）。

## 快速开始

```bash
# 1) 下载数据（国内建议加 --mirror 使用镜像）
pip install -r requirements.txt
python download_dataset.py --mirror
# 产物：FireWorldBench/ 目录，含 main_synthetic 与 mmodalfire_c06 两个正式测试集

# 2) 配置你所用模型的 OpenAI 兼容接口
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-model"

# 3)（可选）先用 2 条做 preflight，确认接口/图像支持
python scripts/run_api.py --provider openai \
  --questions FireWorldBench/main_synthetic/full_test_A/questions.jsonl \
  --source-root FireWorldBench/main_synthetic \
  --output runs/preflight.jsonl --max-requests 2 --model "$OPENAI_MODEL"

# 4) 跑完整测试集（示例：main_synthetic 全部 8,360 题）
python scripts/run_api.py --provider openai \
  --questions FireWorldBench/main_synthetic/full_test_A/questions.jsonl \
  --source-root FireWorldBench/main_synthetic \
  --output runs/main_model.jsonl --model "$OPENAI_MODEL"

# 5) 计算六项指标
python scripts/score_six_metrics.py \
  --gold FireWorldBench/main_synthetic/full_test_A/gold.jsonl \
  --predictions runs/main_model.jsonl \
  --output results/main_model-six-metrics.json
```

用同一流程处理 `mmodalfire_c06/full_test_A`（真实事件，714 题）即可。

## 数据

数据集包含两个**正式测试**子集：

| 子集 | 说明 | QA 数 |
|---|---|---|
| `main_synthetic/full_test_A` | 合成火灾事件（S 轨文本 + I 轨视觉） | 8,360 |
| `mmodalfire_c06/full_test_A` | 真实事件（C06） | 714 |

每个子集内含 `questions.jsonl`（题目）与 `gold.jsonl`（标准答案，打分必需）。
I 轨题目使用图像，图像位于 `assets/events/<event_id>/...`，由题目 `material` 字段中的
`path`/`asset_path` 定位。字段与打分协议详见 HF 数据卡。

## 脚本

| 脚本 | 作用 |
|---|---|
| `download_dataset.py` | 从 Hugging Face 下载数据 |
| `scripts/run_api.py` | 可断点续跑的多模态 API 运行器，**绝不读取 Gold** |
| `scripts/score_six_metrics.py` | 确定性六指标打分器，无任何模型判分 |
| `scripts/score_fg9_*.py` | 打分器内部依赖（归一化、Evidence-F1、Gold-linked Support 等） |
| `scripts/contracts/` | （可选）冻结任务说明，经 `--task-contract` 传给每个题目 |

## 六项指标

| 指标 | 方向 | 适用题型 |
|---|---|---|
| Completion Accuracy (ACC) | 越高越好（主指标） | choice / open |
| Macro-F1 | 越高越好 | choice / open |
| Evidence-F1 | 越高越好 | open |
| Mechanism Alignment | 越高越好 | open |
| Brier Score | 越低越好 | choice / open |
| Gold-linked Support | 越高越好 | open |

- **Completion Accuracy**：choice 为预测选项集合与 Gold 的 Jaccard 相似度均值；open 为必填字段预测正确的比例。
- 结果按 **9 任务 × 2 轨（S/I）× 2 题型（choice/open）= 36 格** 报告，不合并隐藏单格失败。
- 所有指标均为确定性计算，不依赖任何 LLM 判分。

## 许可

- 数据（题目 / Gold / 图像）：**CC BY 4.0**，见 `DATA_LICENSE.md`
- 代码（下载 / 运行 / 打分脚本）：**MIT**，见 `LICENSE`

## 引用

（论文引用由论文作者在此补充）
