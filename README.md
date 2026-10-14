# Agenthon2026 Mars

面向 Agenthon 2026 Track 2 的金融时间序列概率预测 Agent。使用有界的指挥官与数值执行 Agent，将任务文本、面板分析、历史验证和联合预测样本导出串联起来；同时保留无需 API 的纯数值流程。

## 工作流程

1. **指挥官**读取截止日期内的任务文本与数值质量分析，决定候选模型、训练窗口、特征家族以及组合权重。
2. **执行 Agent**通过受限工具完成诊断、因子分析、历史滚动比较及证据审查。
3. **数值引擎**拟合选定模型，导出多资产、多期限联合预测样本。运行失败时尝试有时间上限的公开 M0 重建回退。

当前默认候选为 `random_walk`、`gbm`、`solution_ensemble`。组合模型包含随机游走、阻尼趋势及树模型组件；指挥官可指定权重，随机游走权重允许为 0，也可委托历史误差规则。旧统计模型与波动实验模型保留在代码中，不是默认 Agent 菜单。

工具不提供任意代码执行、Shell 或网络搜索。运行时打印阶段进度与每 10 秒心跳，审计文件与正式预测输出分开保存。

## 安装

需要 Python 3.13、Git；容器构建另需 Docker。以下命令从本仓库根目录运行：

```bash
git clone https://github.com/Agenthon-2026/track2-forecasting-public.git
git -C track2-forecasting-public checkout 30c8019d997f930eab9ba569d089d12298d86d8c
python3.13 -m venv .venv-agent
source .venv-agent/bin/activate
python -m pip install -r requirements-dev.txt
```

官方代码、任务数据及评分器从上游单独获取，本仓库不复制它们。共享工具包固定为 v2.6.0；`requirements-*.lock.txt` 是开发环境的依赖快照，跨平台安装优先使用 `requirements-dev.txt`。`environment.yml` 提供另一条 Conda 基础环境路线，创建后仍需安装数值 Agent 依赖。

## 运行与检查

无需任何密钥的离线流程，自动按任务卡选择截止日期并检查官方输出门禁：

```bash
python tools/smoke_staged_unit.py --out runs/offline_smoke
python -m pytest tests -q
```

每次运行使用新的输出目录。正式输出恰好为 `forecast.parquet`、`forecast_meta.json` 和 `forecast_rationale.md`；审计记录写入相邻目录。

本地 DeepSeek 调试通过隐藏输入读取密钥：

```bash
python -m numeric_agent forecast \
  --panels track2-forecasting-public/units/t2-F1-ai-mom-2024 \
  --text track2-forecasting-public/units/t2-F1-ai-mom-2024/text \
  --asof 2024-05-31 --out runs/local_agent/forecast.parquet --prompt-key
```

默认本地模型名为 `deepseek-flash`，可用 `--model` 指定。`--mode offline` 绕过语言模型；`--commander off` 保留纯数值 Agent 对照。DeepSeek 路径仅用于本地开发，在受限比赛网络中会被拒绝。

比赛适配器从运行环境读取 `MODEL_ENDPOINT`、`MODEL_TOKEN`、`MODEL_NAME`，本仓库不提供实际凭据。可用 `python tools/smoke_house_stub.py` 检查本地模拟 House 协议。

## 目录

| 路径 | 用途 |
| --- | --- |
| `numeric_agent/` | 指挥官、执行 Agent、预测模型、特征和评估 |
| `tests/` | 自有回归测试 |
| `tools/` | 冒烟检查、历史回测、抽样实验、打包工具 |
| `configs/` | 历史实验配置，非默认 Agent 菜单 |
| `Dockerfile.submission` | Linux/amd64、非 root 的提交镜像 |
| `submission/` | 通用构建与打包说明，不含个人提交物 |

模型与验证细节见 [Agent 说明](numeric_agent/README.md)，镜像打包见 [提交说明](submission/README.md)。`tools/local_score_predictor.py` 需要自行提供私有历史版本源码与 `online_scores.json`；仓库不附带个人成绩。`tools/wiki_health.py` 仅服务于本地未发布的项目 Wiki。

## 验证边界

2026-10-14 清理后验证：110 项自有测试通过；日频任务 `t2-F1-conflicting-texts-2024` 的离线流程通过官方 g0–g3 门禁。未在本轮重建 Docker 镜像或调用真实模型 API。

历史滚动窗口用于择模，不是独立预测准确度验证；文本未按历史 vintage 重放，宏观发布日历与缺失真值仍有限制。公开 M0 重建不等于隐藏评分器的真实结果。本地 House stub 不代表真实 House 服务已验证。通过测试和格式门禁不能证明预测质量提升或预估线上排名。

## 凭据与发布范围

API Key、Team Key 通过隐藏输入或运行时环境传入，不要写入源码。打包器使用权限为 0600 的临时 Team Key 文件，并在正常及异常退出路径清理。`.gitignore` 排除凭据文件、运行记录、个人 Wiki、队伍声明、镜像身份和提交包；Docker 构建上下文只包含应用源码。

该仓库发布的是清理后的源码快照，不包含历史 Git 记录、个人账户配置或比赛提交物。比赛规则、模型可用性及提交资格应以主办方当前说明为准。

## 上游项目

- [Agenthon Track 2](https://github.com/Agenthon-2026/track2-forecasting-public)
- [Agenthon 公共工具包](https://github.com/Agenthon-2026/Agenthon2026-public)
- [smolagents](https://github.com/huggingface/smolagents)

第三方代码与数据遵循各自许可。本仓库未另行授予源码许可证。
