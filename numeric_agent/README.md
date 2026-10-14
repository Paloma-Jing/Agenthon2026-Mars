# Forecasting Agent

## 控制流程

`__main__.py` 创建预测会话并选择离线流程或语言模型流程。`commander.py` 筛选截止日期内的文本，读取数值分析，通过受限工具生成并验证计划；`agent.py` 以 smolagents ToolCallingAgent 执行数值工具。无文本时仍可依据数值分析决策。

计划控制候选菜单、训练窗口、特征家族和组合权重。默认菜单为 `random_walk`、`gbm`、`solution_ensemble`。随机游走始终保留为候选，但组合内占比没有固定下限。显式权重必须非负且和为 1；`null` 委托训练历史误差规则，组件失败不会暗中重新分配显式权重。

## 数值实现

- `engine.py`：数据截断、会话状态、历史比较、联合模拟与导出。
- `features.py` / `factors.py`：目标滞后、时变市场特征、经济指标和面板质量分析。
- `tree_forecast.py`：直接期限树预测，训练与残差校准之间保留期限隔离。
- `solution_models.py`：阻尼趋势、月度季节性、相对价格特征、联合残差 KDE 和分布混合。
- `evaluation.py` / `evidence.py`：历史真值、公开 M0 误差尺度重建、逐折比较及删时间块稳定性审查。
- `m0.py`：有界失败回退；`volatility.py`：非默认波动实验候选。

择模损失使用公开 M0 重建尺度归一化的复合损失，取算术平均，失败折按 8 计、损失上限为 8。不同模型必须使用可比的折与尺度。随机游走的归一化分数不固定为 1。

## 工具与运行约束

执行工具包括 `inspect_data`、`diagnose_series`、`compute_factors`、`data_quality`、`build_features`、`evaluate_random_walk`、`compare_models`、`review_model_evidence`、`export_forecast`。Agent 无任意代码或 Shell 工具，执行阶段不接收原始文本。

House 与本地调试共享工具。`house.py` 从环境读取平台模型配置；本地调试读取隐藏输入或 `DEEPSEEK_API_KEY`。House 适配器代码按每任务 25 次请求、单次最多 4000 输出 tokens 设限；这些是实现中的固定约束，实际参赛前须重新核对官方规则。默认执行 8 步、指挥官 4 步、运行预算 300 秒。

审计记录位于正式输出目录之外。详见根目录 [README](../README.md) 的安装、运行和验证边界。
