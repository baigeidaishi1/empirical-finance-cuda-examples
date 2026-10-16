# Empirical Finance — CUDA Synthetic Examples

本包用三个合成市场演示从量价特征、神经网络训练到组合记账的完整流程。可运行 MLP、GRU、GNN、Attention 和 Transformer，并比较同一配置下的 CPU／CUDA 执行结果。

This package uses three synthetic markets to demonstrate the workflow from price/volume features and neural-network training to portfolio accounting. It includes MLP, GRU, GNN, Attention, and Transformer, with CPU/CUDA execution for the same configurations.

**所有收益均为合成示例结果，不代表真实历史表现或未来投资收益。** 本包无需行情缓存、数据库账户或真实数据接口；也不连接交易账户。

**All returns are synthetic example results, not historical investment performance or evidence of future returns.** No market-data cache, database account, or real-data interface is required. The package does not connect to a trading account.

## 三个市场 / Three markets

| 配置 / Configuration | 市场特点 / Market behavior | 观察重点 / What to inspect |
|---|---|---|
| `weak_signal` | 持续信号较弱，随机扰动较明显。/ Weak persistent signals with prominent random variation. | 低信噪比下的验证表现、预测排序和交易成本。/ Validation, rankings, and trading costs when signals are weak. |
| `trend` | 个股具有更容易识别的持续趋势。/ Stocks have more identifiable persistent trends. | 模型是否学到价格历史中的方向信息。/ Whether the model captures directional information from price history. |
| `regime_shift` | 市场生成规律在样本中发生变化。/ The data-generating process changes within the sample. | 训练期规律能否延续至后续时段，以及模型面对变化时的局限。/ Whether patterns persist into later periods and how models behave when conditions change. |

三个市场使用相同的默认随机种子 `17`，但生成规则不同；相同种子不意味着相同行情。每次运行都会重新生成数据、拟合模型、计算预测并回测。

All three markets use the default seed `17`, but have different generation rules. A shared seed does not mean identical prices. Each run generates data, fits models, produces predictions, and evaluates portfolios.

## 快速运行 / Quick start

以下命令在解压后的项目根目录中逐条运行。完整安装、校验与阅读结果的步骤见同目录的 [REPRODUCTION INSTRUCTION.md](REPRODUCTION%20INSTRUCTION.md)。

Run the following commands individually from the extracted project root. For installation, verification, and interpretation of outputs, see [REPRODUCTION INSTRUCTION.md](REPRODUCTION%20INSTRUCTION.md).

先确认目录内容包含 `run.py`、`configs` 和 `src`。

First confirm that the directory contains `run.py`, `configs`, and `src`.

```powershell
Get-ChildItem
```

创建 Python 3.12 环境并选择一种依赖安装方式。以下为 CPU 示例；CUDA 安装见操作说明。

Create a Python 3.12 environment and choose one dependency setup. This example installs the CPU environment; CUDA instructions are in the reproduction guide.

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

```powershell
.\.venv\Scripts\python.exe run.py doctor
```

```powershell
.\.venv\Scripts\python.exe run.py run --example all --device cpu --out outputs/demo_01
```

```powershell
.\.venv\Scripts\python.exe run.py verify --run outputs/demo_01
```

## 模型与计算方式 / Models and execution

| 模型 / Model | CPU | CUDA |
|---|---|---|
| MLP | 常规 FP32 执行 / Eager FP32 | FP32 CUDA Graph 训练 / FP32 CUDA Graph training |
| GRU | 常规 FP32 执行 / Eager FP32 | FP32 CUDA Graph 训练 / FP32 CUDA Graph training |
| GNN | 常规 FP32 执行 / Eager FP32 | FP32 CUDA Graph 训练 / FP32 CUDA Graph training |
| Attention | 常规 FP32 执行 / Eager FP32 | 优化后的常规 FP32 执行 / Optimized eager FP32 |
| Transformer | 常规 FP32 执行 / Eager FP32 | 优化后的常规 FP32 执行 / Optimized eager FP32 |

CUDA 路径复用设备端数据，减少训练中的重复传输和同步；适用模型使用 CUDA Graph 减少重复提交算子的开销。计算仍采用 FP32，不启用 TF32 或混合精度。训练和验证批次均保持每批 2 个交易日。CUDA Graph 的创建本身有开销，小型示例不一定比 CPU 更快，也不能用本包的耗时推断大型训练任务的固定加速比例。

The CUDA path reuses device-resident data and reduces repeated transfers and synchronization. Eligible models use CUDA Graphs to reduce repeated operator-launch overhead. Computation remains FP32; TF32 and mixed precision are disabled. Training and validation batches both contain two sessions. Graph creation has an initial cost, so a small example may not be faster than CPU execution. Timings here do not establish a fixed speedup for larger workloads.

## 默认实验设计 / Default experiment

- **样本 / Sample：** 每个市场 360 个模拟交易日、32 只合成股票。/ 360 simulated sessions and 32 synthetic stocks per market.
- **输入 / Inputs：** 15 个因果量价特征，仅使用当日及此前的信息；MLP／GNN 读取当日特征，GRU／Attention／Transformer 读取最长 20 日特征序列。/ Fifteen causal price/volume features using current and prior observations. MLP/GNN use current features; GRU/Attention/Transformer use sequences of up to 20 sessions.
- **标签 / Target：** 下一交易日开盘至其后第 5 个交易日开盘的收益，用于横截面排序学习。/ Return from the next session's open to the open five sessions later, used for cross-sectional ranking.
- **切分 / Split：** 前 220 日为初始训练区间，随后 60 日为验证区间，第 280 日起进入测试区间；边界按标签结束日并留出 2 日隔离。/ The first 220 sessions form the initial training interval, the next 60 the validation interval, and testing starts at zero-based session 280. Boundary eligibility uses label end dates and a two-session purge.
- **预处理 / Preprocessing：** 训练样本估计 1%／99% 截尾点、均值和标准差；验证与测试使用对应已拟合变换。/ Training observations determine the 1st/99th percentile clipping bounds, means, and standard deviations; validation and test observations use the fitted transform.
- **拟合 / Fitting：** 默认隐藏维度 16、2 层、最多 6 轮、早停耐心值 2、种子 17；根据验证表现选择轮数，再用测试前可用样本重新拟合。/ Defaults are hidden size 16, two layers, at most six epochs, patience two, and seed 17. Validation chooses the epoch count, followed by a refit on eligible pre-test observations.
- **组合 / Portfolio：** 初始资金 1,000,000，最多持有排名前 5 只，目标股票仓位 95%，每 20 个交易日调仓，下一开盘成交，单边成交金额收取 10 个基点成本。/ Initial capital is 1,000,000, with up to five top-ranked stocks, 95% target equity allocation, rebalancing every 20 sessions, next-open execution, and a 10-basis-point cost on each side's traded value.

组合账本允许分数股，是用于理解信号、持仓与成本关系的简化模拟。它没有真实市场的涨跌停、停牌、冲击成本、公司行动或券商成交约束。测试末端仅使用能够计算完整标签的预测日。

The portfolio ledger permits fractional shares and is a simplified simulation of signals, holdings, and costs. It does not implement real-market price limits, suspensions, market impact, corporate actions, or broker execution restrictions. Prediction dates at the end of the test period are restricted to those with complete labels.

默认配置按从 0 开始的序号计数：120 日特征预热后，初始可拟合日期为 120–211，验证为 220–271，最终拟合为 120–271，测试预测为 280–353。组合在最后一个预测日后继续按开盘价计值，并在第 359 日开盘清仓；预测结束日与净值结束日因此不同。

With zero-based session indices, the 120-session feature warm-up leaves initial fit dates 120–211, validation dates 220–271, final refit dates 120–271, and test score dates 280–353 under the defaults. The portfolio continues marking at open prices after the last score date and liquidates at session 359's open, so the score and portfolio-value series end on different dates.

## 文件与结果 / Files and outputs

| 路径 / Path | 用途 / Purpose |
|---|---|
| `run.py` | 统一命令入口。/ Unified command entry point. |
| `configs/` | 三个市场的生成、训练与组合配置。/ Generation, training, and portfolio settings for the three markets. |
| `src/` | 合成数据、特征、模型、训练、模拟账本与校验实现。/ Synthetic data, features, models, training, portfolio simulation, and verification. |
| `tests/` | 边界、数据处理和结果完整性检查。/ Boundary, data-processing, and output-integrity tests. |
| `requirements*.txt` | 基础依赖及独立 CPU／CUDA PyTorch 安装配置。/ Base dependencies and separate CPU/CUDA PyTorch installation requirements. |
| `REPRODUCTION INSTRUCTION.md` | 从环境建立到查看结果的逐步说明。/ Step-by-step environment setup, execution, and output inspection. |
| `MANIFEST.json` | 发布包文件的 SHA-256 清单。/ SHA-256 manifest for release files. |
| `outputs/` | 运行后生成；含 CSV、模型检查点、曲线、配置和校验清单。/ Created by a run; contains CSVs, checkpoints, curves, configurations, and integrity manifests. |

运行目录中的 `runtime.json` 记录软件版本和源码哈希，不记录个人路径。`run.py test` 执行 CPU 测试；CUDA 环境可通过 `run.py test --device cuda` 另做五类模型的常规／优化执行对照。具体命令与警告说明见操作说明书。

Each run's `runtime.json` records software versions and source hashes without personal paths. `run.py test` runs CPU tests; a CUDA environment can add eager/optimized comparisons for all five model families with `run.py test --device cuda`. Full commands and warning interpretation are provided in the reproduction instructions.

`src/efcuda` 中各模块的分工如下。`run.py` 是入口，网络结构与完整实验流程分别位于不同模块。

Modules under `src/efcuda` have the following responsibilities. `run.py` is the entry point; network definitions and the complete experiment workflow live in separate modules.

| 模块 / Module | 职责 / Responsibility |
|---|---|
| `cli.py` | 解析命令，选择设备、市场与模型，提供环境和文件检查。/ Parse commands, select the device, market, and model, and expose environment and integrity checks. |
| `workflow.py` | 串联合成、切分、验证选轮数、最终拟合、预测、组合计算与结果保存。/ Coordinate generation, splitting, validation-based epoch selection, refitting, scoring, portfolio calculation, and output persistence. |
| `data.py` | 生成三个市场，构造 15 个特征与标签，定义切分并拟合预处理参数。/ Generate the three markets, construct 15 features and labels, define splits, and fit preprocessing parameters. |
| `models.py` | 定义五类网络的结构和前向计算。/ Define the five network architectures and their forward computations. |
| `engine.py` | 执行 FP32 训练、验证、早停和推理，管理 CPU／CUDA 张量与执行策略。/ Implement FP32 fitting, validation, early stopping, and inference, including CPU/CUDA tensor and execution management. |
| `native_graph.py` | 为适用网络捕获和复用 CUDA Graph，并处理形状不适用时的常规执行。/ Capture and reuse CUDA Graphs for eligible networks, with eager execution for incompatible shapes. |
| `ledger.py` | 将预测排序转换为前 5 只持仓，计算下一开盘交易、费用、净值与风险指标。/ Convert rankings to top-five positions and calculate next-open trades, costs, portfolio values, and risk metrics. |

初次阅读可沿着 `run.py → cli.main → workflow.run_examples` 找到整条流程，再按需要查看 `data`、`engine`／`models` 和 `ledger`。源码注释重点解释日期边界、张量形状、预处理参数的拟合范围、CUDA Graph 的限制以及成交记账顺序。操作说明书第 8 步列出了对应函数，便于从一个输出追溯到它的计算过程。

Start with `run.py → cli.main → workflow.run_examples` to see the complete workflow, then inspect `data`, `engine`/`models`, and `ledger` as needed. Source comments explain date boundaries, tensor shapes, preprocessing fit scope, CUDA Graph restrictions, and trade-accounting order. Step 8 of the reproduction instructions identifies the relevant functions so that an output can be traced back to its calculation.

汇总 CSV 用于跨市场、跨模型查看累计收益、风险和成本；逐模型输出保留预测、净值、交易、模型参数和训练记录。预测分数用于排序，不应直接读成预测收益百分比。检查点是本次拟合的结果，不是调用真实行情后生成的交易建议。

The summary CSV supports comparisons of return, risk, and costs across markets and models. Per-model outputs retain predictions, portfolio values, trades, parameters, and training records. Scores are ranking signals, not directly interpretable forecast-return percentages. Checkpoints contain the current fit, not investment recommendations generated from real prices.

## 复现边界 / Reproducibility

参考运行环境为 Python 3.12、NumPy 2.5.3、SciPy 1.18.1、PyTorch 2.14.0+cu130。CPU 执行可以在这套 CUDA 版 PyTorch 中进行；CPU 专用 wheel 的全新安装不属于这里记录的环境验收范围。

The reference runtime is Python 3.12, NumPy 2.5.3, SciPy 1.18.1, and PyTorch 2.14.0+cu130. CPU execution can run within this CUDA-enabled PyTorch installation; a fresh installation of the CPU-only wheel is outside the recorded environment validation scope.

固定配置和种子有助于复现，但不同 PyTorch 版本、设备、驱动和数值内核仍可能导致差异。比较 CPU 与 CUDA 时应保留两份独立输出，核对环境信息、配置、预测和结果；文件哈希一致只说明文件字节一致，不能代替统计或投资有效性检验。

Fixed configurations and seeds support reproducibility, but PyTorch versions, devices, drivers, and numerical kernels can still introduce differences. Keep separate CPU and CUDA outputs and compare their environments, configurations, predictions, and results. Matching file hashes establish byte integrity, not statistical or investment validity.

许可证为 MIT，见 [LICENSE](LICENSE)。/ Licensed under MIT; see [LICENSE](LICENSE).
