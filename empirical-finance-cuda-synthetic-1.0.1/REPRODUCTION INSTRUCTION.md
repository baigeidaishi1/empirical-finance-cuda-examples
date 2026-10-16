# 复现操作说明 / Reproduction instructions

本说明以 Windows PowerShell 为例。包内所有行情均由 Python 程序生成，包含弱信号、趋势、状态切换三个市场及五类神经网络。无需下载历史行情，无需数据服务密钥，也不需要将真实股票文件放入项目。

This guide uses Windows PowerShell. Python generates every market observation in the package. The examples cover weak-signal, trend, and regime-shift markets and five neural model families. No historical-data download, data-service key, or real-stock file is required.

只复制代码框内的命令，逐条执行；不要复制代码框外的文字、反引号或 PowerShell 自己显示的提示符。说明中的 `outputs/demo_01` 等均为相对于项目根目录的路径。

Copy only the command inside each code block and execute commands one at a time. Do not copy surrounding prose, backticks, or the prompt displayed by PowerShell. Paths such as `outputs/demo_01` are relative to the project root.

## 1. 解压并确认目录 / Extract and locate the project

如同时取得 ZIP 与同名 `.sha256` 文件，把两者放在同一下载目录。先在文件资源管理器打开该目录，点击地址栏，输入 `powershell` 并按回车；该方法不依赖右键菜单是否提供 PowerShell 选项。

If the ZIP and its matching `.sha256` file are supplied together, place them in the same download directory. Open that directory in File Explorer, click its address bar, enter `powershell`, and press Enter. This works without relying on a PowerShell entry in the context menu.

```powershell
Get-ChildItem
```

计算 ZIP 的 SHA-256，并与校验文件中的完整哈希逐字符比较。以下使用本包名称；如下载文件被重命名，应替换成实际名称。

Compute the ZIP's SHA-256 and compare the full hash with the checksum file. These commands use this package's filename; substitute the actual name if it was renamed during download.

```powershell
Get-FileHash -LiteralPath .\empirical-finance-cuda-synthetic-1.0.1.zip -Algorithm SHA256
```

```powershell
Get-Content -LiteralPath .\empirical-finance-cuda-synthetic-1.0.1.zip.sha256
```

哈希是文件内容的摘要，不是密码。匹配表示下载文件与校验文件记录的字节相符；应从可信渠道同时取得压缩包与校验文件。ZIP 自身的校验文件放在 ZIP 外部。

A hash is a digest of file contents, not a password. A match means the download's bytes match the checksum record. Obtain both files from a trusted source. The checksum for the ZIP itself is distributed outside the ZIP.

在资源管理器中选择“全部解压”，进入实际包含 `run.py` 的目录，再通过地址栏输入 `powershell`。不要直接在压缩包预览中运行代码。

Choose “Extract All” in File Explorer, enter the directory that actually contains `run.py`, and open PowerShell through the address bar again. Do not run the project from the archive preview.

```powershell
Get-ChildItem
```

此时应看到以下结构。`.venv`、`.venv-gpu` 与 `outputs` 是之后在自己电脑上生成的目录，初始压缩包不需要包含它们。

You should see the following structure. `.venv`, `.venv-gpu`, and `outputs` are created later on your computer and need not be included in the archive.

```text
empirical-finance-cuda-synthetic-1.0.1/
├── run.py
├── README.md
├── REPRODUCTION INSTRUCTION.md
├── LICENSE
├── MANIFEST.json
├── requirements.txt
├── requirements-cpu.txt
├── requirements-cuda.txt
├── configs/
├── src/
└── tests/
```

## 2. 选择 Python 环境 / Choose a Python environment

本包以 Python 3.12 为运行基线。先列出已有解释器；这条命令本身不安装 Python。

Python 3.12 is the runtime baseline. First list installed interpreters; this command does not install Python.

```powershell
py --list
```

如果没有 Python 3.12，应先安装兼容解释器，再返回此步骤。已有 Python 的开发者也可以使用自己的兼容虚拟环境，但需要按对应依赖文件安装库。不能仅因为系统存在 `python` 命令就假设它是所需解释器。

If Python 3.12 is absent, install a compatible interpreter before continuing. Developers may use an existing compatible virtual environment, provided they install the corresponding requirements. The presence of a `python` command alone does not establish that it points to the intended interpreter.

**CPU 与 GPU 是两种环境选择，不需要先完成全部 CPU 训练才能运行 GPU。** 没有兼容 NVIDIA GPU 时选择 CPU；需要明确比较两条路径时分别保留环境和输出目录。

**CPU and GPU are alternative environments; completing CPU training is not a prerequisite for GPU execution.** Choose CPU when a compatible NVIDIA GPU is unavailable. Keep separate environments and output directories when comparing the two paths.

### 2A. CPU 环境 / CPU environment

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.version)"
```

第一条命令使用已经安装的 Python 3.12 建立项目虚拟环境，不会另外下载一套 Python。后续命令明确调用 `.venv` 中的解释器，因此依赖安装和运行使用同一环境。pip 的新版本提示是通知，复现期间无需因此升级 pip。

The first command creates a project environment from the installed Python 3.12; it does not download another Python distribution. Subsequent commands explicitly call the environment's interpreter, keeping installation and execution in the same environment. A pip update notice is informational; there is no need to upgrade pip for reproduction.

### 2B. CUDA 环境 / CUDA environment

```powershell
py -3.12 -m venv .venv-gpu
```

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

```powershell
.\.venv-gpu\Scripts\python.exe run.py doctor
```

CUDA 环境需要兼容的 NVIDIA GPU 和驱动。应以 `doctor` 的实际检测为准；安装带 CUDA 的 PyTorch 不等于设备一定可用。不要把另一台电脑的 `.venv-gpu` 直接复制过来使用。

The CUDA environment requires a compatible NVIDIA GPU and driver. Use the actual `doctor` output to establish availability; installing CUDA-enabled PyTorch does not establish that a usable GPU is present. Recreate the environment on each computer rather than copying another machine's `.venv-gpu` directory.

本包不需要手写 C++／CUDA 扩展。依赖安装从软件包源下载库；完成安装后，三种合成示例无需访问网络。运行日志应保留实际 Python、PyTorch 和设备信息，以便解释跨设备差异。

The package does not require custom C++/CUDA extensions. Dependency installation downloads libraries from package sources; after installation, the three synthetic examples do not need network access. Keep the recorded Python, PyTorch, and device details when interpreting differences across computers.

参考运行环境为 Python 3.12、NumPy 2.5.3、SciPy 1.18.1 和 PyTorch 2.14.0+cu130；其中 CPU 执行使用同一套 CUDA 版 PyTorch 的 CPU 路径。另列的 CPU 专用依赖文件用于安装 CPU wheel，其全新安装未作为已完成的环境验收项目。

The reference runtime uses Python 3.12, NumPy 2.5.3, SciPy 1.18.1, and PyTorch 2.14.0+cu130. Its CPU execution uses the CPU path of that same CUDA-enabled installation. The separate CPU requirements file installs a CPU-only wheel; a fresh installation of that wheel is not recorded as a completed environment validation.

## 3. 检查包与运行环境 / Check the package and environment

以下使用 CPU 环境。如果选择 CUDA 环境，将命令开头替换为 `.\.venv-gpu\Scripts\python.exe`。

The following commands use the CPU environment. For CUDA, replace the interpreter at the beginning with `.\.venv-gpu\Scripts\python.exe`.

```powershell
.\.venv\Scripts\python.exe run.py --help
```

```powershell
.\.venv\Scripts\python.exe run.py doctor
```

```powershell
.\.venv\Scripts\python.exe run.py verify-files
```

```powershell
.\.venv\Scripts\python.exe run.py test
```

默认 `test` 执行 12 项 CPU 测试，检查数据因果性、标签与切分、预处理、账本、模型保存及文件完整性；不以“缺少 GPU”为由跳过这些测试。若已配置 CUDA 环境，再执行下面的命令：它先运行同一组 CPU 测试，随后增加五类模型各一组常规／优化 CUDA 执行对照。

The default `test` command runs 12 CPU tests covering causal data processing, labels and splits, preprocessing, the ledger, saved models, and file integrity. These tests do not require a GPU and are not skipped when one is absent. With a configured CUDA environment, the following command runs those CPU tests first, then adds one eager/optimized CUDA comparison for each of the five model families.

```powershell
.\.venv-gpu\Scripts\python.exe run.py test --device cuda
```

CUDA 对照检查训练曲线、选择轮数、模型权重和预测差异。参考环境中的五组检查均通过，所比较数值的最大差为 0；这一结果只适用于对应的小型测试，不等于所有设备和配置都必然逐位相同。

CUDA comparisons check training curves, selected epochs, weights, and predictions. All five comparisons passed in the reference runtime, with a maximum observed difference of zero. This result applies to those small test cases; it does not guarantee bitwise equality across every device and configuration.

`--help` 列出命令；`doctor` 检查运行环境；`verify-files` 检查发布包中的文件完整性；`test` 运行程序测试。它们回答的问题不同。文件校验通过不代表已经训练过模型，单元测试通过也不代表模型具有真实市场获利能力。

`--help` lists commands, `doctor` inspects the runtime, `verify-files` checks release-file integrity, and `test` runs software tests. These checks serve different purposes. File verification does not mean training has occurred, and passing software tests does not establish real-market profitability.

`run.py` 为本次进程加入包内源码路径，再调用 `src` 中的模块；按照这些入口运行，无需额外执行可编辑安装。PowerShell 负责启动程序，实际生成数据、训练和计算由 Python／PyTorch 完成。

`run.py` adds the source directory to its process's import path and calls modules under `src`. These entry points do not require an editable installation. PowerShell launches the process; Python and PyTorch generate data, train models, and compute results.

## 4. 运行三个合成示例 / Run the three synthetic examples

一次运行三个市场、每个市场五类模型：

Run all three markets with five model families per market:

```powershell
.\.venv\Scripts\python.exe run.py run --example all --device cpu --out outputs/demo_01
```

使用 CUDA 环境执行同一组示例，并保留独立结果：

Run the same examples in the CUDA environment and keep separate results:

```powershell
.\.venv-gpu\Scripts\python.exe run.py run --example all --device cuda --out outputs/demo_gpu_01
```

`--device cuda` 明确要求 CUDA。`--device auto` 则根据可用设备选择执行路径；做 CPU／GPU 对比时建议明确指定设备，并检查实际记录。

`--device cuda` explicitly requests CUDA. `--device auto` chooses a path based on availability. Specify the device explicitly and inspect the recorded device when conducting a CPU/GPU comparison.

也可以单独运行一个市场；三个命令都通过根目录 `run.py` 调用同一训练流程。

Each market can also be run separately. All three commands use the same training workflow through the root `run.py`.

```powershell
.\.venv\Scripts\python.exe run.py run --example weak_signal --device cpu --out outputs/weak_01
```

```powershell
.\.venv\Scripts\python.exe run.py run --example trend --device cpu --out outputs/trend_01
```

```powershell
.\.venv\Scripts\python.exe run.py run --example regime_shift --device cpu --out outputs/regime_01
```

首次检查环境时，可以仅运行一个模型。模型名为 `mlp`、`gru`、`gnn`、`attention`、`transformer`。

For an initial environment check, run a single model. Available names are `mlp`, `gru`, `gnn`, `attention`, and `transformer`.

```powershell
.\.venv\Scripts\python.exe run.py run --example weak_signal --model mlp --device cpu --out outputs/weak_mlp_01
```

CUDA 执行时可以使用 `--execution eager` 建立常规执行参照，再用 `--execution optimized` 运行优化路径。比较时应保持市场、模型、配置、设备和软件环境相同，并使用独立输出目录。

For CUDA, `--execution eager` establishes an eager reference and `--execution optimized` selects the optimized path. Keep the market, model, configuration, device, and software environment unchanged and use separate output directories.

```powershell
.\.venv-gpu\Scripts\python.exe run.py run --example weak_signal --model gru --device cuda --execution eager --out outputs/gru_eager_01
```

```powershell
.\.venv-gpu\Scripts\python.exe run.py run --example weak_signal --model gru --device cuda --execution optimized --out outputs/gru_optimized_01
```

每次运行使用新的输出目录，例如将 `demo_01` 改为 `demo_02`。不要把不同配置或不同设备的结果写入同一目录；这有助于保留完整的配置和校验记录。

Use a new output directory for each run, for example `demo_02` after `demo_01`. Keep configurations and devices in separate directories to preserve coherent configuration and verification records.

## 5. 理解训练与模拟规则 / Understand training and simulation

| 项目 / Item | 默认规则 / Default rule |
|---|---|
| 市场规模 / Market size | 360 日、32 只合成股票。/ 360 sessions and 32 synthetic stocks. |
| 初始拟合区间 / Initial fit | 前 220 日；按标签结束日剔除边界不可用样本。/ First 220 sessions, excluding observations whose labels cross the permitted boundary. |
| 验证区间 / Validation | 第 220 至 279 日，用于选择训练轮数；边界采用 2 日 purge。/ Zero-based sessions 220–279, used to choose the epoch count, with a two-session purge at boundaries. |
| 测试区间 / Test | 从第 280 日起；预测日还须具备完整的五日标签。/ Starts at zero-based session 280; score dates must also have a complete five-session label. |
| 特征 / Features | 15 个仅使用已知量价信息的特征。/ Fifteen features based only on available price and volume information. |
| 输入历史 / Input history | MLP／GNN 使用当日特征；GRU／Attention／Transformer 使用最长 20 日特征序列。/ MLP/GNN use current features; GRU/Attention/Transformer use sequences of up to 20 sessions. |
| 标签 / Label | 次日开盘至其后第五个交易日开盘的收益，学习横截面排序。/ Next-open to five-sessions-later open return, used for cross-sectional ranking. |
| 预处理 / Preprocessing | 在当前拟合样本上估计 1%／99% 截尾点及标准化参数。/ Estimate 1st/99th percentile clipping bounds and standardization parameters on the current fit observations. |
| 网络 / Network | 默认隐藏维度 16、2 层，最多 6 轮，早停耐心值 2，种子 17。/ Hidden size 16, two layers, up to six epochs, patience two, seed 17. |
| 最终拟合 / Final refit | 使用验证选出的轮数，在测试前允许的训练与验证样本上重新拟合；不读取测试标签选择参数。/ Refit for the validation-selected epoch count on eligible pre-test training and validation observations; test labels do not choose parameters. |
| 初始本金 / Initial capital | 1,000,000 个模拟货币单位。/ 1,000,000 simulated currency units. |
| 选仓 / Selection | 最多持有排名前 5 只，占 32 只合成股票中的一部分。/ Hold up to the five highest-ranked stocks from the 32-stock synthetic universe. |
| 仓位与调仓 / Exposure and rebalance | 目标股票仓位 95%，每 20 个交易日调仓。/ 95% target equity allocation, rebalanced every 20 sessions. |
| 成交与费用 / Execution and costs | 下一交易日开盘成交；单边成交金额的 10 个基点；允许分数股。/ Next-open execution, 10 basis points on each side's traded value, fractional shares allowed. |

上述切分区间不等于所有日期都参加拟合。默认特征最长需要 120 日预热，边界还要考虑标签结束日和 2 日隔离。以从 0 开始的交易日序号表示，实际初始拟合为 120–211（92 日），验证为 220–271（52 日），最终拟合为 120–271（152 日），测试预测为 280–353（74 日）。这些是当前默认配置对应的可用日期，不是另一套训练窗口。

Not every date within a split participates in fitting. The longest default feature requires 120 sessions of warm-up, and boundaries must also allow for label completion and the two-session purge. With zero-based session indices, the usable initial fit dates are 120–211 (92 sessions), validation dates 220–271 (52), final refit dates 120–271 (152), and test score dates 280–353 (74). These are the eligible observations under the default configuration, not a separate window design.

“5 日标签”是模型学习的目标期限，“20 日调仓”是组合操作频率，二者不是同一个参数。本包保留这一组合便于观察预测期限与持仓周期的区别，不表示它已经在真实市场中得到优化。

The five-session label is the learning horizon; the 20-session rebalance interval is the portfolio action frequency. These are distinct parameters. Their combination illustrates the difference between forecast and holding horizons; it is not presented as an optimized real-market strategy.

调仓信号在生成后于下一开盘执行，不能使用同一时点尚未形成的成交价格。成本随买卖金额计算，并进入净值。简化账本不包含停牌、涨跌停、流动性容量、公司行动或完整的实盘执行机制。

A rebalance signal executes at the following open, without using an execution price unavailable when the signal was formed. Costs are calculated from traded value and deducted from portfolio value. The simplified ledger does not include suspensions, price limits, liquidity capacity, corporate actions, or a complete live-execution mechanism.

日净值记录为开盘价格下、当日交易后的账户价值；末端在约定的最后开盘清仓并计入卖出成本。日历只是模拟工作日序列，不代表任何交易所的实际交易日历。

Daily portfolio values are marked at open prices after that day's trades. Positions are liquidated at the designated final open with selling costs included. The calendar is a synthetic business-day sequence, not an exchange trading calendar.

最后一个预测日在第 353 日；账本仍会对之后的持仓计值，直至第 359 日开盘清仓。因此 `scores.csv` 和 `nav.csv` 的最后日期不同是本模拟的正常设计。开头还保留一个未建仓的初始本金记录。

The final score date is session 353; the ledger continues valuing positions until liquidation at session 359's open. Different final dates in `scores.csv` and `nav.csv` are therefore intentional. The ledger also begins with an initial-capital record before the first purchase.

## 6. 查看全部模型结果 / Inspect all model results

运行完成后，先查看本次目录中的文件。

After execution finishes, list the run directory.

```powershell
Get-ChildItem -LiteralPath .\outputs\demo_01
```

```powershell
Get-ChildItem -LiteralPath .\outputs\demo_01 -Recurse -Filter *.csv
```

打开全部已完成市场与模型的汇总：

Open the summary for all completed markets and models:

```powershell
Invoke-Item .\outputs\demo_01\summary.csv
```

GPU 运行的对应汇总为：

For the GPU run, open its corresponding summary:

```powershell
Invoke-Item .\outputs\demo_gpu_01\summary.csv
```

| 产物 / Artifact | 内容与阅读方式 / Contents and interpretation |
|---|---|
| `summary.csv` | 同一次运行中所有已完成市场／模型的指标，用 Excel 或 VS Code 比较。/ Metrics for all completed market/model combinations in the run, viewable in Excel or VS Code. |
| `metrics.json` | 单模型收益、回撤、成本和运行信息。/ Per-model returns, drawdown, costs, and run information. |
| `scores.csv` | 按日期、合成股票标识保存的预测排序分数。/ Prediction ranking scores by date and synthetic stock identifier. |
| `nav.csv` | 模拟净值路径与资金账户。/ Simulated portfolio-value path and account information. |
| `trades.csv` | 逐笔模拟买卖及费用。/ Simulated trades and their costs. |
| `checkpoint.pt` | 本次训练得到的神经网络参数。/ Neural parameters fitted in this run. |
| `training.json`、`scaler.json` | 训练／验证过程及拟合的预处理参数。/ Training/validation records and fitted preprocessing parameters. |
| `config.json`、`scenario.json` | 每个市场的实际配置与市场生成信息。/ Actual configuration and generation information for each market. |
| `manifest.json` | 结果文件的哈希，用于后续完整性核验。/ Output-file hashes for later integrity checks. |
| `completion.json` | 本次运行是否完成、完成模型数、设备与软件版本。/ Completion state, number of completed models, device, and software versions. |
| `runtime.json` | 实际软件版本与源码哈希，不记录个人路径。/ Actual software versions and source hashes, without personal paths. |

单模型文件位于“市场／模型”子目录，例如 `weak_signal/gru/metrics.json`。可以直接查看该模型的指标：

Per-model files live in market/model subdirectories, for example `weak_signal/gru/metrics.json`. Read that model's metrics directly:

```powershell
Get-Content .\outputs\demo_01\weak_signal\gru\metrics.json
```

全部三个市场、五类模型均完成时，应有 15 个市场／模型组合。仅运行一个市场或通过 `--model` 限定模型时，汇总行数相应减少。一次 CPU 运行的汇总不包含另一目录的 GPU 运行；比较两者时分别打开两个运行目录中的汇总文件。

A complete run of three markets and five model families contains 15 market/model combinations. A single-market or `--model` run contains fewer rows. A CPU run's summary does not include a GPU run stored elsewhere; open the summaries from both directories to compare them.

CSV 为普通文本表格，可用 Excel 的“数据 → 从文本／CSV”导入，选择 UTF-8 和逗号分隔。JSON 可在 VS Code、记事本中查看，也可用 `Get-Content` 读取。不要把查看后的手工修改保存回原结果文件，否则原有哈希不再匹配。

CSV files are text tables. In Excel, use “Data → From Text/CSV,” select UTF-8, and use a comma delimiter. Open JSON in VS Code or Notepad, or read it with `Get-Content`. Avoid saving manual changes over original outputs, because their stored hashes will no longer match.

预测分数表示模型对股票相对顺序的判断，不能直接当作收益百分比。累计收益来自预测经过选仓、成交、费用和净值计算后的模拟结果。弱信号、趋势和状态切换市场的结果差异，首先反映各自的数据生成规则。

Prediction scores represent the model's relative ordering of stocks and are not directly forecast-return percentages. Cumulative return is the simulated result after selection, execution, costs, and valuation. Differences among the three markets primarily reflect their data-generating rules.

## 7. 核验本次输出 / Verify this run

```powershell
.\.venv\Scripts\python.exe run.py verify --run outputs/demo_01
```

如果核验 GPU 输出，应传入 GPU 运行目录。

To verify GPU outputs, supply the GPU run directory.

```powershell
.\.venv-gpu\Scripts\python.exe run.py verify --run outputs/demo_gpu_01
```

核验程序依据本次生成的清单检查文件。返回的文件数量是被校验文件数量，不是训练模型数量，也不是成功交易数量。哈希匹配说明文件没有相对清单发生字节变化，不意味着 CPU／GPU 必然逐位相同，更不代表真实投资表现已验证。

Verification checks files against the manifest generated for that run. A reported file count is the number of checked files, not the number of trained models or successful trades. Hash matches establish byte integrity relative to the manifest; they do not establish bitwise CPU/GPU equality or real investment performance.

## 8. 修改配置或直接使用 Python / Change settings or call Python directly

`configs` 中的 JSON 是参数文件，Python 模块负责解释参数并实施数据生成、训练和模拟。修改前先阅读实际键名与取值。配置修改会改变实验，不能再把新结果当成默认配置的复现结果。

JSON files under `configs` hold parameters; Python modules interpret them and implement generation, training, and simulation. Read the actual keys and values before editing. A configuration change creates a different experiment, rather than a reproduction of the defaults.

开发者可以直接使用已安装依赖的兼容 Python 来运行 `run.py`，无需特定编辑器。下面示例中的 `python` 必须指向该环境；可先确认解释器路径。

Developers can invoke `run.py` with any compatible Python environment containing the required dependencies; no particular editor is required. In the examples below, `python` must refer to that environment. Confirm its executable path first.

```powershell
python -c "import sys; print(sys.executable)"
```

```powershell
python run.py run --example trend --device cpu --out outputs/trend_python_01
```

### 源码阅读顺序 / Reading the source

先查看 `run.py`：它设置源码导入路径，然后调用 `src/efcuda/cli.py` 中的 `main`。`main` 解析命令并交给 `workflow.run_examples` 执行。网络结构定义在 `models.py`，完整实验的组织则在 `workflow.py`；直接运行网络定义文件不会开始训练。

Start with `run.py`: it sets the source import path and calls `main` in `src/efcuda/cli.py`. `main` parses the command and dispatches a run to `workflow.run_examples`. Network architectures live in `models.py`; `workflow.py` organizes the experiment. Running the architecture file alone does not start training.

下面按一次运行的顺序列出关键位置。除 `run.py` 外，表中的 Python 模块均位于 `src/efcuda/`。

The following table follows a single run. All Python modules except `run.py` are under `src/efcuda/`.

| 阅读位置 / Where to read | 重点 / What to follow |
|---|---|
| `cli.py`：`main` | `--example` 如何选择配置，`--model` 如何缩小模型范围，`--device` 如何确定设备。/ How the arguments select configurations, restrict model families, and choose a device. |
| `workflow.py`：`run_examples` | 将整条流程连起来；先验证选轮数，再重新拟合，最后预测、回测和保存。/ Follow the full sequence: validation-based epoch selection, refitting, prediction, backtesting, and saving. |
| `data.py`：`generate`、`make_features`、`make_labels` | 行情如何生成、15 个特征如何计算，以及日期 `t` 的标签为何读取 `t+1` 与 `t+6` 的开盘价。/ Trace generation, the 15 features, and why a label at session `t` uses opens at `t+1` and `t+6`. |
| `data.py`：`split_masks`、`scale_panel` | 哪些日期进入拟合；截尾与标准化参数只从本轮允许的拟合样本估计，随后用于验证或测试。/ Identify eligible fit dates and how clipping/scaling parameters are estimated only from the permitted fit observations before use on validation or test data. |
| `models.py` 与 `engine.py`：`DevicePanel` | 网络接收当日特征还是历史序列；按批次拼接数据时，交易日、股票、特征各位于哪个维度。/ Check whether each network receives current features or a sequence, and track the session, stock, and feature dimensions when batches are assembled. |
| `engine.py`：`fit_network`、`ValidationRanks`、`predict` | 均方误差用于更新参数，验证 RankIC 用于早停选轮数；测试阶段固定参数，只生成预测。/ MSE drives parameter updates, validation RankIC selects the epoch count, and test inference uses fixed fitted parameters. |
| `native_graph.py`：`capture` | 固定形状批次如何复用 CUDA Graph，以及为何预热后恢复随机数状态。/ See how fixed-shape batches reuse a CUDA Graph and why random state is restored after warm-up. |
| `ledger.py`：`backtest` | 排序如何变成持仓，信号为何在下一开盘成交，以及买卖费用如何扣入账户。/ Trace rankings into positions, next-open execution, and the deduction of trading costs. |
| `workflow.py`：`save_csv`、`manifest`；`cli.py`：`verify` | 指标怎样写入 `summary.csv`，预测与净值怎样保存，以及文件哈希检查覆盖什么。/ Follow metrics into `summary.csv`, predictions and portfolio values into their files, and the scope of hash verification. |

例如，若想弄清 `summary.csv` 中的累计收益，可先查看 `ledger.backtest` 的账户计算，再回到 `workflow.run_examples` 查看指标如何写出。若想理解某个预测分数，则从 `scores.csv` 对应的日期和股票出发，查看 `engine.predict`、网络前向计算以及 `data.scale_panel` 的输入变换。注释中的张量维度、日期边界和资金流顺序可以与这些输出逐项对照。

For example, to understand cumulative return in `summary.csv`, read the account calculations in `ledger.backtest`, then follow their export in `workflow.run_examples`. For a prediction score, start with its date and stock in `scores.csv`, then inspect `engine.predict`, the network's forward pass, and the input transformation in `data.scale_panel`. The comments on tensor dimensions, date boundaries, and cash flows can be checked against those outputs.

## 9. 计算优化与使用边界 / Optimization and scope

- MLP、GRU、GNN 在 CUDA 上使用 FP32 CUDA Graph 训练；Attention 与 Transformer 使用优化后的常规执行路径。CPU 采用常规 FP32 执行。/ MLP, GRU, and GNN use FP32 CUDA Graph training on CUDA; Attention and Transformer use optimized eager execution. CPU execution is eager FP32.
- TF32 和混合精度不启用；训练与验证每批均为 2 日。优化集中在数据驻留、减少同步和复用执行图。/ TF32 and mixed precision are disabled; training and validation each use two-session batches. The optimizations focus on device-resident data, fewer synchronizations, and reusable execution graphs.
- 小数据、短训练中，GPU 初始化和 CUDA Graph 捕获可能占较大比例。应比较完整运行耗时，不能只依据 GPU 利用率或某一轮的耗时作判断。/ GPU initialization and graph capture can dominate small, short runs. Compare complete elapsed times rather than GPU utilization or an isolated epoch.
- 本包包含三个合成市场。它不接收真实行情，不实现历史缓存接口，也不提供实盘部署服务。/ The package contains three synthetic markets. It does not ingest real prices, expose a historical-cache interface, or provide live-trading deployment.
- 模拟结果可用于学习与代码检查，不能证明真实市场有效性。/ Simulation outputs support learning and code inspection; they do not establish real-market effectiveness.

### CUDA 警告的阅读 / Reading CUDA warnings

参考 PyTorch 构建在 CUDA Graph 运行时出现过 cuBLAS 初始上下文提示，以及包含 `AccumulateGrad` 的 CUDA stream mismatch 警告。程序继续运行，随后五组 CUDA 对照检查通过。运行日志保留这些提示，便于检查环境与执行状态。

The reference PyTorch build emitted a cuBLAS initial-context warning and an `AccumulateGrad` CUDA stream mismatch warning during CUDA Graph execution. Execution continued and the five CUDA comparisons subsequently passed. The messages remain in the log for environment and execution checks.

如果遇到相同提示，应检查命令是否正常完成以及对照测试是否通过；警告本身不等于 Python 异常，但也不能据此推断所有数值与性能行为都正确。若出现 `Traceback`、测试失败或程序中断，应保留完整日志并停止使用该次未完成的结果。

If the same messages appear, check that the command completes and that comparison tests pass. A warning is not itself a Python exception, but it does not establish that every numerical or performance behavior is correct. If a `Traceback`, failed test, or interruption occurs, retain the complete log and do not use the incomplete run as a finished result.
