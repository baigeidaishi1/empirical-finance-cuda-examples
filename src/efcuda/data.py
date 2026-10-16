"""生成合成行情，并按信号日构造特征、标签和时间切分。

所有价格、成交量和日期均由本模块生成，不连接行情服务。
Generate synthetic observations, close-time features, forward labels and time splits.
Prices, volumes and dates are generated locally; no market-data service is used.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.stats import rankdata

# 名称顺序必须与 make_features 中的列顺序一致；模型按列读取，不按名称查找。
# Keep this order aligned with make_features: models consume columns by position.
FEATURES = ['momentum5', 'momentum10', 'reversal3', 'volatility10', 'momentum20',
            'momentum60', 'volatility60', 'downside20', 'volume_ratio20', 'turnover20',
            'illiquidity20', 'momentum120', 'drawdown120', 'volatility120', 'volume_ratio120']
SCENARIOS = ('weak_signal', 'trend', 'regime_shift')


@dataclass
class Panel:
    """一份市场面板；T 为交易日数，N 为股票数，F 为因子数。

    raw、x 的形状为 (T, N, F)，分别保存原始因子与模型输入；x 在生成时
    仅将 NaN 置零，调用 scale_panel 后才是标准化结果。target_raw、target
    和 present 的形状均为 (T, N)：它们分别是未来收益率、截面排名标签和
    当日因子是否齐全的标记。present 不表示该日已经有可用的未来标签。

    opens、closes 为 (T, N) 的合成每股价格；volume 为同形状的成交股数。
    价格与资金账本使用一致的名义货币单位。history 是时序模型回看长度，
    features、codes、dates 分别对应 F、N、T 三个轴。

    A panel with T sessions, N stocks and F factors. raw and x have shape (T, N, F).
    x is initially zero-filled and becomes standardized only after scale_panel.
    target_raw, target and present have shape (T, N): forward fractional returns,
    cross-sectional rank targets and feature availability, respectively. Feature
    availability does not imply that a forward label is available.
    opens/closes are synthetic prices per share in the ledger's currency unit;
    volume is shares traded, all shaped (T, N). history is the sequence lookback;
    features, codes and dates label the F, N and T axes.
    """
    raw: np.ndarray
    x: np.ndarray
    target: np.ndarray
    target_raw: np.ndarray
    present: np.ndarray
    history: int
    features: list
    opens: np.ndarray
    closes: np.ndarray
    volume: np.ndarray
    codes: list
    dates: list


def make_features(closes, volume, shares):
    """用截至当日收盘的价格、成交量及固定股本构造 15 个因子。

    收益率使用小数口径，0.01 代表 1%；波动率为日收益的总体标准差，
    不做年化。前 120 行保留 NaN，等到最长回看窗口完整后再生成因子。
    Build 15 factors from observations available at each close and fixed share counts.
    Returns are fractions (0.01 means 1%). Volatility is the population standard
    deviation of daily returns, without annualization. The first 120 rows remain
    NaN until the longest lookback is available.
    """
    closes, volume, shares = map(np.asarray, (closes, volume, shares))
    if closes.shape != volume.shape or closes.ndim != 2 or shares.shape != (closes.shape[1],):
        raise ValueError('Expected day-by-stock prices and volumes, and one share count per stock')
    if not all(np.isfinite(x).all() and (x > 0).all() for x in (closes, volume, shares)):
        raise ValueError('Prices, volumes and share counts must be finite and positive')
    daily = np.full_like(closes, np.nan, dtype=float)
    daily[1:] = closes[1:] / closes[:-1] - 1
    raw = np.full((*closes.shape, len(FEATURES)), np.nan, dtype=np.float32)
    # t 是收盘信号日。n 日收益窗口含 t 日，动量则比较 t 与 t-n 的收盘价。
    # t is the close-time signal day. Return windows include t; momentum compares t with t-n.
    for t in range(120, len(closes)):
        returns = lambda n: daily[t-n+1:t+1]
        volumes = lambda n: volume[t-n+1:t+1]
        # 非流动性代理 = 20 日平均的 |日收益| / 成交额，再缩放并取 log1p。
        # Illiquidity proxy: mean |daily return| / traded value, then rescale and apply log1p.
        impact = np.mean(np.abs(returns(20)) / (volumes(20) * closes[t-19:t+1]), axis=0)
        raw[t] = np.stack([
            # reversal3 是三日动量的相反数；downside20 是负收益部分的均方根。
            # reversal3 negates three-day momentum; downside20 is the RMS of negative returns.
            closes[t] / closes[t-5] - 1, closes[t] / closes[t-10] - 1,
            1 - closes[t] / closes[t-3], returns(10).std(axis=0),
            closes[t] / closes[t-20] - 1, closes[t] / closes[t-60] - 1,
            returns(60).std(axis=0), np.sqrt(np.mean(np.minimum(returns(20), 0)**2, axis=0)),
            # 量比减 1 后以零为参照；turnover20 为日成交股数 / 股本的 20 日均值。
            # Volume ratios are centered at zero; turnover20 averages volume / shares over 20 days.
            volumes(5).mean(axis=0) / volumes(20).mean(axis=0) - 1,
            (volumes(20) / shares).mean(axis=0), np.log1p(impact * 1e10),
            # drawdown120 是当前价距窗口最高价的跌幅，不是该窗口的最大回撤。
            # drawdown120 measures the current price below the window high, not maximum drawdown.
            closes[t] / closes[t-120] - 1, closes[t] / closes[t-119:t+1].max(axis=0) - 1,
            returns(120).std(axis=0), volumes(20).mean(axis=0) / volumes(120).mean(axis=0) - 1,
        ], axis=-1)
    return raw


def make_labels(opens, horizon):
    """把 t 日信号对应到下一开盘买入、horizon 日后开盘退出的收益。

    公式为 open[t + 1 + horizon] / open[t + 1] - 1。末尾缺少退出价格的
    行保留 NaN。训练标签按同一天所有股票的收益升序排名，同值取平均名次，
    然后除以股票数并减 0.5；这是相对强弱目标，不是预测收益率百分比。

    Label signal day t with open[t + 1 + horizon] / open[t + 1] - 1. Trailing
    rows without an exit price remain NaN. Training targets rank returns within
    each day, average tied ranks, divide by stock count and subtract 0.5. They
    describe relative performance rather than forecast return percentages.
    """
    raw = np.full(opens.shape, np.nan, dtype=np.float32)
    count = len(opens) - horizon - 1
    raw[:count] = opens[horizon+1:] / opens[1:len(opens)-horizon] - 1
    target = np.full_like(raw, np.nan)
    target[:count] = rankdata(raw[:count], axis=1, method='average') / opens.shape[1] - .5
    return raw, target


def generate(config):
    """按配置中的种子生成弱信号、趋势或状态切换市场。

    种子固定随机抽样；scenario 决定漂移和噪声系数，所以同一种子也会生成
    不同的价格路径。各场景仅用于观察模型行为，不代表真实市场收益分布。
    Generate a weak-signal, trending or regime-switching market from a fixed seed.
    The seed fixes the draws; scenario-specific drift and noise coefficients still
    produce different price paths. These scenarios illustrate model behavior and
    are not estimates of real market return distributions.
    """
    scenario = config['scenario']
    if scenario not in SCENARIOS:
        raise ValueError('Only the three registered synthetic scenarios are supported')
    n, count = int(config['stocks']), int(config['sessions'])
    rng = np.random.default_rng(config['seed'])
    stock_trend = np.linspace(-.0015, .0015, n)[rng.permutation(n)]
    # 市场扰动由同一天所有股票共享；个股扰动和隔夜跳空分别独立抽样。
    # Stocks share each day's market shock; stock shocks and overnight gaps are drawn separately.
    market = rng.normal(size=(count, 1))
    noise = rng.normal(size=(count, n))
    gaps = rng.normal(size=(count, n))
    if scenario == 'weak_signal':
        drift, market_std, noise_std, gap_std = stock_trend * .1, .004, .007, .002
    elif scenario == 'trend':
        drift, market_std, noise_std, gap_std = stock_trend, .0015, .002, .0005
    else:
        # 在序列的 65% 处同时反转个股漂移并提高噪声强度，位置不随训练结果调整。
        # Reverse stock drift and increase noise at 65% of the sequence, independent of fitted results.
        change = np.arange(count)[:, None] >= int(count * .65)
        drift = np.where(change, -stock_trend, stock_trend)
        market_std = np.where(change, .0045, .0015)
        noise_std = np.where(change, .006, .002)
        gap_std = np.where(change, .0015, .0005)
    # 收盘价由累计对数收益生成；开盘价由前收盘价与单独的隔夜扰动生成。
    # Closes compound log returns; opens combine the previous close with a separate overnight shock.
    log_returns = drift + market_std * market + noise_std * noise
    starts = rng.uniform(20, 100, n)
    closes = starts * np.exp(np.cumsum(log_returns, axis=0))
    previous = np.vstack([starts, closes[:-1]])
    opens = previous * np.exp(gap_std * gaps)
    shares = rng.uniform(5e7, 2e8, n)
    volume = shares * np.exp(rng.normal(-4.5, .3, (count, n)))
    raw = make_features(closes, volume, shares)
    target_raw, target = make_labels(opens, config['label_horizon'])
    present = np.isfinite(raw).all(axis=2)
    # 日期只排除周末，不含交易所节假日；它们是合成序列的显示标签。
    # Dates skip weekends only, not exchange holidays; they label the synthetic sequence.
    dates = np.busday_offset(np.datetime64('2020-01-01'), np.arange(count)).astype(str).tolist()
    return Panel(raw=raw, x=np.nan_to_num(raw), target=target, target_raw=target_raw, present=present,
                 history=config['network']['history'], features=FEATURES.copy(), opens=opens,
                 closes=closes, volume=volume, codes=[f'SIM{i+1:03d}' for i in range(n)], dates=dates)


def split_masks(panel, config):
    """用信号日和标签退出日共同限制训练、验证及最终拟合样本。

    train_end、validation_end 都是从零开始的日序号，不属于前一个区间。
    end 为标签实际退出日；严格小于“边界减 purge”才可进入对应拟合区间，
    从而把标签末端也留在边界之前。final 合并边界之前可用的训练和验证样本，
    供选好设置后重拟合；测试日从 validation_end 开始，直到标签仍可计算为止。

    Split by both signal date and label exit date. train_end and validation_end
    are zero-based, exclusive boundaries. A label's exit index must be strictly
    below boundary minus purge. final combines eligible pre-test observations
    for refitting after selection. Test signal days start at validation_end and
    stop before forward labels become unavailable.
    """
    count = len(panel.dates)
    day = np.arange(count)[:, None]
    end = day + config['label_horizon'] + 1
    train_end, valid_end, purge = config['train_end'], config['validation_end'], config['purge_sessions']
    eligible = panel.present & np.isfinite(panel.target)
    train = eligible & (day < train_end) & (end < train_end - purge)
    valid = eligible & (day >= train_end) & (day < valid_end) & (end < valid_end - purge)
    final = eligible & (day < valid_end) & (end < valid_end - purge)
    return dict(train_mask=train, valid_mask=valid, final_mask=final,
                train_days=np.flatnonzero(train.any(axis=1)), valid_days=np.flatnonzero(valid.any(axis=1)),
                final_days=np.flatnonzero(final.any(axis=1)),
                test_days=np.arange(valid_end, count - config['label_horizon'] - 1))


def scale_panel(panel, mask):
    """只在 mask 指定的样本上估计截尾阈值、均值和标准差，再应用到全序列。

    各因子合并训练日期与股票估计 1%/99% 分位数；截尾后计算均值和总体
    标准差。验证及测试值沿用这些统计量，标准化结果再限于 [-5, 5]。
    未就绪的因子行置零只是为张量计算准备，present 仍负责标记有效样本。

    Fit per-factor 1%/99% clipping bounds, means and population standard deviations
    on mask-selected observations, pooling dates and stocks. Apply the same
    statistics to validation and test data, then cap standardized values at [-5, 5].
    Unavailable rows become zero for tensor operations; present retains validity.
    """
    values = panel.raw[mask & panel.present]
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Scaler requires finite training observations')
    low, high = np.quantile(values, [.01, .99], axis=0)
    clipped = np.clip(values, low, high)
    mean, std = clipped.mean(axis=0), clipped.std(axis=0)
    std = np.where(std < 1e-8, 1., std)
    x = np.clip((np.clip(panel.raw, low, high) - mean) / std, -5, 5).astype(np.float32)
    x[~panel.present] = 0
    scaler = {key: value.tolist() for key, value in [('low', low), ('high', high), ('mean', mean), ('std', std)]}
    return replace(panel, x=x), scaler
