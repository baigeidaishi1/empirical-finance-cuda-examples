"""将合成市场预测转为下一开盘成交的持仓、成本和净值记录。

支持零碎股，按统一费率计收双边交易成本，不模拟真实证券的整手或限价规则。
Convert synthetic-market scores into next-open positions, costs and NAV records.
The ledger permits fractional shares and charges a uniform rate on buys and sells;
it does not model exchange lot sizes or price limits.
"""
import numpy as np


def backtest(panel, days, scores, settings):
    """按日期对齐的预测得分回测；scores 的形状为 (信号日数, 股票数)。

    第一个信号日从全现金开始记账，收盘信号在下一交易日开盘执行。
    调仓间隔按面板日序号计算，不按调用次数计算。净值均按当日开盘价格
    估值，最后一个面板交易日开盘全部平仓并计费。

    Backtest scores shaped (number of signal days, number of stocks). The initial
    record is all cash on the first signal day; close-time scores trade at the
    next open. Rebalancing uses panel-session offsets, not function-call counts.
    NAV is marked at each open, with full liquidation and costs at the final open.
    """
    days = np.asarray(days, dtype=int)
    scores = np.asarray(scores)
    if len(days) == 0 or scores.shape != (len(days), len(panel.codes)) or not np.isfinite(scores).all():
        raise ValueError('Finite scores are required for each signal day and stock')
    if np.any(np.diff(days) <= 0) or days[0] < 0 or days[-1] + 1 >= len(panel.opens):
        raise ValueError('Signals must be chronological with a following open')
    capital, fraction = float(settings['initial_capital']), float(settings['equity_fraction'])
    # cost_bps 为每一侧成交额的基点成本；10 bps = 0.001，即 0.1%。
    # cost_bps applies to each side's traded value; 10 bps = 0.001, or 0.1%.
    fee_rate = float(settings['cost_bps']) / 10000
    top_n, rebalance = int(settings['top_n']), int(settings['rebalance_sessions'])
    if not (capital > 0 and 0 < fraction <= .95 and 0 <= fee_rate <= .005 and 1 <= top_n <= len(panel.codes) and rebalance >= 1):
        raise ValueError('Invalid portfolio configuration')
    cash, holdings = capital, np.zeros(len(panel.codes))
    fees = 0.
    trades, nav = [], [dict(session=int(days[0]), date=panel.dates[days[0]], nav=capital, cash=capital, equity=0.)]
    lookup = {int(day): i for i, day in enumerate(days)}
    final_day = len(panel.opens) - 1
    for t in range(int(days[0]) + 1, final_day + 1):
        prices = panel.opens[t]
        value = cash + float(holdings @ prices)
        target = None
        # 最后一天优先执行平仓；即使当天符合调仓间隔，也不会重新买入。
        # Final liquidation takes precedence over any scheduled rebalance.
        if t == final_day:
            target = np.zeros_like(holdings)
        elif t-1 in lookup and (t-1-int(days[0])) % rebalance == 0:
            eligible = np.flatnonzero(panel.present[t-1])
            # 得分从高到低排序，分数相同按股票列号排序。默认前 5 只由配置指定。
            # Rank scores descending, breaking ties by stock-column index. The default top five is configurable.
            ranks = np.lexsort((eligible, -scores[lookup[t-1], eligible]))
            selected = eligible[ranks[:top_n]]
            target = np.zeros_like(holdings)
            # 用交易前开盘净值分配等额槽位：每只占 equity_fraction / top_n。
            # 默认 95% 股票仓位、前 5 只，即每只目标 19%；交易费另从现金扣除。
            # 股票不足时空余槽位留现金；即使入选名单不变，也会调回目标股数。
            # Allocate equal slots from pre-trade opening NAV: equity_fraction / top_n each.
            # Defaults of 95% equity and five names imply 19% per name; costs reduce cash separately.
            # Unfilled slots stay in cash. Retained names are also resized to their new target quantities.
            target[selected] = value * fraction / top_n / prices[selected]
        if target is not None:
            change = target - holdings
            gross = change * prices
            # 买卖价差、冲击和手续费在此合并为同一比例代理，不另调整成交价。
            # A single proportional cost proxy covers trading frictions; fill prices are not adjusted separately.
            cost = np.abs(gross) * fee_rate
            cash -= float(gross.sum() + cost.sum())
            if cash < -1e-7:
                raise ValueError('Insufficient cash after transaction costs')
            holdings = target
            fees += float(cost.sum())
            for j in np.flatnonzero(np.abs(change) > 1e-10):
                trades.append(dict(session=t, date=panel.dates[t], symbol=panel.codes[j],
                                   side='buy' if change[j] > 0 else 'sell', quantity=float(abs(change[j])),
                                   price=float(prices[j]), gross_value=float(abs(gross[j])), fee=float(cost[j])))
        equity = float(holdings @ prices)
        nav.append(dict(session=t, date=panel.dates[t], nav=cash + equity, cash=cash, equity=equity))
    wealth = np.array([row['nav'] for row in nav])
    # 统计基于完整资金净值，已经计入交易费与末次平仓；初始全现金记录也保留。
    # Metrics use total portfolio NAV after costs and liquidation, retaining the initial cash record.
    returns = wealth[1:] / wealth[:-1] - 1
    volatility = float(returns.std(ddof=0) * np.sqrt(252))
    # 以 252 个间隔年化，无风险利率取零；这里的 Sharpe 不代表超越市场基准。
    # Annualize with 252 intervals and zero risk-free rate; this Sharpe is not excess return over a market benchmark.
    sharpe = float(returns.mean() * 252 / volatility) if volatility > 0 else 0.
    metrics = dict(initial_capital=capital, final_nav=float(wealth[-1]), total_return=float(wealth[-1] / capital - 1),
                   annualized_return=float((wealth[-1] / capital)**(252 / len(returns)) - 1),
                   annualized_volatility=volatility, sharpe_zero_rf=sharpe,
                   max_drawdown=float((wealth / np.maximum.accumulate(wealth) - 1).min()),
                   fees=fees, trades=len(trades), return_intervals=len(returns),
                   evidence_type='SYNTHETIC ONLY / 仅合成示例', final_liquidation=True)
    return metrics, nav, trades
