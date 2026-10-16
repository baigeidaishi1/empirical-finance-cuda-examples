"""串起一个市场从数据生成到结果落盘的流程。 / Coordinate a market run from data to files.

每个模型先用验证集选训练轮数，再以选定轮数重新拟合，最后生成测试期评分和账本。
Each model selects its epoch count on validation data, refits for that count,
then produces test-period scores and a portfolio ledger.
"""
import csv
import gc
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
import scipy
from .data import SCENARIOS, generate, split_masks, scale_panel
from .engine import fit_network, predict, ValidationRanks
from .models import MODEL_NAMES
from .ledger import backtest


def save_json(path, value):
    """写入 UTF-8 JSON；拒绝 NaN，避免保存其他读取器无法处理的数值。
    Write UTF-8 JSON and reject NaN values that other readers may not support.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows, columns=None):
    """用 UTF-8 BOM 方便表格软件识别中文；空表需要显式给出列名。
    Use a UTF-8 BOM for spreadsheet readers; empty tables need explicit columns.
    """
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    """分块计算文件指纹，避免一次读入整个权重文件。
    Hash in chunks rather than loading an entire checkpoint into memory.
    """
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def manifest(directory):
    """记录输出文件指纹，清单自身不参与递归哈希。
    Record output hashes, excluding the manifest itself.
    """
    files = {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'manifest.json'}
    save_json(directory / 'manifest.json', {'algorithm': 'sha256', 'files': files})


def validate_config(config):
    """检查本示例支持的参数范围和时间顺序。
    Check supported parameter ranges and chronological boundaries.

    这不是任意 JSON 的完整模式校验；后续模块还会检查输入形状和有效样本。
    This is not a general JSON schema validator; later modules also check shapes
    and usable observations.
    """
    if config['scenario'] not in SCENARIOS or not 25 <= config['stocks'] <= 300:
        raise ValueError('Registered scenario and 25–300 synthetic stocks required')
    if not 120 < config['train_end'] < config['validation_end'] < config['sessions'] - config['label_horizon'] - 2:
        raise ValueError('Invalid chronological boundaries')
    if config['label_horizon'] != 5 or config['purge_sessions'] < 2:
        raise ValueError('This package uses five-session labels and at least two purge sessions')
    if not config['models'] or len(set(config['models'])) != len(config['models']) or any(m not in MODEL_NAMES for m in config['models']):
        raise ValueError('Models must be a nonempty distinct subset of the five supported networks')
    if config['training']['batch_days'] != 2:
        raise ValueError('The accepted training and validation minibatch is two sessions')


def run_examples(root, examples, destination, device, model_name=None, execution='optimized'):
    """执行所选市场与模型，并返回终端摘要。
    Run the selected markets and models, then return a terminal summary.

    root 用于定位包内配置；destination 用于保存本次运行的独立产物。
    root locates bundled configurations; destination holds this run's own artifacts.
    """
    # 新目录让评分、权重和配置属于同一次运行，防止结果相互混合。
    # A fresh directory keeps scores, weights, and configurations from one run together.
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new output directory / 请使用新的输出目录')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or install a compatible CUDA PyTorch')
    configs = []
    for example in examples:
        config = json.loads((root / 'configs' / f'{example}.json').read_text(encoding='utf-8'))
        validate_config(config)
        if config['scenario'] != example:
            raise ValueError('Scenario and configuration filename differ')
        configs.append(config)
    torch.set_num_threads(4)
    destination.mkdir(parents=True)
    rows = []
    for config in configs:
        scenario = config['scenario']
        folder = destination / scenario
        folder.mkdir()
        panel = generate(config)
        split = split_masks(panel, config)
        # 验证选轮数时，缩放参数仅从训练部分估计。最终重拟合另用 final_mask
        # 估计缩放参数；两套缩放参数的估计均不使用测试样本。
        # Epoch selection fits preprocessing on training observations only. Refit
        # uses a separate scaler fitted on final_mask; neither is fitted on test observations.
        tuned_panel, _ = scale_panel(panel, split['train_mask'])
        final_panel, final_scaler = scale_panel(panel, split['final_mask'])
        save_json(folder / 'config.json', config)
        save_json(folder / 'scenario.json', dict(evidence_type='SYNTHETIC ONLY / 仅合成示例',
            features=panel.features, codes=panel.codes, seed=config['seed'],
            train_signal_days=split['train_days'].tolist(), validation_signal_days=split['valid_days'].tolist(),
            refit_signal_days=split['final_days'].tolist(), test_signal_days=split['test_days'].tolist(),
            dates=panel.dates, calendar='Artificial weekday calendar / 人工工作日日历'))
        settings, spec = config['training'], config['network']
        for kind in ([model_name] if model_name else config['models']):
            target = folder / kind
            target.mkdir()
            if device == 'cuda':
                # CUDA 工作异步提交；同步后再计时，避免计入上一项尚未完成的工作。
                # CUDA launches are asynchronous; finish earlier work before timing.
                torch.cuda.synchronize()
            started = time.perf_counter()
            print(f'TRAIN {scenario} {kind} {device}', flush=True)
            tuned, selected_epoch, tuning_ic, tuning_curve = fit_network(
                tuned_panel, kind, spec, split['train_days'], split['train_mask'], split['valid_days'],
                split['valid_mask'], settings['max_epochs'], settings['patience'], settings['seed'],
                settings['batch_days'], device=device, execution=execution)
            # 验证阶段保留轮数和指标；下一次 fit 从相同种子重新初始化模型。
            # Keep the selected epoch count and metrics. The next fit initializes
            # a new model from the same seed rather than continuing these weights.
            del tuned
            gc.collect()
            model, _, _, refit_curve = fit_network(
                final_panel, kind, spec, split['final_days'], split['final_mask'], np.array([], dtype=int),
                np.zeros_like(panel.present), selected_epoch, None, settings['seed'], settings['batch_days'],
                device=device, execution=execution)
            if device == 'cuda':
                torch.cuda.synchronize()
            # 耗时覆盖选轮数与重拟合；后面的测试预测、回测及写文件不计入。
            # Time covers epoch selection and refitting, excluding test inference,
            # portfolio accounting, and file writes below.
            fit_seconds = time.perf_counter() - started
            scores = predict(model, final_panel, split['test_days'], batch_days=2)
            # RankIC 衡量同一天的股票评分与标签排序关系；组合收益还取决于
            # ledger 中的调仓、权重和成本规则。评分本身不是收益率百分数。
            # RankIC compares cross-sectional score and label rankings. Portfolio
            # returns also depend on ledger rules and costs; scores are not return percentages.
            test_ic = ValidationRanks(panel.target[split['test_days']]).score(scores)
            metrics, nav, trades = backtest(panel, split['test_days'], scores, config['portfolio'])
            metrics.update(model=kind, scenario=scenario, device=device, execution=model.execution_policy,
                           selected_epochs=selected_epoch, validation_rankic=tuning_ic,
                           test_rankic=test_ic if np.isfinite(test_ic) else None, fit_seconds=fit_seconds)
            save_json(target / 'metrics.json', metrics)
            save_json(target / 'training.json', dict(tuning_curve=tuning_curve, refit_curve=refit_curve,
                selected_epoch=selected_epoch, selection_uses_test=False, execution=model.execution_policy))
            save_json(target / 'scaler.json', final_scaler)
            save_csv(target / 'nav.csv', nav)
            save_csv(target / 'trades.csv', trades, ['session', 'date', 'symbol', 'side', 'quantity', 'price', 'gross_value', 'fee'])
            score_rows = [dict(session=int(day), date=panel.dates[day], symbol=code,
                               score=float(scores[i, j]), target_rank=float(panel.target[day, j]))
                          for i, day in enumerate(split['test_days']) for j, code in enumerate(panel.codes)]
            save_csv(target / 'scores.csv', score_rows)
            # 保存最终重拟合权重、因子顺序和缩放器，才能按相同输入口径再推理。
            # Save refit weights, feature order, and scaler for consistent re-inference.
            torch.save(dict(model=kind, spec=spec, feature_names=panel.features, scaler=final_scaler,
                            state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
                            synthetic_config=config, selected_epoch=selected_epoch), target / 'checkpoint.pt')
            rows.append(metrics)
            # 每完成一个模型就刷新汇总；中途存在 summary 不等于整次运行完成。
            # Refresh the summary after each model; its presence alone does not imply completion.
            save_csv(destination / 'summary.csv', rows)
            print(f'DONE {scenario} {kind}: {fit_seconds:.2f}s, synthetic return {metrics["total_return"]:.2%}', flush=True)
            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    # 环境版本和源码指纹用于追溯本次运行；completion 仅在全部所选模型结束后写出。
    # Record runtime versions and source hashes. Write completion only after all selected fits finish.
    save_json(destination / 'runtime.json', dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, scipy=scipy.__version__, cuda_runtime=torch.version.cuda, device=device,
        gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
        source_hashes={p.relative_to(root).as_posix(): digest(p)
                       for p in [root / 'run.py', *sorted((root / 'src').rglob('*.py'))]}))
    save_json(destination / 'completion.json', dict(status='complete', models_completed=len(rows),
        scenarios=examples, device=device, torch=torch.__version__, numpy=np.__version__,
        evidence_type='SYNTHETIC ONLY / 仅合成示例', real_data_used=False))
    manifest(destination)
    return {'status': 'complete', 'models_completed': len(rows), 'summary': 'summary.csv', 'device': device}
