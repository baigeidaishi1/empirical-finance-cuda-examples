"""在 CUDA 上对照常规与优化执行。 / Compare eager and optimized execution on CUDA.

用同一份合成数据和种子检查权重、损失与预测。此检查不测量加速比，
也不要求 CPU 和 GPU 的训练结果相同。
Compare weights, losses, and predictions on the same synthetic data and seed.
This check measures neither speedup nor CPU/GPU training equivalence.
"""
import json
from pathlib import Path
import sys
import gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from efcuda.data import generate, split_masks, scale_panel
from efcuda.engine import fit_network, predict
import numpy as np
import torch


def main():
    """显式要求 CUDA 可用，避免把设备检查跳过后当作通过。
    Require CUDA explicitly rather than treating an unavailable-device skip as a pass.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('This check requires CUDA; use run.py test for CPU tests')
    torch.set_num_threads(4)
    config = json.loads((ROOT / 'configs/weak_signal.json').read_text())
    panel = generate(config)
    split = split_masks(panel, config)
    panel, _ = scale_panel(panel, split['train_mask'])
    rows = []
    for kind in config['models']:
        results = []
        for execution in ('eager', 'optimized'):
            # 七个训练日按两日组批，最后一批只有一天，用于覆盖尾批处理。
            # Seven training days in batches of two also exercise a one-day tail batch.
            model, epoch, ic, curve = fit_network(panel, kind, config['network'], split['train_days'][-7:],
                split['train_mask'], split['valid_days'][:4], split['valid_mask'], 2, None, 17, 2,
                device='cuda', execution=execution)
            scores = predict(model, panel, split['test_days'][:4])
            results.append(dict(epoch=epoch, ic=ic, scores=scores, curve=curve, policy=model.execution_policy,
                weights={name: value.detach().cpu().clone() for name, value in model.state_dict().items()}))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        # 对比完整参数与分数，而不只检查程序是否结束或损失是否下降。
        # Compare parameters and scores as well as successful execution and loss values.
        a, b = results
        score_error = float(np.abs(a['scores'] - b['scores']).max())
        weight_error = max(float((a['weights'][name] - b['weights'][name]).abs().max()) for name in a['weights'])
        curve_error = max(abs(x['train_mse'] - y['train_mse']) for x, y in zip(a['curve'], b['curve']))
        passed = a['epoch'] == b['epoch'] and max(score_error, weight_error, curve_error, abs(a['ic']-b['ic'])) <= 1e-6
        rows.append(dict(model=kind, execution=b['policy'], score_error=score_error,
                         weight_error=weight_error, loss_error=curve_error, passed=passed))
    print(json.dumps({'cuda_checks': rows}, indent=2))
    if not all(row['passed'] for row in rows):
        raise ValueError('Paired CUDA execution check failed')


if __name__ == '__main__':
    main()
