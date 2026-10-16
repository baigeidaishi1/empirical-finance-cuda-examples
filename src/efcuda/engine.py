"""网络拟合、验证和推理共用的 FP32 执行流程。
Shared FP32 execution for fitting, validation and inference.

日期切分与特征缩放由调用方完成；这里接收面板和样本掩码，负责参数更新。
Splitting and feature scaling belong to the caller; this module updates parameters
from the supplied panel and sample masks.
"""

from __future__ import annotations

import gc
import os
import random

# 确定性 CUDA 矩阵乘法需要此工作区设置，须在首次相关运算前生效。
# Deterministic CUDA GEMM needs this workspace setting before its first use.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.stats import rankdata

from .models import FullRankNet, MODEL_NAMES


def configure_runtime(kind: str, device: str | torch.device) -> torch.device:
    """检查所选设备，并统一设置确定性算法与 FP32 计算精度。
    Validate the selected device and configure deterministic FP32 computation.

    显式请求 CUDA 时，设备不可用会报错，避免把 CPU 耗时记录成 GPU 耗时。
    An unavailable CUDA request fails, keeping CPU timings out of GPU run records.
    这些设置约束当前执行路径，不能保证不同硬件上的独立训练逐位一致。
    These settings constrain execution; independent fits on different hardware
    can still differ numerically.
    """
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported")
    if selected.type == "cpu":
        selected = torch.device("cpu")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; install a matching PyTorch build or select cpu")
        selected = torch.device("cuda", torch.cuda.current_device() if selected.index is None else selected.index)
        if selected.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device does not exist")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # GRU 的 eager 和图捕获路径都使用原生实现，便于比较执行优化前后的结果。
    # Both eager and graphed GRU runs use the native implementation for comparison.
    torch.backends.cudnn.enabled = kind != "gru"
    # float32 张量仍可能采用 TF32 矩阵运算；这里显式关闭该模式。
    # Float32 tensors can still use TF32 matrix arithmetic; disable that mode here.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return selected


def _days(values, count: int, name: str) -> np.ndarray:
    """验证面板的日期位置索引，保留调用方给出的顺序。
    Validate positional day indices without changing the caller's ordering.
    """
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and not np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer day index")
    array = array.astype(np.int64, copy=False)
    if len(array) and (array.min() < 0 or array.max() >= count):
        raise ValueError(f"{name} contains an out-of-range day")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} contains duplicate days")
    return array


class DevicePanel:
    """把基础特征和在场标记放到训练设备，按批次提取历史窗口。
    Keep features and presence flags on the device and gather windows per batch.

    基础特征形状为 [日期 D, 股票 N, 因子 F]。只缓存这份基础面板，避免
    预先复制成 [D, N, 历史长度 L, F]；每批序列在设备端临时组装。
    Features have shape [days D, stocks N, factors F]. Caching the base panel
    avoids storing all [D, N, history L, F] windows in advance.

    拟合期间应把源面板视为只读；下面的身份检查能发现数组替换，但不会
    检测数组原地修改，因此修改数据后应重新创建缓存。
    Treat the source panel as read-only during a fit. Identity checks detect
    array replacement, not in-place edits; rebuild the cache after data changes.
    """

    def __init__(self, panel, kind: str, device: str | torch.device):
        x, present, target = np.asarray(panel.x), np.asarray(panel.present), np.asarray(panel.target)
        if x.ndim != 3 or min(x.shape) < 1 or x.dtype != np.float32:
            raise ValueError("panel.x must be a nonempty float32 [day, stock, feature] array")
        if present.shape != x.shape[:2] or present.dtype != np.bool_ or target.shape != present.shape:
            raise ValueError("Panel target/presence dimensions must match the feature panel")
        if target.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("Targets must be float32 and feature values must all be finite")
        if len(panel.features) != x.shape[2] or int(panel.history) < 1:
            raise ValueError("Feature names or history length do not match the panel")
        self.source, self.kind, self.device = panel, kind, torch.device(device)
        self.x_identity, self.present_identity = panel.x, panel.present
        self.history = int(panel.history)
        self.x_device = torch.as_tensor(x, device=self.device)
        self.present_device = torch.as_tensor(present, device=self.device)
        self.offsets = torch.arange(self.history - 1, -1, -1, device=self.device)

    def batch(self, days):
        """返回当日特征、历史序列、历史有效掩码和当日在场标记。
        Return current features, history, history mask and current presence flags.

        B 个日期的序列形状为 [B, N, L, F]，历史掩码为 [B, N, L]。
        MLP/GNN 只用当日横截面，因此两个历史相关返回值均为 None。
        For B days, sequences have shape [B, N, L, F] and masks [B, N, L].
        MLP/GNN use the current cross-section and return None for both history fields.
        """
        if (self.source.x is not self.x_identity or self.source.present is not self.present_identity
                or self.source.history != self.history):
            raise ValueError("Panel replaced or history changed: rebuild the device cache")
        index = days if isinstance(days, torch.Tensor) else torch.as_tensor(days, device=self.device, dtype=torch.long)
        current, present = self.x_device[index], self.present_device[index]
        if self.kind in ("mlp", "gnn"):
            return current, None, None, present
        past = index[:, None] - self.offsets[None]
        # 负日期先夹到零以便安全索引，随后屏蔽；不会把第零日当作更早的观测。
        # Clamp negative indices for a safe gather, then mask them out as unavailable.
        safe = past.clamp(min=0)
        sequence = self.x_device[safe].permute(0, 2, 1, 3).contiguous()
        mask = self.present_device[safe].permute(0, 2, 1).contiguous()
        mask &= (past >= 0)[:, None, :]
        sequence.masked_fill_(~mask[..., None], 0)
        return current, sequence, mask, present


class ValidationRanks:
    """计算每日横截面 Spearman 相关，再对有效日期等权平均。
    Compute cross-sectional Spearman correlation, averaged equally over valid days.

    验证标签在各轮之间不变，因此先缓存其排名。每日至少需要 20 个有效
    证券；常量排名和无效相关不参与均值。没有合格日期时返回 NaN。
    Validation labels are fixed across epochs, so their ranks are cached. A day
    needs at least 20 valid names and nonconstant ranks; no eligible day yields NaN.
    """

    def __init__(self, truth):
        self.truth = np.asarray(truth)
        self.masks = np.isfinite(self.truth)
        self.ranks = [rankdata(row[mask]) for row, mask in zip(self.truth, self.masks)]

    def score(self, prediction) -> float:
        prediction = np.asarray(prediction)
        if prediction.shape != self.truth.shape:
            raise ValueError("Prediction and validation target shapes differ")
        values = []
        for pred, truth, mask, ranks in zip(prediction, self.truth, self.masks, self.ranks):
            valid = mask & np.isfinite(pred)
            if valid.sum() >= 20:
                # 若预测缺失导致证券集合改变，标签也必须在同一子集重新排名。
                # If missing predictions change the universe, rerank labels on that subset.
                y = ranks if np.array_equal(valid, mask) else rankdata(truth[valid])
                x = rankdata(pred[valid])
                if x.std() == 0 or y.std() == 0:
                    continue
                correlation = np.corrcoef(x, y)[0, 1]
                if np.isfinite(correlation):
                    values.append(correlation)
        return float(np.mean(values)) if values else float("nan")


def predict(model, panel, days, batch_days=2):
    """按日期分批推理，返回 [所选日期, 股票] 分数矩阵。
    Infer in day batches and return a [selected days, stocks] score matrix.

    eval() 关闭 dropout，inference_mode() 省去梯度记录。这里不更新权重；
    输出仍包含所有股票列，后续选股由调用方结合有效性标记处理。
    eval() disables dropout and inference_mode() omits gradient tracking. Weights
    stay fixed; output keeps all stock columns for the caller's eligibility filter.
    """
    if not isinstance(batch_days, int) or batch_days < 1:
        raise ValueError("batch_days must be a positive integer")
    device = configure_runtime(model.kind, next(model.parameters()).device)
    cache = panel if isinstance(panel, DevicePanel) else DevicePanel(panel, model.kind, device)
    if cache.device != device or cache.kind != model.kind or cache.history != model.history:
        raise ValueError("Prediction cache does not match the model")
    selected = _days(days, cache.x_device.shape[0], "prediction days")
    if not len(selected):
        return np.empty((0, cache.x_device.shape[1]), dtype=np.float32)
    indices = torch.as_tensor(selected, device=device, dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        outputs = [model(*cache.batch(indices[start:start + batch_days]))
                   for start in range(0, len(indices), batch_days)]
        result = torch.cat(outputs).float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model predictions")
    return result


def fit_network(panel, kind, spec, train_days, train_mask, valid_days, valid_mask,
                epochs, patience, seed, batch_days, device="cpu", execution="optimized"):
    """从随机初始化拟合网络，以标签均方误差更新参数。
    Fit from random initialization using mean squared error on the supplied labels.

    本包的标签是五日收益的横截面排名变换，模型输出用于排序；损失值不是
    收益率预测误差。有效样本由日期列表、掩码、在场标记和有限标签共同决定。
    Package labels are transformed cross-sectional ranks of five-session returns.
    The loss measures rank-target error. Dates, masks, presence and finite labels
    jointly define the observed samples.

    有验证日期时按平均 RankIC 选轮数并恢复最佳参数。无验证日期时训练固定
    轮数，供调用方在更大的训练段重新拟合；每次调用都会创建新模型和优化器。
    With validation dates, select epochs by mean RankIC and restore the best state.
    Without them, run a fixed epoch count for refitting on a larger training slice.
    Each call creates a fresh model and optimizer.
    """
    if kind not in MODEL_NAMES or execution not in ("eager", "optimized"):
        raise ValueError("Unknown model or execution policy")
    if (not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_days, int) or batch_days < 1
            or (patience is not None and (not isinstance(patience, int) or patience < 1))):
        raise ValueError("Positive integer epochs, batch_days and optional patience required")
    if not np.isfinite(spec["lr"]) or spec["lr"] <= 0 or not np.isfinite(spec["weight_decay"]) or spec["weight_decay"] < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    if int(panel.history) != int(spec["history"]):
        raise ValueError("Panel history must match the model specification")
    selected_device = configure_runtime(kind, device)
    cache = DevicePanel(panel, kind, selected_device)
    train_days = _days(train_days, len(panel.x), "training days")
    valid_days = _days(valid_days, len(panel.x), "validation days")
    if not len(train_days) or np.intersect1d(train_days, valid_days).size:
        raise ValueError("Training days must be nonempty and separate from validation days")
    train_mask, valid_mask = np.asarray(train_mask), np.asarray(valid_mask)
    if any(mask.shape != panel.present.shape or mask.dtype != np.bool_ for mask in (train_mask, valid_mask)):
        raise ValueError("Training and validation masks must be boolean panel-shaped arrays")
    observed = train_mask & panel.present & np.isfinite(panel.target)
    if not observed[train_days].any():
        raise ValueError("No observed training labels")
    # 已验收的图路径覆盖这三类模型；其余模型仍受益于设备缓存和批量索引传输。
    # The accepted graph path covers these families; the others still use cached data and bulk indices.
    graph_enabled = selected_device.type == "cuda" and execution == "optimized" and kind in ("mlp", "gru", "gnn")
    if graph_enabled:
        gc.collect()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    target = torch.as_tensor(panel.target, device=selected_device)
    model = FullRankNet(kind, len(panel.features), spec).float().to(selected_device)
    if graph_enabled:
        from .native_graph import capture
        model = capture(model, cache.batch(train_days[:batch_days]))
    model.execution_policy = "cuda_graph" if graph_enabled else "eager"
    # AdamW 在图预热后创建；预热不会推进优化器步数，更新仍发生在每个有效批次。
    # Create AdamW after graph warmup; only observed training batches advance optimizer steps.
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    validation = (ValidationRanks(np.where(valid_mask[valid_days] & panel.present[valid_days],
                                           panel.target[valid_days], np.nan)) if len(valid_days) else None)
    best_ic, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = np.random.default_rng(seed + epoch).permutation(train_days)
        batches = [shuffled[i:i + batch_days] for i in range(0, len(shuffled), batch_days)]
        # 一个批次包含完整的若干日横截面；位置索引指向展平后的 [批内日期, 股票]。
        # Each batch holds whole cross-sections; positions index flattened [batch days, stocks].
        positions = [np.flatnonzero(observed[days]) for days in batches]
        # 每轮只传一次日期和有效标签索引；不改变小批次及样本顺序。
        # Upload the epoch plan once, retaining the same minibatches and sample order.
        device_days = torch.as_tensor(shuffled, device=selected_device)
        device_positions = torch.as_tensor(np.concatenate(positions), device=selected_device)
        offset, losses = 0, []
        for i, position in enumerate(positions):
            indices = device_positions[offset:offset + len(position)]
            offset += len(position)
            if not len(position):
                continue
            days = device_days[i * batch_days:(i + 1) * batch_days]
            truth = target[days].reshape(-1).index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*cache.batch(days))
            # 先抽取有效标签再求均值，缺失标签不会进入损失；每批有效观测等权。
            # Gather valid labels before averaging so missing targets never enter the loss.
            # Valid observations have equal weight within a batch.
            loss = ((prediction.reshape(-1).index_select(0, indices) - truth) ** 2).mean()
            loss.backward()
            # 对全网梯度的总范数限幅，再执行一次 AdamW 参数更新。
            # Clip the global gradient norm before one AdamW update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
        # 每轮只同步一次损失；非有限值使本次拟合失败，不发布检查点。
        # Synchronize losses once per epoch; any non-finite value rejects the fit.
        loss_values = torch.stack(losses).cpu().numpy() if losses else np.array([])
        if not len(loss_values) or not np.isfinite(loss_values).all():
            raise ValueError("Empty or non-finite training epoch")
        ic = None
        if validation is not None:
            ic = validation.score(predict(model, cache, valid_days, batch_days))
            # 提升须超过 1e-5；微小波动和无效 IC 都会累加未改善轮数。
            # Require an improvement above 1e-5; tiny changes and invalid IC count as stale epochs.
            if np.isfinite(ic) and ic > best_ic + 1e-5:
                best_ic, best_epoch, stale = float(ic), epoch, 0
                # 最佳参数保留在训练设备，避免每轮将整网搬回 CPU。
                # Keep the best parameters on the training device until fitting is complete.
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
        # 日志记录批次 MSE 的等权均值；有效样本数不同的批次仍各占一份权重。
        # Log the unweighted mean of batch MSE values, even when observed batch sizes differ.
        curve.append(dict(epoch=epoch, train_mse=float(loss_values.astype(np.float64).mean()), validation_rankic=ic))
        if validation is not None and patience is not None and stale >= patience:
            break
    if validation is not None:
        if best_state is None:
            raise ValueError("No finite validation RankIC; need at least 20 observed stocks with variable scores")
        model.load_state_dict(best_state)
    if not bool(torch.stack([torch.isfinite(value).all() for value in model.parameters()]).all()):
        raise ValueError("Non-finite final model parameters")
    return model, best_epoch if validation is not None else epochs, best_ic if validation is not None else float("nan"), curve
