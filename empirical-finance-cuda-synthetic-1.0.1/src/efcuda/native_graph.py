"""复用固定形状的前向与反向 CUDA 操作，减少逐步从 Python 发起调用的开销。
Replay fixed-shape forward/backward CUDA operations to reduce Python launch overhead.

捕获范围是网络调用；损失、梯度裁剪和 AdamW 更新仍在 engine.py 的批次循环中。
The captured region is the network call; loss, gradient clipping and AdamW updates
remain in the engine.py batch loop.
"""

import torch


def capture(model, sample_args):
    """用样本批次确定图的形状，并给模型安装训练时的图调用入口。
    Use a sample batch to establish graph shapes and wrap training-time calls.

    图捕获会执行三次预热，使 CUDA 的内部缓存和工作区就绪。预热会消耗
    dropout 的随机数，因此结束后恢复 CPU 和当前 CUDA 设备的随机状态。
    Three warmup iterations prepare CUDA caches and workspaces. Since dropout
    consumes random numbers, restore CPU and current-device RNG states afterward.

    此处没有优化器；预热不更新参数或推进 AdamW 步数。真正的更新由训练循环执行。
    No optimizer exists here: warmup neither updates parameters nor advances AdamW;
    the training loop performs the actual updates.
    """
    device = next(model.parameters()).device
    if device.type != "cuda" or model.kind not in ("mlp", "gru", "gnn"):
        raise ValueError("Native graphs require CUDA and an approved network family")
    cuda_state = torch.cuda.get_rng_state(device)
    cpu_state = torch.get_rng_state()
    empty = torch.empty(0, device=device)
    # 图接口需要张量实参；把 MLP/GNN 不使用的历史输入替换为空张量占位。
    # Graph calls require tensor arguments; empty tensors stand in for unused MLP/GNN history.
    tensors = tuple(empty if value is None else value for value in sample_args)
    eager = model.forward
    try:
        # 参数仍属于同一个模块；保存 state_dict 时不需要特殊格式。
        # Parameters remain on the original module, so state_dict keeps its ordinary format.
        with torch.cuda.device(device):
            model = torch.cuda.make_graphed_callables(
                model, tensors, num_warmup_iters=3, allow_unused_input=True,
            )
        graphed = model.forward

        def forward(*values):
            # 缓存保证股票数和历史长度固定；当前特征形状改变通常意味着尾批较小。
            # The cache fixes stock count and history length; a changed current shape usually means a tail batch.
            # 验证关闭 dropout，形状不同的尾批也走 eager，避免把不匹配输入交给固定图。
            # Evaluation and differently shaped tails use eager execution instead of the fixed graph.
            if not model.training or values[0].shape != tensors[0].shape:
                return eager(*values)
            return graphed(*(empty if value is None else value for value in values))

        model.forward = forward
        return model
    finally:
        # 捕获失败也恢复 RNG；错误交给调用方，不能悄悄改走另一条训练路径。
        # Restore RNG even on failure; report the failure instead of silently changing execution.
        torch.cuda.set_rng_state(cuda_state, device)
        torch.set_rng_state(cpu_state)
