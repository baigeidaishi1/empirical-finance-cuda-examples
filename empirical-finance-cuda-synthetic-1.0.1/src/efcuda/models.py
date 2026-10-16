"""五类网络共享输入约定和逐证券打分头。
Five network families share an input contract and per-stock scoring head.

B 表示批内日期数，N 为股票数，L 为历史长度，F 为因子数，H 为隐藏宽度。
B denotes batch days, N stocks, L history length, F factors and H hidden width.
MLP/GNN 使用 [B, N, F] 当日横截面，其余网络使用 [B, N, L, F] 历史窗口。
MLP/GNN use current [B, N, F] cross-sections; the other networks use [B, N, L, F] history.
"""

from __future__ import annotations

import math

import torch
from torch import nn

MODEL_NAMES = ("mlp", "gru", "gnn", "attention", "transformer")


class ResidualBlock(nn.Module):
    """沿最后一维做归一化和非线性变换，并保留同宽度残差通路。
    Normalize and transform the last dimension, with a same-width residual path.
    """

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FullRankNet(nn.Module):
    """输出 [B, N] 排序分数，与训练标签的尺度对应。
    Produce [B, N] ranking scores on the scale of the training target.

    每个分支最终生成 [B, N, H] 隐藏表示，再由同样结构的线性头输出标量。
    交易日选择、组合权重与成本计算由上层流程负责。
    Every branch produces [B, N, H] hidden states for a common head architecture.
    The surrounding workflow handles trade timing, portfolio weights and costs.
    """

    def __init__(self, kind: str, inputs: int, spec: dict):
        super().__init__()
        if kind not in MODEL_NAMES:
            raise ValueError(f"Unknown network: {kind}")
        self.kind = kind
        self.history = int(spec["history"])
        width, depth = int(spec["hidden"]), int(spec["layers"])
        dropout = float(spec["dropout"])
        heads = int(spec.get("heads", 1))
        if min(inputs, self.history, width, depth, heads) < 1 or not 0 <= dropout < 1:
            raise ValueError("Positive network dimensions and 0 <= dropout < 1 required")
        if kind in ("attention", "transformer") and width % heads:
            raise ValueError("Hidden width must be divisible by attention heads")
        if kind == "transformer" and width % 2:
            raise ValueError("Sinusoidal positions require an even hidden width")

        self.project = nn.Linear(inputs, width)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 1))
        if kind == "mlp":
            # MLP 独立处理每只股票的当日因子，不在股票之间或历史日期之间传递信息。
            # MLP processes each stock's current factors independently across stocks and dates.
            self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(depth)))
        elif kind == "gru":
            # 一层 GRU 没有层间 dropout；输出头的 dropout 仍生效。
            # A one-layer GRU has no inter-layer dropout; the output head still does.
            self.gru = nn.GRU(inputs, width, num_layers=depth,
                              dropout=dropout if depth > 1 else 0.0, batch_first=True)
        elif kind == "gnn":
            self.graph_blocks = nn.ModuleList(ResidualBlock(width, dropout) for _ in range(depth))
        elif kind == "attention":
            # 一个可学习查询汇总单只股票的历史；本分支没有位置编码。
            # One learned query pools each stock's history; this branch has no position encoding.
            self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.query = nn.Parameter(torch.zeros(1, 1, width))
            self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(depth)))
        else:
            layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            # 正弦位置编码保存为 buffer，随模型搬移和保存，无需梯度更新。
            # Sinusoidal positions are buffers: moved and saved with the model, without gradients.
            steps = torch.arange(self.history, dtype=torch.float32).unsqueeze(1)
            divisor = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            position = torch.zeros(self.history, width)
            position[:, 0::2] = torch.sin(steps * divisor)
            position[:, 1::2] = torch.cos(steps * divisor)
            self.register_buffer("position", position)

    def forward(self, current: torch.Tensor, sequence: torch.Tensor | None,
                sequence_mask: torch.Tensor | None, present: torch.Tensor) -> torch.Tensor:
        """共享权重跨股票计算；各日期批次之间没有信息交互。
        Apply shared weights across stocks without mixing separate dates in a batch.

        在场标记用于构图与序列处理；输出保留所有股票槽位，训练损失另行筛选
        有效样本，因此不能直接把输出中的每一列都视为可交易证券。
        Presence flags control graph and sequence handling. Outputs keep every stock
        slot; the loss selects valid samples separately from these raw scores.
        """
        if self.kind == "mlp":
            hidden = self.blocks(torch.nn.functional.gelu(self.project(current)))
        elif self.kind == "gnn":
            # 用当日已变换因子的欧氏距离找最多 8 个近邻，形成随日期变化的相似图。
            # Build a daily similarity graph from up to eight nearest neighbors in transformed features.
            # 这里没有行业关系或供应链关系输入，边仅反映当前因子空间的接近程度。
            # Edges reflect feature-space proximity; industry and supply-chain relations are absent.
            distance = torch.cdist(current, current)
            names = current.shape[1]
            diagonal = torch.eye(names, device=current.device, dtype=torch.bool)[None]
            distance = distance.masked_fill(~present[:, None, :] | diagonal, float("inf"))
            nearest = distance.topk(min(8, names - 1), largest=False).indices
            adjacency = torch.zeros_like(distance).scatter_(2, nearest, 1.0)
            # 近邻关系取双向并集，加自环，再去掉缺席证券的所有边。
            # Symmetrize neighbor links, add self-loops and remove edges touching absent stocks.
            adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
            adjacency = ((adjacency + diagonal.to(adjacency.dtype))
                         * present[:, :, None] * present[:, None, :])
            degree = adjacency.sum(-1).clamp(min=1).rsqrt()
            # D^(-1/2) A D^(-1/2) 缩放邻居聚合，避免度数直接放大隐藏表示。
            # D^(-1/2) A D^(-1/2) scales aggregation so degree alone does not inflate hidden states.
            adjacency = adjacency * degree[:, :, None] * degree[:, None, :]
            hidden = torch.nn.functional.gelu(self.project(current))
            for block in self.graph_blocks:
                hidden = block(adjacency @ hidden)
        else:
            batch, names, length, inputs = sequence.shape
            # 把股票并入批次轴：[B, N, L, F] -> [B*N, L, F]；共享网络逐股处理历史。
            # Fold stocks into the batch axis: [B, N, L, F] -> [B*N, L, F] for shared temporal layers.
            values = sequence.reshape(batch * names, length, inputs)
            mask = sequence_mask.reshape(batch * names, length).clone()
            # 全缺失序列保留一个值为零的末位，避免注意力面对全部被屏蔽的键而产生 NaN。
            # Give an empty sequence one zero-valued final slot to avoid attention over all-masked keys.
            # 固定列更新保持形状不变，也避免动态 nonzero 索引引入设备同步。
            # Updating a fixed column preserves shape and avoids synchronization from dynamic indices.
            mask[:, -1] |= ~mask.any(1)
            values = values * mask[:, :, None]
            if self.kind == "gru":
                # GRU 逐步读取零填充的序列，最后取有效日期的状态；填充步仍经过递推。
                # GRU steps through the zero-filled sequence; select the final valid state afterward.
                # 当前实现没有压缩缺失步，历史中的空缺仍会影响后续隐藏状态。
                # Missing steps are not packed away and can affect subsequent hidden states.
                steps, _ = self.gru(values)
                last = torch.where(mask, torch.arange(length, device=mask.device), -1).max(1).values
                hidden = steps[torch.arange(len(last), device=steps.device), last]
            else:
                projected = self.project(values)
                if self.kind == "attention":
                    query = self.query.expand(len(values), -1, -1)
                    hidden = self.attention(query, projected, projected,
                                            key_padding_mask=~mask, need_weights=False)[0][:, 0]
                    hidden = self.blocks(hidden)
                else:
                    # 编码器可查看窗口内全部有效位置；窗口截止于信号日，输入没有未来日期。
                    # The encoder attends across valid window positions, all at or before the signal day.
                    encoded = self.encoder(projected + self.position[None], src_key_padding_mask=~mask)
                    last = torch.where(mask, torch.arange(length, device=mask.device), -1).max(1).values
                    hidden = encoded[torch.arange(len(last), device=encoded.device), last]
            hidden = hidden.reshape(batch, names, -1)
        return self.output(hidden).squeeze(-1)
