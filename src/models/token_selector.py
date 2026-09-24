# SMELL 3 token_selector NEW — 层级 token 选择器：Jenga predictor + 块 topk（CATV 占位）
# SMELL 3 token_selector REWRITTEN — 实装 CATVSelector（共识锚掩码）+ compute_consensus_mask / mask_intersection_rate + CPU 自测

# SMELL 3 token_selector imports ADD — math.floor 用于 k=floor(r*N)
import math

import torch
import torch.nn as nn

from jenga.models.predictor import PrunableAttnPredictorInfer


class BaseTokenSelector:
    """按层选择保留 token 的接口；select 返回该层保留 token 的索引（升序）。"""

    def __init__(self, config):
        self.config = config

    def select(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.LongTensor:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        raise NotImplementedError


class JengaSparseSelector(BaseTokenSelector, nn.Module):
    """复刻 Jenga OPT 块稀疏：predictor 打分 → 块 topk → 展开为 token 索引。"""

    BLOCK_SIZE = 64

    def __init__(self, config):
        nn.Module.__init__(self)
        BaseTokenSelector.__init__(self, config)
        self.predictors = nn.ModuleDict()

    def _num_blocks(self, q_len: int) -> int:
        if q_len % self.BLOCK_SIZE != 0:
            raise ValueError(
                f"sequence length {q_len} must be divisible by block size {self.BLOCK_SIZE}"
            )
        return q_len // self.BLOCK_SIZE

    def _make_predictor(self):
        return PrunableAttnPredictorInfer(
            dim=self.config.hidden_size // self.config.num_attention_heads,
            hidden_dim=128,
            n_head=self.config.num_attention_heads,
        )

    def _get_predictor(self, layer_idx: int, hidden_states: torch.Tensor) -> nn.Module:
        key = str(layer_idx)
        if key not in self.predictors:
            predictor = self._make_predictor()
            predictor.eval()
            self.predictors[key] = predictor
        predictor = self.predictors[key]
        ref = predictor.qlinear1.weight
        if ref.device != hidden_states.device or ref.dtype != hidden_states.dtype:
            predictor = predictor.to(device=hidden_states.device, dtype=hidden_states.dtype)
            self.predictors[key] = predictor
        return predictor

    def select(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.LongTensor:
        bsz, q_len, _ = hidden_states.size()
        num_blocks = self._num_blocks(q_len)
        if layer_idx < self.config.num_hidden_layers // 2 - 1:
            idx = torch.arange(q_len, device=hidden_states.device)
            if bsz == 1:
                return idx
            return idx.unsqueeze(0).expand(bsz, q_len).contiguous()

        with torch.no_grad():
            predictor = self._get_predictor(layer_idx, hidden_states)
            block_scores = predictor(hidden_states).sum(dim=-2)  # (bsz, num_blocks)
            if block_scores.size(-1) != num_blocks:
                raise ValueError(
                    f"predictor returned {block_scores.size(-1)} blocks, expected {num_blocks}"
                )
            keep = int(num_blocks * getattr(self.config, "sparse", 0.2))
            keep = max(1, min(keep, num_blocks))
            topk_idx = torch.topk(block_scores, keep, largest=True, dim=-1).indices
            topk_idx, _ = torch.sort(topk_idx, dim=-1)
            base = torch.arange(self.BLOCK_SIZE, device=hidden_states.device)
            expanded = (topk_idx[..., None] * self.BLOCK_SIZE + base).reshape(bsz, -1)

        if bsz == 1:
            return expanded.reshape(-1).sort().values
        return expanded.sort(dim=-1).values

    def state_dict(self) -> dict:
        return nn.Module.state_dict(self)

    def load_state_dict(self, state_dict: dict):
        layer_keys = sorted(
            {k.split(".")[1] for k in state_dict if k.startswith("predictors.")},
            key=int,
        )
        for layer_idx in layer_keys:
            if layer_idx not in self.predictors:
                ref = next(
                    v for k, v in state_dict.items() if k.startswith(f"predictors.{layer_idx}.")
                )
                predictor = self._make_predictor().to(device=ref.device, dtype=ref.dtype)
                predictor.eval()
                self.predictors[layer_idx] = predictor
        result = nn.Module.load_state_dict(self, state_dict, strict=True)
        self.predictors.eval()
        return result


# SMELL 3 CATV helpers BEGIN — compute_consensus_mask / mask_intersection_rate
def compute_consensus_mask(votes_by_layer: dict, n_blocks: int, r: float, s: float) -> dict:
    """CATV 服务器端：聚合各 client 的块投票 → 每层 ±inf 共识锚掩码。

    votes_by_layer: {layer_idx: tensor[N] 或 list[N]}，客户端上传的原始块分数之和
    n_blocks: 每层块数 N（16k / pool_size 64 = 256）
    r: anchor ratio，top/bottom k=floor(r*N) 块分别强制保留/排除
    s: sparse budget（top s*N 保留）
    返回 {layer_idx: float32 tensor[N]}，元素 ∈ {+inf, -inf, 0}
    约束（作者确认）：r < s（否则所有 client 掩码相同）且 s + r <= 1（中间块才能填满预算）
    """
    r = float(r)
    s = float(s)
    n_blocks = int(n_blocks)
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    if r <= 0.0:
        raise ValueError(f"CATV anchor ratio r must be in (0, 1), got r={r}")
    if r >= s:
        raise ValueError(
            f"CATV requires anchor ratio r < sparse budget s, got r={r} >= s={s}"
        )
    if s + r > 1.0 + 1e-9:
        raise ValueError(
            f"CATV requires s + r <= 1 so middle blocks can fill the budget, got s={s}, r={r}"
        )
    k = int(math.floor(r * n_blocks))
    if k < 1:
        raise ValueError(f"CATV anchor count floor(r*N) must be >= 1, got r={r}, N={n_blocks}")
    masks = {}
    for layer_idx, votes in votes_by_layer.items():
        scores = torch.as_tensor(votes, dtype=torch.float32).reshape(-1)
        if scores.numel() != n_blocks:
            raise ValueError(
                f"layer {layer_idx} vote has {scores.numel()} blocks, expected {n_blocks}"
            )
        # 确定性 tie-break：按 (-score, index) 排序（stable descending 保留原索引升序）
        order = torch.argsort(scores, descending=True, stable=True)
        layer_mask = torch.zeros(n_blocks, dtype=torch.float32)
        layer_mask[order[:k]] = float("inf")
        layer_mask[order[n_blocks - k:]] = float("-inf")
        masks[int(layer_idx)] = layer_mask
    return masks


def mask_intersection_rate(client_vote_block_set, central_anchor_set) -> float:
    """IR = |M_local ∩ M_central| / |M_central|（M_central 为空时返回 0.0）。"""
    central = {int(index) for index in central_anchor_set}
    if not central:
        return 0.0
    local = {int(index) for index in client_vote_block_set}
    return len(local & central) / float(len(central))
# SMELL 3 CATV helpers END


class CATVSelector(BaseTokenSelector):
    # SMELL 3 CATVSelector REWRITTEN — 持有服务器下发的共识锚掩码，apply 时加到 predictor 原始分数上

    def __init__(self, config):
        BaseTokenSelector.__init__(self, config)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    # SMELL 3 CATVSelector apply ADD — raw_scores 原样加掩码（+inf/-inf/0），无掩码时不变
    def apply(self, raw_scores: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.mask is None or int(layer_idx) not in self.mask:
            return raw_scores
        layer_mask = self.mask[int(layer_idx)]
        return raw_scores + layer_mask.to(device=raw_scores.device, dtype=raw_scores.dtype)

    def select(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.LongTensor:
        raise NotImplementedError(
            "CATV 选择在 OptFlashAttention2 内通过 config.consensus_mask 完成；请使用 apply()"
        )


# SMELL 3 token_selector selftest BEGIN — CATV 掩码/约束/IR/tie-break 覆盖
def _selftest() -> bool:
    results = {}
    votes = {3: torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])}
    mask = compute_consensus_mask(votes, 10, 0.2, 0.4)
    layer_mask = mask[3]
    results["mask shape"] = tuple(layer_mask.shape) == (10,)
    results["mask dtype"] = layer_mask.dtype == torch.float32
    results["anchor_in count"] = int(torch.isposinf(layer_mask).sum()) == 2
    results["anchor_out count"] = int(torch.isneginf(layer_mask).sum()) == 2
    results["non-anchor zero"] = int((layer_mask == 0).sum()) == 6
    results["top blocks +inf"] = bool(
        torch.isposinf(layer_mask[9]) and torch.isposinf(layer_mask[8])
    )
    results["bottom blocks -inf"] = bool(
        torch.isneginf(layer_mask[0]) and torch.isneginf(layer_mask[1])
    )

    tie_mask = compute_consensus_mask({0: torch.ones(10)}, 10, 0.3, 0.5)[0]
    tie_in = set(int(index) for index in torch.isposinf(tie_mask).nonzero().reshape(-1).tolist())
    tie_out = set(int(index) for index in torch.isneginf(tie_mask).nonzero().reshape(-1).tolist())
    results["tie-break +inf"] = tie_in == {0, 1, 2}
    results["tie-break -inf"] = tie_out == {7, 8, 9}

    r_ge_s_raised = False
    try:
        compute_consensus_mask(votes, 10, 0.4, 0.4)
    except ValueError:
        r_ge_s_raised = True
    results["r >= s raises"] = r_ge_s_raised

    sum_gt_one_raised = False
    try:
        compute_consensus_mask(votes, 10, 0.2, 0.9)
    except ValueError:
        sum_gt_one_raised = True
    results["s + r > 1 raises"] = sum_gt_one_raised

    results["IR full overlap"] = mask_intersection_rate({1, 2, 3}, {1, 2}) == 1.0
    results["IR partial overlap"] = mask_intersection_rate({1, 7}, {1, 2}) == 0.5
    results["IR disjoint"] = mask_intersection_rate({4, 5}, {1, 2}) == 0.0

    selector = CATVSelector(config=None)
    raw = torch.tensor([1.0, 2.0, 3.0])
    results["apply without mask"] = torch.equal(selector.apply(raw, 0), raw)
    selector.set_mask({0: torch.tensor([float("inf"), 0.0, float("-inf")])})
    applied = selector.apply(raw, 0)
    results["apply with mask"] = bool(
        torch.isposinf(applied[0]) and applied[1] == 2.0 and torch.isneginf(applied[2])
    )

    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return all(results.values())


if __name__ == "__main__":
    ok = _selftest()
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
# SMELL 3 token_selector selftest END
