# SMELL 3 token_selector NEW — 层级 token 选择器：Jenga predictor + 块 topk（CATV 占位）

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


class CATVSelector(BaseTokenSelector):
    # SMELL 3 CATVSelector NEW — 占位，待 CATV 设计

    def select(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.LongTensor:
        raise NotImplementedError("CATVSelector 占位，待 CATV 设计")
