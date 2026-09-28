# SMELL 3 modeling_opt_smell COPY — 从 third_party/Jenga/src/jenga/models/modeling_opt.py 原样复制；后续在此接入 TokenSelector/CATV
# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OPT model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.opt.configuration_opt import OPTConfig
from jenga.models.predictor import PrunableAttnPredictorInfer

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)
def pack_hook(tensor):
    with torch.no_grad():
        seq_len = tensor.size(0) 
        return (tensor[:seq_len//2,:],seq_len)

# unpack_hook: 反向时，再随便把一半元素置 0
# 这里不需要还原前向的内容，因为我们只测性能，不关心准确性
def unpack_hook(saved):
    # (packed, dtype) = saved
    # mask_bwd = (torch.rand_like(packed) > 0.5).float()
    with torch.no_grad():
        
        (x, seq_len) = saved
        out = torch.zeros(seq_len, x.shape[1], device=x.device, dtype=x.dtype)
        out[:seq_len//2,:] = x
        # return (packed * mask_bwd).to(dtype)
        return out

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias
        self.layer_idx = layer_idx

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        # SMELL 3 modeling_opt_smell predictor_layers BEGIN — 上游 OPTAttention 忽略 config.predictor_layers（llama 路径已支持），
        # 这里按 pruned_config 的逐层 outdim 重建剪枝后的 predictor 形状；无配置时退回默认（未剪枝）形状。
        predictor_kwargs = {}
        predictor_layers = getattr(config, "predictor_layers", None)
        if predictor_layers is not None and layer_idx is not None:
            layer_cfg = predictor_layers[layer_idx]
            if layer_cfg:
                predictor_kwargs = {
                    "q1_outdim": layer_cfg["q1_outdim"],
                    "q2_outdim": layer_cfg["q2_outdim"],
                    "k1_outdim": layer_cfg["k1_outdim"],
                    "k2_outdim": layer_cfg["k2_outdim"],
                }
        self.predictor = PrunableAttnPredictorInfer(
            dim=int(config.hidden_size / config.num_attention_heads),
            hidden_dim=128,
            n_head=config.num_attention_heads,
            **predictor_kwargs,
        )
        # SMELL 3 modeling_opt_smell predictor_layers END
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OptFlashAttention2(OPTAttention):
    """
    OPT flash attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
    The only required change would be on the forward pass where it needs to correctly call the public API of flash
    attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, q_len, _ = hidden_states.size()
        if self.layer_idx != self.config.num_hidden_layers-1: 
            with torch.no_grad():
                # attn_maxpool = block_attn_pool(query_states.transpose(1,2).contiguous(), key_states.transpose(1,2).transpose(2,3).contiguous())
                hidden_states_ = hidden_states.clone()
                predict_attn = self.predictor(hidden_states_)
                sum_q = predict_attn.sum(dim=-2)
                # SMELL 3 vote_callback ADD — CATV: 上报原始块分数（topk/掩码之前，无梯度）
                vote_callback = getattr(self.config, "vote_callback", None)
                if vote_callback is not None:
                    vote_callback(self.layer_idx, sum_q.detach())
                # SMELL 3 consensus_mask ADD — CATV: 服务器共识锚掩码注入原始分数（+inf 强制保留 / -inf 强制排除）
                consensus_mask = getattr(self.config, "consensus_mask", None)
                if consensus_mask is not None and self.layer_idx in consensus_mask:
                    # SMELL 3 consensus_mask ADD — 掩码转至 sum_q 的 device/dtype 后相加
                    sum_q = sum_q + consensus_mask[self.layer_idx].to(device=sum_q.device, dtype=sum_q.dtype)
                if self.layer_idx < self.config.num_hidden_layers//2 - 1:
                    q_len_now = int(sum_q.size(1))
                else:
                    q_len_now = int(sum_q.size(1) * self.config.sparse)
                
                _, idx = torch.topk(sum_q, q_len_now, largest=True, dim=-1)
                idx = idx.sort().values
                
                # expanded_idx = []
                # for n in idx:
                #     expanded_idx.extend(range(n * 64, (n + 1) * 64))
                # idx = torch.tensor(expanded_idx, device=query_states.device)
                base = torch.arange(64, device=idx.device).view(1, 1, 64)  # (1, 1, 64)
                expanded_idx = idx[..., None] * 64 + base  # (bsz, q_len_now, 64)
                idx = expanded_idx.view(-1)  # (bsz, q_len_now * 64)
                
                q_len_now *= 64               
                
                del  sum_q, hidden_states_
            # query_states = query_states[:, idx, :, :]
            hidden_states = hidden_states[:, idx, :]
        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_length = query_states.shape[1]
        tgt_len = key_states.shape[-2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = query_states.view(bsz, query_length, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        if self.layer_idx != self.config.num_hidden_layers-1: 
            ql = q_len_now
        else:
            ql = q_len
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            ql,
            dropout=attn_dropout,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_weights_reshaped = attn_output.reshape(bsz, ql, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights_reshaped)

        if not output_attentions:
            attn_weights_reshaped = None
            
        if self.layer_idx != self.config.num_hidden_layers-1: 
            output = torch.zeros((bsz, q_len, attn_output.shape[-1]), device='cuda:0', dtype=attn_output.dtype)
            output[0].scatter_(0, idx.unsqueeze(1).expand(-1, attn_output.size(-1)), attn_output[0])

            return output, attn_weights_reshaped, past_key_value
        return attn_output, attn_weights_reshaped, past_key_value


class OptSdpaAttention(OPTAttention):
    """
    # SMELL 3 modeling_opt_smell sdpa ADD — PyTorch SDPA 注意力路径：复用 OPTAttention 的 q/k/v/out 投影与 LoRA
    # 逻辑，用 F.scaled_dot_product_attention(is_causal=True) 取代手工 causal mask + softmax + bmm，支持 fp32
    # 并走 mem-efficient 后端（O(seq) 显存），bf16/fp16 亦可用。
    # SMELL 3 modeling_opt_smell sdpa_sparse ADD — 当 config.thresh ∈ (0,1) 且 full-seq self-attn 时，
    # 按 pool_size 分块、每 query 块保留 top-k 个 causal KV 块（k=int(n_blocks*thresh)，Jenga 同口径），
    # 块分优先取 Jenga predictor 分数（(bsz,n_blocks,n_blocks)），失败退化为 q·k 块均值打分；
    # 其余 token 位置加 -inf 掩码后仍走同一个 F.scaled_dot_product_attention。thresh<=0/非法/past/cross → dense。
    """

    def _build_sparse_attn_mask(self, hidden_states, query_states, key_states, bsz, tgt_len):
        # SMELL 3 modeling_opt_smell sdpa_sparse ADD — 返回 (bsz,1,L,L) 加性掩码；不满足稀疏条件返回 None（dense 退路）
        pool = int(getattr(self.config, "pool_size", 64))
        thresh = getattr(self.config, "thresh", None)
        if thresh is None or not (0.0 < float(thresh) < 1.0):
            return None
        if pool <= 0 or tgt_len % pool != 0:
            return None
        src_len = key_states.shape[-2]
        if src_len != tgt_len:
            return None
        n_blocks = tgt_len // pool
        if n_blocks < 2:
            return None

        device = query_states.device
        with torch.no_grad():
            # SMELL 3 modeling_opt_smell sdpa_sparse ADD — 块分优先 predictor；形状不符/不可用则 q·k 块均值退化
            scores = None
            scored_by = "qk_block_mean"
            try:
                hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
                predict_attn = self.predictor(hidden)
                if predict_attn.dim() == 3 and tuple(predict_attn.shape[-2:]) == (n_blocks, n_blocks):
                    scores = predict_attn
                    scored_by = "predictor"
            except RuntimeError:
                scores = None
            if scores is None:
                q_block = query_states.view(bsz, self.num_heads, n_blocks, pool, self.head_dim).mean(dim=3)
                k_block = key_states.view(bsz, self.num_heads, n_blocks, pool, self.head_dim).mean(dim=3)
                scores = torch.einsum("bhid,bhjd->bij", q_block, k_block)
            scores = scores.reshape(bsz, n_blocks, n_blocks).float()

            # SMELL 3 modeling_opt_smell sdpa_sparse ADD — 仅在 causal 候选块内 top-k，保证每行至少 1 个可见 key
            block_idx = torch.arange(n_blocks, device=device)
            causal_blocks = block_idx[:, None] >= block_idx[None, :]
            scores = scores.masked_fill(~causal_blocks[None], float("-inf"))
            k = min(max(1, int(n_blocks * float(thresh))), n_blocks)
            top_idx = torch.topk(scores, k, dim=-1).indices
            allowed = torch.zeros(bsz, n_blocks, n_blocks, dtype=torch.bool, device=device)
            allowed.scatter_(2, top_idx, True)
            allowed &= causal_blocks[None]

            # SMELL 3 modeling_opt_smell sdpa_sparse ADD — 块级 keep 展开到 token 级并与 token causal 相交
            allowed_tok = allowed.view(bsz, n_blocks, 1, n_blocks, 1).expand(
                bsz, n_blocks, pool, n_blocks, pool).reshape(bsz, 1, tgt_len, tgt_len)
            token_causal = torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device).tril()[None, None]
            keep = allowed_tok & token_causal
            mask = torch.full((bsz, 1, tgt_len, tgt_len), float("-inf"),
                              dtype=query_states.dtype, device=device)
            mask.masked_fill_(keep, 0.0)

            if getattr(self.config, "sdpa_sparse_debug", False):
                per_q = allowed.sum(dim=-1)
                self.last_sparse_stats = {
                    "scored_by": scored_by,
                    "n_blocks": int(n_blocks),
                    "k": int(k),
                    "allowed_blocks_mean": float(per_q.float().mean()),
                    "allowed_blocks_min": int(per_q.min()),
                    "allowed_blocks_max": int(per_q.max()),
                    "token_keep_frac": float(keep.sum()) / float(keep.numel()),
                    "mask_shape": list(mask.shape),
                }
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # SMELL 3 modeling_opt_smell sdpa ADD — cross-attention 与 OPTAttention 语义保持一致
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # SMELL 3 modeling_opt_smell sdpa ADD — 注意：缩放由 SDPA 内部按 head_dim 完成，故此处的 q_proj 不再预先乘 scaling
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            src_len = key_value_states.size(1)
            key_states = self._shape(self.k_proj(key_value_states), src_len, bsz)
            value_states = self._shape(self.v_proj(key_value_states), src_len, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
            value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
            value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        had_past = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_states, value_states)

        # SMELL 3 modeling_opt_smell sdpa ADD — is_causal 仅在 decoder 自注意力且 q_len==k_len 时使用（训练 full-seq）
        is_causal = bool(self.is_causal and not is_cross_attention and tgt_len == key_states.shape[-2])
        # SMELL 3 modeling_opt_smell sdpa_sparse ADD — full-seq self-attn 时构建块级 top-k 掩码（含 causal）；
        # 掩码非空则 is_causal=False（causal 已编码进 mask），否则保持原 dense is_causal 路径
        attn_mask = None
        # SMELL 3 modeling_opt_smell sdpa_prune_fallback FIXED — _enable_sparse_mask=False（子类 OptSdpaPruneAttention 回退时）
        # 跳过 dense (bsz,1,L,L) 加性掩码，走纯 causal SDPA，避免 fp32 math 核 O(L^2) OOM
        if is_causal and not had_past and getattr(self, "_enable_sparse_mask", True):
            attn_mask = self._build_sparse_attn_mask(hidden_states, query_states, key_states, bsz, tgt_len)
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # SMELL 3 modeling_opt_smell sdpa ADD — SDPA 不暴露注意力权重；保持三元组接口，权重占位为 None
        attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped, past_key_value


class OptSdpaSparseGatherAttention(OptSdpaAttention):
    """
    # SMELL 3 modeling_opt_smell sparse_gather ADD — 实验用「gather 式块稀疏」注意力：不再构造 L×L 掩码，
    # 而是先用 predictor（退化 q·k 块均值）得到块分 S[b, n_qblk, n_kvblk]，对每个 query 块只 top-k 个 causal KV 块，
    # 再用 torch.gather 把被选 KV 块的 token 收集成 [b, h, k·pool, dh]，在 pool × (k·pool) 小块上算 SDPA；
    # 每个 query 块用 torch.utils.checkpoint 包裹以限 BP 激活显存。thresh<=0/非法/cross/past/长度不整除 → 回退 dense。
    """

    @staticmethod
    def _sdpa_gather_chunk(query, key, value, tok, attn_mask, dropout_p, pool):
        # SMELL 3 modeling_opt_smell sparse_gather ADD — 单个 query 块：chunk 内 gather KV token 后做 SDPA；
        # gather 放在 checkpoint 内，BP 只保留共享的整段 K/V 与每块 token 索引，避免每块各存一份 gathered K/V
        bsz, heads, _, head_dim = query.shape
        g_index = tok[:, None, :, None].expand(bsz, heads, tok.shape[1], head_dim)
        k_g = torch.gather(key, 2, g_index)
        v_g = torch.gather(value, 2, g_index)
        return nn.functional.scaled_dot_product_attention(
            query, k_g, v_g, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )

    def _gather_sparse_ok(self, tgt_len):
        # SMELL 3 modeling_opt_smell sparse_gather ADD — 判断是否走 gather 稀疏路径（thresh∈(0,1]、长度按 pool 整除、≥2 块）
        thresh = getattr(self.config, "thresh", None)
        pool = int(getattr(self.config, "pool_size", 64))
        if thresh is None:
            return False
        try:
            t = float(thresh)
        except (TypeError, ValueError):
            return False
        if not (0.0 < t <= 1.0):
            return False
        if pool <= 0 or tgt_len % pool != 0:
            return False
        return (tgt_len // pool) >= 2

    def _block_scores(self, hidden_states, query_states, key_states, bsz, pool):
        # SMELL 3 modeling_opt_smell sparse_gather ADD — 块分优先 Jenga predictor；形状不符/不可用则 q·k 块均值退化；纯打分无梯度
        n_blocks = query_states.shape[-2] // pool
        scores = None
        with torch.no_grad():
            try:
                hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
                predict_attn = self.predictor(hidden)
                if predict_attn.dim() == 3 and tuple(predict_attn.shape[-2:]) == (n_blocks, n_blocks):
                    scores = predict_attn.reshape(bsz, n_blocks, n_blocks).float()
            except (RuntimeError, ValueError):
                scores = None
            if scores is None:
                q_block = query_states.view(bsz, self.num_heads, n_blocks, pool, self.head_dim).mean(dim=3)
                k_block = key_states.view(bsz, self.num_heads, n_blocks, pool, self.head_dim).mean(dim=3)
                scores = torch.einsum("bhid,bhjd->bij", q_block, k_block).float()
        return scores

    def _gather_attention(self, hidden_states, query_states, key_states, value_states, bsz):
        # SMELL 3 modeling_opt_smell sparse_gather ADD — 按 query 块循环 gather 被选 KV token 并在小块上算 SDPA
        pool = int(self.config.pool_size)
        thresh = float(self.config.thresh)
        tgt_len = query_states.shape[-2]
        n_blocks = tgt_len // pool
        k_keep = max(1, int(n_blocks * thresh))
        scores = self._block_scores(hidden_states, query_states, key_states, bsz, pool)
        device = query_states.device
        dtype = query_states.dtype
        neg = torch.finfo(dtype).min
        base = torch.arange(pool, device=device)
        chunk_outputs = []
        for qi in range(n_blocks):
            # SMELL 3 modeling_opt_smell sparse_gather ADD — 每 query 块仅在 causal 候选块内 top-k（kj≤qi+1）
            kj = min(k_keep, qi + 1)
            cand = scores[:, qi, : qi + 1]
            idx_q = torch.topk(cand, kj, dim=-1).indices.sort(dim=-1).values
            tok = (idx_q[..., None] * pool + base).reshape(bsz, kj * pool)
            q_start = qi * pool
            q_chunk = query_states[:, :, q_start:q_start + pool, :]
            # SMELL 3 modeling_opt_smell sparse_gather ADD — 块内 causal：future token 位置加 -inf
            q_global = q_start + base
            add_mask = torch.where(
                tok[:, None, :] > q_global[None, :, None],
                torch.full((), neg, device=device, dtype=dtype),
                torch.zeros((), device=device, dtype=dtype),
            )[:, None, :, :]
            dropout_p = self.dropout if self.training else 0.0
            if self.training and q_chunk.requires_grad:
                # SMELL 3 modeling_opt_smell sparse_gather ADD — checkpoint 每个 chunk（含 gather），BP 时重算以限激活显存
                out = torch.utils.checkpoint.checkpoint(
                    OptSdpaSparseGatherAttention._sdpa_gather_chunk,
                    q_chunk, key_states, value_states, tok, add_mask, dropout_p, pool,
                    use_reentrant=False,
                )
            else:
                out = OptSdpaSparseGatherAttention._sdpa_gather_chunk(
                    q_chunk, key_states, value_states, tok, add_mask, dropout_p, pool
                )
            chunk_outputs.append(out)
        return torch.cat(chunk_outputs, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # SMELL 3 modeling_opt_smell sparse_gather ADD — full-seq causal self-attn 且 thresh 合法时走 gather 稀疏，否则回退父类 dense
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        if (not is_cross_attention) and past_key_value is None and self._gather_sparse_ok(tgt_len):
            query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
            key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
            value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)
            if self.is_decoder:
                past_key_value = (key_states, value_states)
            attn_output = self._gather_attention(hidden_states, query_states, key_states, value_states, bsz)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
            attn_output = self.out_proj(attn_output)
            return attn_output, None, past_key_value
        return super().forward(
            hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )


class OptSdpaPruneAttention(OptSdpaAttention):
    """
    # SMELL 3 modeling_opt_smell sdpa_prune ADD — Jenga 真语义「token 子集化」稀疏注意力（fp32 SDPA 路径）。
    # 权威语义见 JengaForMemoryTest/Jenga/src/jenga/models/modeling_opt.py:294-321：
    #   layer_idx != num_layers-1 时，predictor(hidden_states)->(b,n_blk,n_blk)，sum_q=sum(dim=-2)->(b,n_blk)；
    #   下半层 (layer_idx < num_layers//2 - 1) 保留全部块，上半层保留 int(n_blk*config.sparse) 块；
    #   topk(sum_q).sort() 得 KV 块 idx，展开为 token 索引后 *直接子集化 hidden_states*（L -> kept 个 token），
    #   再在缩短序列上算 q/k/v 与 dense causal SDPA（is_causal=True）；输出 scatter 回全长 L 供残差使用。
    # 末层豁免（走父类 dense）。pool=config.pool_size(64)，sparse=config.sparse(0.4)，均不硬编码。
    # 位置嵌入在 decoder 入口一次性加好（OPTDecoder.forward:1096），层内无位置/无 token 位置算子，
    # 故层内子集化不破坏位置；子集保持原顺序（idx.sort），causal 在缩短序列上仍单调。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SMELL 3 modeling_opt_smell sdpa_prune ADD — 最近一次 forward 的剪枝统计（供 smoke/TD 打印 kept token 数）
        self.last_prune_stats = None
        # SMELL 3 modeling_opt_smell sdpa_prune_fallback FIXED — 末层/不满足剪枝条件时回退父类 OptSdpaAttention.forward，
        # 须禁用它构造 dense (bsz,1,L,L) fp32 加性掩码（thresh∈(0,1) 时），否则 math 核 O(L^2) OOM（16k≈16GiB）
        self._enable_sparse_mask = False

    def _prune_ok(self, hidden_states, is_cross_attention, past_key_value):
        # SMELL 3 modeling_opt_smell sdpa_prune ADD — 仅 full-seq causal self-attn 且非末层、sparse/pool 合法时走子集化
        if is_cross_attention or past_key_value is not None:
            return False
        num_layers = int(self.config.num_hidden_layers)
        if self.layer_idx is None or self.layer_idx == num_layers - 1:
            return False
        pool = int(getattr(self.config, "pool_size", 64))
        sparse = getattr(self.config, "sparse", None)
        if pool <= 0 or sparse is None:
            return False
        try:
            s = float(sparse)
        except (TypeError, ValueError):
            return False
        if not (0.0 < s <= 1.0):
            return False
        tgt_len = hidden_states.size(1)
        if tgt_len % pool != 0:
            return False
        return (tgt_len // pool) >= 2

    def _prune_block_scores(self, hidden_states, bsz, n_blocks, pool):
        # SMELL 3 modeling_opt_smell sdpa_prune ADD — 块分优先 predictor；形状不符/报错/显式 qk 则退化为 q·k 块均值；纯打分无梯度
        scorer = str(getattr(self.config, "sdpa_prune_scorer", "auto"))
        scored_by = "qk_block_mean"
        scores = None
        if scorer != "qk":
            try:
                hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
                predict_attn = self.predictor(hidden)
                if predict_attn.dim() == 3 and tuple(predict_attn.shape[-2:]) == (n_blocks, n_blocks):
                    # SMELL 3 modeling_opt_smell sdpa_prune ADD — 与 Jenga 同：对 query 块维求和，得每 KV 块一个标量
                    scores = predict_attn.sum(dim=-2)
                    scored_by = "predictor"
            except (RuntimeError, ValueError):
                scores = None
        if scores is None:
            # SMELL 3 modeling_opt_smell sdpa_prune ADD — predictor 不可用时以 q/k 投影的块均值点积打分（无梯度）
            hidden = hidden_states
            q = self.q_proj(hidden).view(bsz, n_blocks, pool, self.num_heads, self.head_dim)
            k = self.k_proj(hidden).view(bsz, n_blocks, pool, self.num_heads, self.head_dim)
            qb = q.mean(dim=2)
            kb = k.mean(dim=2)
            scores = torch.einsum("bihd,bjhd->bij", qb, kb).mean(dim=1)
            scored_by = "qk_block_mean"
        return scores.float(), scored_by

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        if not self._prune_ok(hidden_states, is_cross_attention, past_key_value):
            # SMELL 3 modeling_opt_smell sdpa_prune ADD — 不满足条件（末层/cross/past/长度不整除）回退父类 dense SDPA
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        pool = int(self.config.pool_size)
        sparse = float(self.config.sparse)
        num_layers = int(self.config.num_hidden_layers)
        n_blocks = tgt_len // pool
        with torch.no_grad():
            scores, scored_by = self._prune_block_scores(hidden_states, bsz, n_blocks, pool)
            # SMELL 3 modeling_opt_smell sdpa_prune ADD — 层规则与 Jenga 一致：下半层全留、上半层按 sparse 剪
            if self.layer_idx < num_layers // 2 - 1:
                q_len_blocks = n_blocks
            else:
                q_len_blocks = max(1, min(int(n_blocks * sparse), n_blocks))
            _, idx = torch.topk(scores, q_len_blocks, largest=True, dim=-1)
            idx = idx.sort().values  # (bsz, q_len_blocks) KV 块索引，升序保持原始顺序
            base = torch.arange(pool, device=idx.device).view(1, 1, pool)
            expanded_idx = (idx[..., None] * pool + base).reshape(bsz, -1)  # (bsz, q_len_blocks*pool)
        keep_len = expanded_idx.size(1)

        # SMELL 3 modeling_opt_smell sdpa_prune ADD — 进入 q/k/v 前直接子集化 hidden_states（bsz==1 与 Jenga 的 hidden_states[:, idx, :] 完全一致）
        if bsz == 1:
            hidden_states = hidden_states[:, expanded_idx.view(-1), :]
        else:
            hidden_states = torch.gather(
                hidden_states, 1, expanded_idx[..., None].expand(-1, -1, hidden_states.size(-1))
            )

        query_states = self._shape(self.q_proj(hidden_states), keep_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), keep_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), keep_len, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, keep_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # SMELL 3 modeling_opt_smell sdpa_prune ADD — 子集输出 scatter 回全长 L（层残差/add 需 full-seq）；bsz==1 与 Jenga 一致
        output = torch.zeros((bsz, tgt_len, self.embed_dim), device=attn_output.device, dtype=attn_output.dtype)
        if bsz == 1:
            output[0].scatter_(0, expanded_idx.view(-1).unsqueeze(1).expand(-1, self.embed_dim), attn_output[0])
        else:
            output = output.scatter(1, expanded_idx[..., None].expand(-1, -1, self.embed_dim), attn_output)

        self.last_prune_stats = {
            "layer_idx": int(self.layer_idx),
            "scored_by": scored_by,
            "pool": pool,
            "n_blocks": int(n_blocks),
            "kept_blocks": int(q_len_blocks),
            "kept_tokens": int(keep_len),
            "full_tokens": int(tgt_len),
            "keep_frac": float(keep_len) / float(tgt_len),
        }
        return output, None, past_key_value


OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,
    "flash_attention_2": OptFlashAttention2,
    "sdpa": OptSdpaAttention,
    "sdpa_gather": OptSdpaSparseGatherAttention,  # SMELL 3 modeling_opt_smell sparse_gather ADD — 实验用 gather 式块稀疏入口
    "sdpa_prune": OptSdpaPruneAttention,  # SMELL 3 modeling_opt_smell sdpa_prune ADD — Jenga 真语义 token 子集化入口
}


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size

        # SMELL 3 modeling_opt_smell sparse_gather ADD — config.sparse_gather=True 时用 gather 式块稀疏（attn_implementation 仍为 sdpa）
        # SMELL 3 modeling_opt_smell sdpa_prune ADD — config.sparse_prune=True 时用 Jenga 真语义 token 子集化（attn_implementation 仍为 sdpa）
        attn_cls = OPT_ATTENTION_CLASSES[config.attn_implementation]
        if config.attn_implementation == "sdpa" and getattr(config, "sparse_prune", False):
            attn_cls = OptSdpaPruneAttention
        elif getattr(config, "sparse_gather", False) and config.attn_implementation == "sdpa":
            attn_cls = OptSdpaSparseGatherAttention
        self.self_attn = attn_cls(config=config, is_decoder=True ,layer_idx=layer_idx)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)
        self.ffn_dim = config.ffn_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        seq_len = hidden_states.shape[1]
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = torch.utils.checkpoint.checkpoint(self.self_attn_layer_norm,hidden_states)
            # hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = torch.utils.checkpoint.checkpoint(self.final_layer_norm,hidden_states)
            # hidden_states = self.final_layer_norm(hidden_states)
        
        hidden_states = self.fc1(hidden_states)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = torch.utils.checkpoint.checkpoint(self.final_layer_norm,hidden_states)#hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True  # SMELL 3 modeling_opt_smell sdpa ADD — 声明支持 config.attn_implementation="sdpa"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config.attn_implementation == "flash_attention_2"
        # SMELL 3 modeling_opt_smell sdpa_prune ADD — sdpa/sdpa_gather/sdpa_prune 均只传 2D mask，避免 4D causal mask
        self._use_sdpa = str(config.attn_implementation).startswith("sdpa")

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2 or self._use_sdpa:
            # 2d mask is passed through the layers
            # SMELL 3 modeling_opt_smell sdpa ADD — sdpa 与 flash 一样只传 2d mask，避免构造 O(seq^2) 的 4d causal mask
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        def forward_fn_chunk(x_chunk, lm_head, label_chunk):
            """
            针对输入的一部分做 forward
            """
            logits = lm_head(x_chunk)
            return nn.functional.cross_entropy(logits, label_chunk, reduction='sum')
        
        if labels is not None:
            loss = 0.0 
            N = hidden_states.shape[1]
            hidden_states = hidden_states.view(N,-1)
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            for start in range(0, N-1, 2048):
                end = min(N-1, start + 2048)
                x_chunk = hidden_states[start:end]
                label_chunk = shift_labels[start:end]
                loss_chunk = torch.utils.checkpoint.checkpoint(forward_fn_chunk, x_chunk, self.lm_head, label_chunk)
                loss += loss_chunk 
            loss /= N
        # logits = self.lm_head(outputs[0]).contiguous()

        # loss = None
        # if labels is not None:
        #     # move labels to correct device to enable model parallelism
        #     labels = labels.to(logits.device)
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning_once(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index).to(logits.device)
            end_positions = end_positions.clamp(0, ignored_index).to(logits.device)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value
