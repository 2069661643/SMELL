# SMELL 3 modeling_distilbert_smell NEW — DistilBERT fp32 SDPA 注意力（免复制 __class__ patch）+ 分类加载入口

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers.models.distilbert import modeling_distilbert as _hf_distilbert

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HF_ATTN_BASE = getattr(_hf_distilbert, "DistilBertAttention", _hf_distilbert.MultiHeadSelfAttention)


def _sdpa_additive_mask(mask, dtype, bsz, k_length):
    """HF 4.45.2 的 DistilBERT 传 1=keep/0=pad 的 2D mask；兼容 additive 2D/3D/4D。"""
    if mask is None:
        return None
    if mask.dim() == 2:
        reshaped = mask.reshape(bsz, 1, 1, k_length).to(dtype)
        if bool(reshaped.min() >= 0):
            return (1.0 - reshaped) * torch.finfo(dtype).min
        return reshaped
    return mask.to(dtype)


class DistilBertSdpaAttention(HF_ATTN_BASE):
    """与 HF eager 注意力同权重/同布局，仅把 softmax(qk)V 换成 F.scaled_dot_product_attention。"""

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False, **kwargs):
        bsz, q_length, _ = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads

        def shape(x):
            return x.view(bsz, -1, self.n_heads, dim_per_head).transpose(1, 2)

        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))

        attn_mask = _sdpa_additive_mask(mask, q.dtype, bsz, k_length)
        dropout_p = self.dropout.p if self.training else 0.0
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)

        # SMELL 3 modeling_distilbert_smell head_mask ADD — SDPA 无法后乘 attn 权重；按头置零 context 与 eager 数学等价
        if head_mask is not None:
            context = context * head_mask.reshape(-1, self.n_heads, 1, 1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(bsz, q_length, self.n_heads * dim_per_head)
        context = self.out_lin(context)
        if output_attentions:
            # SMELL 3 modeling_distilbert_smell output_attentions ADD — SDPA 不返回 attn 权重，返回 None 占位（调用方 tuple 解包保持成立）
            return (context, None)
        return (context,)


def _find_distilbert_core(model):
    candidates = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        candidates.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            candidates.append(inner)
    for candidate in candidates:
        core = getattr(candidate, "distilbert", None)
        if core is not None and hasattr(core, "transformer"):
            return core
        if hasattr(candidate, "transformer") and hasattr(candidate, "embeddings"):
            return candidate
    raise AttributeError(f"cannot locate DistilBertModel on {type(model).__name__}")


def patch_distilbert_sdpa(model):
    core = _find_distilbert_core(model)
    patched = 0
    for layer in core.transformer.layer:
        if layer.attention.__class__ is not DistilBertSdpaAttention:
            layer.attention.__class__ = DistilBertSdpaAttention
            patched += 1
    print(f"[sdpa-bert] patched {patched} attention layers (idempotent)")
    return model


# SMELL 3 modeling_distilbert_smell sdpa_prune BEGIN — Jenga 真语义「token 子集化」稀疏注意力（双向、无 causal）
def _keep_mask_2d(mask, bsz, q_len, device):
    """归一 HF 4.45.2 的 1=keep/0=pad mask 为 (b, L) float 0/1；None = 全 1（无 padding）。

    该版 DistilBERT 只在「mask 含 0」时下传 2D keep-mask，否则给 None；additive 形态直接报错，
    避免静默按错误语义子集化。
    """
    if mask is None:
        return torch.ones(bsz, q_len, device=device, dtype=torch.float32)
    if mask.dim() != 2 or tuple(mask.shape) != (bsz, q_len):
        raise ValueError(f"unsupported attention mask shape {tuple(mask.shape)} for prune path "
                         f"(expected ({bsz}, {q_len}) keep-mask)")
    return mask.to(device=device, dtype=torch.float32)


def _predictor_dims(predictor):
    if predictor is None:
        return None
    return {
        "q1_outdim": int(predictor.qlinear1.out_features),
        "q2_outdim": int(predictor.qlinear2.out_features),
        "k1_outdim": int(predictor.klinear1.out_features),
        "k2_outdim": int(predictor.klinear2.out_features),
    }


def _predictor_needs_rebuild(attn, layer_idx, predictor_layers):
    if getattr(attn, "predictor", None) is None:
        return True
    if predictor_layers is None or layer_idx >= len(predictor_layers):
        return False
    entry = predictor_layers[layer_idx]
    if not entry:
        return False
    wanted = {key: int(value) for key, value in dict(entry).items() if value is not None}
    return _predictor_dims(attn.predictor) != wanted


class DistilBertSdpaPruneAttention(DistilBertSdpaAttention):
    """Jenga token 子集化剪枝的 DistilBERT 版（无 causal、batch-safe、含 padding mask）。

    语义对齐 `src/models/modeling_opt_smell.py` 的 OptSdpaPruneAttention：
    层规则 = 下半层（layer_idx < num_layers//2 - 1）全保留、上半层保留 int(n_blk*sparse) 块、末层豁免；
    块分优先 predictor(hidden)->(b,n_blk,n_blk) 对 query 块维求和，CATV 共识掩码（±inf）加在 topk 之前，
    vote_callback 上报原始块分；选中块展开为 token 索引后 *直接子集化 hidden* 再算双向 SDPA，输出 scatter 回全长。
    与 OPT 的差异：双向（is_causal=False）、被 gather 的 padding 位置按 keep-mask 屏蔽（含整块 padding 不参与 topk）。
    """

    # SMELL 3 modeling_distilbert_smell sdpa_prune FIXED — __class__ 换类不跑 __init__，用类级默认保证属性存在
    # （predictor 不能给类级默认：nn.Module 的正常属性查找会命中它、遮蔽 _modules 注册的真实 predictor）
    layer_idx = None
    last_prune_stats = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = None
        # SMELL 3 modeling_distilbert_smell sdpa_prune ADD — predictor 由 patch_distilbert_sdpa_prune 按 pruned_config 重建
        self.last_prune_stats = None

    @property
    def predictor(self):
        return self.__dict__.get("_modules", {}).get("predictor")

    def _prune_ok(self, q_len):
        if self.layer_idx is None or self.predictor is None:
            return False
        num_layers = int(getattr(self.config, "n_layers", 0)) or int(
            getattr(self.config, "num_hidden_layers", 0))
        if num_layers <= 0:
            return False
        if self.layer_idx == num_layers - 1:
            return False
        if self.layer_idx < num_layers // 2 - 1:
            return False
        pool = int(getattr(self.config, "pool_size", 64))
        sparse = getattr(self.config, "sparse", None)
        if pool <= 0 or sparse is None:
            return False
        try:
            s = float(sparse)
        except (TypeError, ValueError):
            return False
        # SMELL 3 modeling_distilbert_smell sdpa_prune FIXED — s>=1 视为 dense（回退父类），避免「全选块」仍走子集路径
        if not (0.0 < s < 1.0):
            return False
        if q_len % pool != 0 or (q_len // pool) < 2:
            return False
        return True

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False, **kwargs):
        bsz, q_len, _ = query.size()
        if not self._prune_ok(q_len):
            return super().forward(query, key, value, mask, head_mask=head_mask,
                                   output_attentions=output_attentions, **kwargs)

        pool = int(self.config.pool_size)
        sparse = float(self.config.sparse)
        n_blocks = q_len // pool
        dim_per_head = self.dim // self.n_heads
        keep_mask = _keep_mask_2d(mask, bsz, q_len, query.device)

        with torch.no_grad():
            scores = self.predictor(query)
            if scores.dim() != 3 or tuple(scores.shape) != (bsz, n_blocks, n_blocks):
                raise RuntimeError(
                    f"predictor output {tuple(scores.shape)} != ({bsz}, {n_blocks}, {n_blocks})")
            scores = scores.float().sum(dim=-2)
            block_keep = keep_mask.view(bsz, n_blocks, pool).amax(dim=-1)
            pad_block = block_keep < 0.5
            # SMELL 3 modeling_distilbert_smell sdpa_prune ADD — 全 padding 块置 -inf，不占 topk 预算
            scores = scores.masked_fill(pad_block, torch.finfo(scores.dtype).min)
            # SMELL 3 modeling_distilbert_smell vote_valid_mean FIXED — 投票载荷以 NaN 标注 pad 块（有效均值口径，P0a）；
            # 旧版把 -3.4e38 占位符直接送进投票 → 掩码恒为 {0..50}/{205..255}（ailog 261014-111800 §1）
            vote_callback = getattr(self.config, "vote_callback", None)
            if vote_callback is not None:
                vote_payload = scores.masked_fill(pad_block, float("nan"))
                vote_callback(self.layer_idx, vote_payload.detach())
            consensus_mask = getattr(self.config, "consensus_mask", None)
            if consensus_mask is not None and self.layer_idx in consensus_mask:
                scores = scores + consensus_mask[self.layer_idx].to(
                    device=scores.device, dtype=scores.dtype)
            keep_blocks = max(1, min(int(n_blocks * sparse), n_blocks))
            _, idx = torch.topk(scores, keep_blocks, largest=True, dim=-1)
            idx = idx.sort(dim=-1).values
            # SMELL 3 modeling_distilbert_smell keep_block0 ADD — BERT 特供（D2，ailog 261014-111800 §4.2）：
            # block0 载 [CLS]（唯一 readout），所有剪枝层强制保留；挤位规则 = 丢已入选块中分数最低者（block0 自身不可丢）；
            # 位置在 consensus_mask + topk 之后（掩码把 block0 置 -inf 也会被救回）；keep 预算不变。
            block0_forced = 0
            if bool(getattr(self.config, "keep_block0", True)):
                miss = ~(idx == 0).any(dim=-1)
                if bool(miss.any()):
                    sel = scores.gather(1, idx).masked_fill(idx == 0, float("inf"))
                    idx = idx.clone()
                    idx[miss, sel[miss].argmin(dim=-1)] = 0
                    idx = idx.sort(dim=-1).values
                    block0_forced = int(miss.sum())
            base = torch.arange(pool, device=query.device).view(1, 1, pool)
            expanded = (idx[..., None] * pool + base).reshape(bsz, -1)

        hidden = torch.gather(query, 1, expanded[..., None].expand(-1, -1, query.size(-1)))
        kept_mask = torch.gather(keep_mask, 1, expanded)
        attn_mask = (1.0 - kept_mask).reshape(bsz, 1, 1, -1) * torch.finfo(query.dtype).min

        def shape(x):
            return x.view(bsz, -1, self.n_heads, dim_per_head).transpose(1, 2)

        q = shape(self.q_lin(hidden))
        k = shape(self.k_lin(hidden))
        v = shape(self.v_lin(hidden))
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask.to(query.dtype),
            dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
        if head_mask is not None:
            context = context * head_mask.reshape(-1, self.n_heads, 1, 1).to(context.dtype)
        context = context.transpose(1, 2).contiguous().view(bsz, expanded.size(1), self.dim)
        context = self.out_lin(context)

        output = torch.zeros((bsz, q_len, self.dim), device=context.device, dtype=context.dtype)
        output = output.scatter(1, expanded[..., None].expand(-1, -1, self.dim), context)
        self.last_prune_stats = {
            "layer_idx": int(self.layer_idx),
            "pool": pool,
            "n_blocks": int(n_blocks),
            "kept_blocks": int(keep_blocks),
            "kept_tokens": int(expanded.size(1)),
            "full_tokens": int(q_len),
            "keep_frac": float(expanded.size(1)) / float(q_len),
            # SMELL 3 modeling_distilbert_smell keep_block0 ADD — block0 诊断（D2 验收用）
            "block0_kept": bool((idx == 0).any(dim=-1).all()),
            "block0_forced": int(block0_forced),
        }
        if output_attentions:
            return (output, None)
        return (output,)


def patch_distilbert_sdpa_prune(model, predictor_layers=None):
    """把全部注意力层换成剪枝 SDPA 类并挂 predictor（形状按 predictor_layers 重建；None=默认形状）。"""
    from jenga.models.predictor import PrunableAttnPredictorInfer

    core = _find_distilbert_core(model)
    # SMELL 3 modeling_distilbert_smell keep_block0 ADD — BERT 特供默认开启（D2；写进 config 供 run config.json 复现）
    if getattr(core, "config", None) is not None:
        core.config.keep_block0 = bool(getattr(core.config, "keep_block0", True))
    patched = 0
    built = 0
    for layer_idx, layer in enumerate(core.transformer.layer):
        attn = layer.attention
        if attn.__class__ is not DistilBertSdpaPruneAttention:
            attn.__class__ = DistilBertSdpaPruneAttention
            patched += 1
        attn.layer_idx = int(layer_idx)
        if _predictor_needs_rebuild(attn, layer_idx, predictor_layers):
            entry = None
            if predictor_layers is not None and layer_idx < len(predictor_layers):
                entry = predictor_layers[layer_idx]
            kwargs = {key: int(value) for key, value in dict(entry).items() if value is not None} if entry else {}
            reference = attn.q_lin.weight
            predictor = PrunableAttnPredictorInfer(
                dim=int(attn.dim // attn.n_heads), hidden_dim=128, n_head=int(attn.n_heads), **kwargs)
            attn.predictor = predictor.to(device=reference.device, dtype=reference.dtype).eval()
            built += 1
    print(f"[sdpa-bert] prune-patched {patched} attention layers, built {built} predictors")
    return model


def block_attn_pool_bidir(query_heads, key_heads, keep_mask, block=64):
    """双向 ReLU(qk^T) 块池化目标（pytorch 版，替代 Jenga Triton 的 HEAD_DIM=128 硬编码）。

    query_heads/key_heads: (b, h, L, d)（q 已按 1/sqrt(d) 缩放，与 Jenga train-predictor 同）；
    keep_mask: (b, L) 1/0 或 None；返回 (b, n_blk, n_blk) float，= maxpool_{block} ReLU(qk^T) / block。
    """
    bsz, heads, n_ctx, _ = query_heads.shape
    if n_ctx % block != 0:
        raise ValueError(f"sequence length {n_ctx} must be divisible by {block}")
    n_blocks = n_ctx // block
    key_mask = None
    if keep_mask is not None:
        key_mask = keep_mask.to(device=query_heads.device, dtype=query_heads.dtype)
    out = torch.empty((bsz, n_blocks, n_blocks), dtype=query_heads.dtype, device=query_heads.device)
    with torch.no_grad():
        for row in range(n_blocks):
            q_block = query_heads[:, :, row * block:(row + 1) * block, :]
            attn = torch.matmul(q_block, key_heads.transpose(-1, -2)).float()
            attn.relu_()
            if key_mask is not None:
                attn = attn * key_mask[:, None, None, :]
            attn = attn.sum(dim=1)
            pooled = attn.view(bsz, block, n_blocks, block).amax(dim=(1, 3)) / float(block)
            out[:, row, :] = pooled.to(out.dtype)
    return out
# SMELL 3 modeling_distilbert_smell sdpa_prune END


def _load_pos_checkpoint(model, pos_checkpoint, pos_mode):
    from src.models.position_embed_bert import ensure_positions_bert

    payload = torch.load(pos_checkpoint, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("position_embeddings.weight", "embed_positions.weight", "weight"):
            if key in payload:
                payload = payload[key]
                break
    assert torch.is_tensor(payload), f"unsupported pos checkpoint payload: {type(payload)} at {pos_checkpoint}"
    embeddings = _find_distilbert_core(model).embeddings
    target = embeddings.position_embeddings.weight
    if tuple(payload.shape) != tuple(target.shape):
        same_width = payload.dim() == 2 and payload.shape[1] == target.shape[1]
        if same_width and payload.shape[0] > target.shape[0]:
            model = ensure_positions_bert(model, int(payload.shape[0]), mode=pos_mode)
            target = embeddings.position_embeddings.weight
    assert tuple(payload.shape) == tuple(target.shape), (
        f"pos checkpoint shape {tuple(payload.shape)} != position_embeddings {tuple(target.shape)}")
    with torch.no_grad():
        target.copy_(payload.to(device=target.device, dtype=target.dtype))
    print(f"[sdpa-bert] pos_checkpoint loaded path={pos_checkpoint} shape={tuple(target.shape)}")
    return model


def _load_head_checkpoint(model, head_checkpoint, prefix="smell_cls_head."):
    # SMELL 3 modeling_distilbert_smell head_checkpoint ADD — 载入 Step1 训练的分类头（pre_classifier/classifier 4 张量）
    payload = torch.load(head_checkpoint, map_location="cpu")
    assert isinstance(payload, dict), f"unsupported head checkpoint payload {type(payload).__name__}" \
                                      f" at {head_checkpoint}"
    state = {key[len(prefix):] if key.startswith(prefix) else key: value
             for key, value in payload.items() if torch.is_tensor(value)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    assert loaded > 0, f"no head tensors loaded from {head_checkpoint}: state={sorted(state)}"
    print(f"[sdpa-bert] head_checkpoint loaded tensors={loaded} missing={list(missing)} "
          f"unexpected={list(unexpected)} path={head_checkpoint}")
    return model


def load_distilbert_classifier(model_dir, num_labels=11, dtype="fp32", attn="sdpa",
                               seq_len=0, pos_checkpoint=None, pos_mode="interpolate",
                               head_checkpoint=None, predictor=None, pruned_config=None,
                               sparse=0.4, pool_size=64):
    from transformers import DistilBertForSequenceClassification

    from src.models.position_embed_bert import ensure_positions_bert

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype not in dtype_map:
        raise ValueError(f"unknown dtype '{dtype}', expected one of {tuple(dtype_map)}")
    if attn not in ("sdpa", "eager", "sdpa_prune"):
        raise ValueError(f"unknown attn '{attn}', expected 'sdpa', 'eager' or 'sdpa_prune'")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model = model.to(dtype=dtype_map[dtype])
    # SMELL 3 modeling_distilbert_smell prune_config FIXED — 剪枝注意力从 config 读 sparse/pool_size；不设会静默退化为 dense
    model.config.pool_size = int(pool_size)
    model.config.sparse = float(sparse)
    predictor_layers = None
    if pruned_config:
        payload = torch.load(pruned_config, map_location="cpu")
        assert isinstance(payload, dict) and "layers" in payload, \
            f"unsupported pruned_config payload from {pruned_config}"
        predictor_layers = payload["layers"]
    if attn == "sdpa":
        model = patch_distilbert_sdpa(model)
    elif attn == "sdpa_prune":
        model = patch_distilbert_sdpa_prune(model, predictor_layers=predictor_layers)
    if int(seq_len) > 0:
        model = ensure_positions_bert(model, int(seq_len), mode=pos_mode)
    if pos_checkpoint:
        model = _load_pos_checkpoint(model, pos_checkpoint, pos_mode)
    if head_checkpoint:
        model = _load_head_checkpoint(model, head_checkpoint)
    if predictor:
        # SMELL 3 modeling_distilbert_smell predictor ADD — 复用 run_fed 的加载/零填充/硬缺失校验
        from src.fed.run_fed import load_predictor_weights
        load_predictor_weights(model, predictor)
    # SMELL 3 modeling_distilbert_smell consensus_mask ADD — 挂到每层 attention 的 config（CATV 评测掩码注入点）
    core = _find_distilbert_core(model)
    base_config = core.config if hasattr(core, "config") else getattr(model, "config", None)
    if base_config is not None:
        base_config.consensus_mask = None
    return model


def _run_selftest():
    import tempfile

    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    from src.models.position_embed_bert import ensure_positions_bert, get_model_max_position_bert

    results = {}
    torch.manual_seed(0)
    config = DistilBertConfig(vocab_size=128, max_position_embeddings=64, n_layers=2, n_heads=4,
                              dim=64, hidden_dim=128, dropout=0.0, attention_dropout=0.0,
                              seq_classif_dropout=0.0, pad_token_id=0)
    model = DistilBertForSequenceClassification(config)
    model.eval()

    input_ids = torch.randint(1, config.vocab_size, (2, 32))
    input_ids[1, 20:] = 0
    attention_mask = (input_ids != 0).long()
    with torch.no_grad():
        eager = model(input_ids=input_ids, attention_mask=attention_mask).logits

    patch_distilbert_sdpa(model)
    patch_distilbert_sdpa(model)
    with torch.no_grad():
        sdpa = model(input_ids=input_ids, attention_mask=attention_mask).logits
    results["sdpa vs eager logits (padding)"] = bool(torch.allclose(eager, sdpa, atol=1e-5, rtol=1e-4))
    results["all layers patched"] = all(
        layer.attention.__class__ is DistilBertSdpaAttention
        for layer in model.distilbert.transformer.layer)

    with torch.no_grad():
        attention_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_attentions=True)
    results["output_attentions None placeholder"] = bool(
        attention_outputs.attentions is not None
        and all(item is None for item in attention_outputs.attentions))

    capacity_before = get_model_max_position_bert(model)
    ensure_positions_bert(model, 128, mode="interpolate")
    results["interpolate 64->128"] = bool(
        capacity_before == 64 and get_model_max_position_bert(model) == 128)
    with torch.no_grad():
        extended_logits = model(input_ids=torch.randint(1, config.vocab_size, (1, 100)),
                                attention_mask=torch.ones(1, 100, dtype=torch.long)).logits
    results["forward at extended length"] = bool(
        tuple(extended_logits.shape) == (1, config.num_labels) and torch.isfinite(extended_logits).all())

    ensure_positions_bert(model, 256, mode="duplicate")
    weight_dup = model.distilbert.embeddings.position_embeddings.weight
    results["duplicate 128->256 exact"] = bool(
        get_model_max_position_bert(model) == 256
        and torch.equal(weight_dup[128:256], weight_dup[0:128]))
    ensure_positions_bert(model, 512, mode="dup_scaled")
    weight_scaled = model.distilbert.embeddings.position_embeddings.weight
    results["dup_scaled 256->512 ramp"] = bool(
        get_model_max_position_bert(model) == 512
        and torch.equal(weight_scaled[0:256], weight_dup)
        and torch.allclose(weight_scaled[256:512], 2.0 * weight_dup))

    checkpoint = {"position_embeddings.weight": torch.randn(1024, config.dim)}
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(checkpoint, handle.name)
        model = _load_pos_checkpoint(model, handle.name, pos_mode="interpolate")
    results["pos_checkpoint extends+loads"] = bool(
        get_model_max_position_bert(model) == 1024
        and torch.allclose(model.distilbert.embeddings.position_embeddings.weight,
                           checkpoint["position_embeddings.weight"]))

    checkpoint_alt = {"embed_positions.weight": torch.randn(1024, config.dim)}
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(checkpoint_alt, handle.name)
        model = _load_pos_checkpoint(model, handle.name, pos_mode="interpolate")
    results["pos_checkpoint alt key loads"] = bool(
        torch.allclose(model.distilbert.embeddings.position_embeddings.weight,
                       checkpoint_alt["embed_positions.weight"]))

    # SMELL 3 modeling_distilbert_smell sdpa_prune_selftest ADD — 剪枝路径：层规则/keep 比例/掩码一致/vote/共识掩码
    prune_config = DistilBertConfig(vocab_size=128, max_position_embeddings=128, n_layers=4,
                                    n_heads=4, dim=64, hidden_dim=128, dropout=0.0,
                                    attention_dropout=0.0, seq_classif_dropout=0.0, pad_token_id=0)
    prune_config.pool_size = 64
    prune_config.sparse = 0.5
    torch.manual_seed(3)
    prune_model = DistilBertForSequenceClassification(prune_config)
    prune_model.eval()
    dense_ref = copy.deepcopy(prune_model)
    patch_distilbert_sdpa(dense_ref)
    patch_distilbert_sdpa_prune(prune_model)
    seq_ids = torch.randint(1, prune_config.vocab_size, (2, 128))
    seq_ids[1, 70:] = 0
    seq_mask = (seq_ids != 0).long()
    with torch.no_grad():
        prune_logits = prune_model(input_ids=seq_ids, attention_mask=seq_mask).logits
        dense_logits = dense_ref(input_ids=seq_ids, attention_mask=seq_mask).logits
    results["prune logits finite"] = bool(torch.isfinite(prune_logits).all())
    stats = {index: layer.attention.last_prune_stats for index, layer in
             enumerate(prune_model.distilbert.transformer.layer)}
    results["prune layer rule"] = bool(
        stats[0] is None and stats[3] is None
        and stats[1]["keep_frac"] == 0.5 and stats[2]["keep_frac"] == 0.5)
    results["prune kept blocks"] = bool(
        stats[1]["kept_blocks"] == 1 and stats[1]["n_blocks"] == 2 and stats[1]["kept_tokens"] == 64)
    votes = {}
    prune_config.vote_callback = lambda layer_idx, scores: votes.__setitem__(int(layer_idx), scores)
    with torch.no_grad():
        prune_model(input_ids=seq_ids, attention_mask=seq_mask)
    prune_config.vote_callback = None
    results["prune vote_callback layers"] = sorted(votes) == [1, 2]
    results["prune vote_callback shape"] = bool(
        tuple(votes[1].shape) == (2, 2) and votes[1].dtype == torch.float32)

    # SMELL 3 modeling_distilbert_smell vote_valid_mean ADD — P0a：pad 块以 NaN 上报（有效均值口径），有效块保持有限
    short_ids = torch.randint(1, prune_config.vocab_size, (1, 128))
    short_ids[0, 40:] = 0
    short_mask = (short_ids != 0).long()
    votes_short = {}
    prune_config.vote_callback = lambda layer_idx, payload: votes_short.__setitem__(int(layer_idx), payload)
    with torch.no_grad():
        prune_model(input_ids=short_ids, attention_mask=short_mask)
    prune_config.vote_callback = None
    payload = votes_short[1][0]
    results["vote payload NaN on pad blocks"] = bool(
        torch.isfinite(payload[0]) and torch.isnan(payload[1]))

    # SMELL 3 modeling_distilbert_smell keep_block0 ADD — D2：-inf 掩码下 block0 仍被保留；关闭开关时不救
    prune_config.consensus_mask = {1: torch.tensor([float("-inf"), 0.0]),
                                   2: torch.tensor([float("-inf"), 0.0])}
    with torch.no_grad():
        prune_model(input_ids=seq_ids, attention_mask=seq_mask)
    stats_layer1 = prune_model.distilbert.transformer.layer[1].attention.last_prune_stats
    results["block0 forced under -inf mask"] = bool(
        stats_layer1["block0_kept"] and stats_layer1["block0_forced"] >= 1)
    prune_config.keep_block0 = False
    with torch.no_grad():
        prune_model(input_ids=seq_ids, attention_mask=seq_mask)
    stats_layer1 = prune_model.distilbert.transformer.layer[1].attention.last_prune_stats
    results["block0 not forced when disabled"] = bool(stats_layer1["block0_kept"] is False)
    prune_config.keep_block0 = True
    prune_config.consensus_mask = None

    prune_config.sparse = 1.0
    with torch.no_grad():
        dense_equiv = prune_model(input_ids=seq_ids, attention_mask=seq_mask).logits
    results["sparse=1.0 equals dense"] = bool(torch.allclose(dense_equiv, dense_logits, atol=1e-5, rtol=1e-4))
    prune_config.sparse = 0.5

    # 共识掩码：layer1 强制保留块 1、排除块 0；无效块分值 -inf 时 topk 只剩块 1
    prune_config.consensus_mask = {
        1: torch.tensor([float("-inf"), float("inf")]),
        2: torch.tensor([float("-inf"), float("inf")]),
    }
    with torch.no_grad():
        prune_model(input_ids=seq_ids, attention_mask=seq_mask)
    forced = prune_model.distilbert.transformer.layer[1].attention.last_prune_stats
    results["consensus mask forces anchor"] = bool(forced["kept_blocks"] == 1)
    prune_config.consensus_mask = None

    return results


if __name__ == "__main__":
    results = _run_selftest()
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("PASS" if all(results.values()) else "FAIL")
    raise SystemExit(0 if all(results.values()) else 1)
