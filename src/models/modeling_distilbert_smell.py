# SMELL 3 modeling_distilbert_smell NEW — DistilBERT fp32 SDPA 注意力（免复制 __class__ patch）+ 分类加载入口

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


def load_distilbert_classifier(model_dir, num_labels=11, dtype="fp32", attn="sdpa",
                               seq_len=0, pos_checkpoint=None, pos_mode="interpolate"):
    from transformers import DistilBertForSequenceClassification

    from src.models.position_embed_bert import ensure_positions_bert

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype not in dtype_map:
        raise ValueError(f"unknown dtype '{dtype}', expected one of {tuple(dtype_map)}")
    if attn not in ("sdpa", "eager"):
        raise ValueError(f"unknown attn '{attn}', expected 'sdpa' or 'eager'")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model = model.to(dtype=dtype_map[dtype])
    if attn == "sdpa":
        model = patch_distilbert_sdpa(model)
    if int(seq_len) > 0:
        model = ensure_positions_bert(model, int(seq_len), mode=pos_mode)
    if pos_checkpoint:
        model = _load_pos_checkpoint(model, pos_checkpoint, pos_mode)
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

    return results


if __name__ == "__main__":
    results = _run_selftest()
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("PASS" if all(results.values()) else "FAIL")
    raise SystemExit(0 if all(results.values()) else 1)
