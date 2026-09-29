# SMELL 3 position_embed_bert NEW — DistilBERT 位置嵌入扩展：线性插值 / 块重复 / 块放大（offset=0）

import torch
import torch.nn.functional as F

POSITION_MODES_BERT = ("interpolate", "duplicate", "dup_scaled")


def _find_embeddings(model):
    """定位 DistilBertEmbeddings，兼容 DistilBertModel / ForSequenceClassification / PEFT 包装。"""
    stack = [model]
    visited = set()
    while stack:
        module = stack.pop()
        if id(module) in visited:
            continue
        visited.add(id(module))
        if hasattr(module, "position_embeddings") and hasattr(module, "word_embeddings"):
            return module
        for name in ("distilbert", "base_model", "model", "module", "embeddings"):
            child = getattr(module, name, None)
            if child is not None:
                stack.append(child)
    raise AttributeError(f"cannot locate distilbert position embeddings on {type(model).__name__}")


def get_model_max_position_bert(model):
    """BERT 无 OPT 的 offset=+2 特殊行，位置表容量即 num_embeddings。"""
    return int(_find_embeddings(model).position_embeddings.num_embeddings)


def _interpolate_positions(weight, model_max_length):
    body = weight.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(body, size=(int(model_max_length), weight.size(1)),
                            mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).to(weight.dtype)


def _dup_positions_bert(weight, model_max_length, scale):
    orig_n = int(weight.size(0))
    assert int(model_max_length) % orig_n == 0, (
        f"model_max_length={model_max_length} not divisible by original table size {orig_n}")
    repeats = int(model_max_length) // orig_n
    if scale:
        blocks = [weight * i for i in range(1, repeats + 1)]
    else:
        blocks = [weight] * repeats
    duplicated = torch.cat(blocks, dim=0)
    assert duplicated.size(0) == int(model_max_length), (
        f"duplicated table has {duplicated.size(0)} rows, expected {model_max_length}")
    return duplicated


def extend_bert_positions(model, model_max_length, mode="interpolate"):
    if mode not in POSITION_MODES_BERT:
        raise ValueError(f"unknown position mode '{mode}', expected one of {POSITION_MODES_BERT}")
    model_max_length = int(model_max_length)
    embeddings = _find_embeddings(model)
    old_emb = embeddings.position_embeddings
    old_weight = old_emb.weight.data
    if model_max_length == int(old_weight.size(0)):
        return model
    if mode == "interpolate":
        new_weight = _interpolate_positions(old_weight, model_max_length)
    else:
        new_weight = _dup_positions_bert(old_weight, model_max_length, scale=(mode == "dup_scaled"))
    new_emb = torch.nn.Embedding(model_max_length, old_emb.embedding_dim)
    new_emb = new_emb.to(device=old_weight.device, dtype=old_weight.dtype)
    new_emb.weight.data = new_weight
    new_emb.weight.requires_grad_(bool(old_emb.weight.requires_grad))
    embeddings.position_embeddings = new_emb
    # SMELL 3 position_embed_bert position_ids FIXED — Embeddings 缓存了 max_position_embeddings 长的 position_ids buffer，必须同步扩展
    if "position_ids" in dict(embeddings.named_buffers()):
        embeddings.position_ids = torch.arange(model_max_length, device=old_weight.device).expand((1, -1))
    config = getattr(model, "config", None)
    if config is not None:
        config.max_position_embeddings = model_max_length
    return model


def ensure_positions_bert(model, seq_len, mode="interpolate"):
    seq_len = int(seq_len)
    capacity = get_model_max_position_bert(model)
    if capacity >= seq_len:
        print(f"[positions-bert] mode={mode} capacity={capacity} -> seq_len={seq_len} (already ok)")
        return model
    model = extend_bert_positions(model, seq_len, mode=mode)
    print(f"[positions-bert] mode={mode} capacity={capacity} -> seq_len={seq_len} "
          f"(extended to {get_model_max_position_bert(model)})")
    return model
