# SMELL 3 position_embed NEW — OPT 位置嵌入扩展：Jenga 复制缩放 / 纯重复 / 线性插值

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
JENGA_SRC = REPO / "third_party" / "Jenga" / "src"
if str(JENGA_SRC) not in sys.path:
    sys.path.insert(0, str(JENGA_SRC))

from jenga.models.modeling_opt import OPTLearnedPositionalEmbedding  # noqa: E402

POSITION_MODES = ("jenga_dup_scaled", "duplicate", "interpolate")


def _find_decoder(model):
    candidates = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        candidates.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            candidates.append(inner)
    for candidate in candidates:
        decoder = getattr(getattr(candidate, "model", None), "decoder", None)
        if decoder is not None and hasattr(decoder, "embed_positions"):
            return decoder
    raise AttributeError(f"cannot locate embed_positions on {type(model).__name__}")


def get_model_max_position(model):
    embed_positions = _find_decoder(model).embed_positions
    offset = int(getattr(embed_positions, "offset", 0))
    return int(embed_positions.num_embeddings) - offset


def _dup_positions(weight, model_max_length, scale):
    max_positions = model_max_length - 2
    orig_n = weight.size(0) - 2
    assert (max_positions + 2) % orig_n == 0, (
        f"model_max_length={model_max_length} not divisible by original table size {orig_n}"
    )
    repeats = (max_positions + 2) // orig_n
    if scale:
        blocks = [weight[:-2] * i for i in range(1, repeats + 1)]
    else:
        blocks = [weight[:-2]] * repeats
    duplicated = torch.cat(blocks + [weight[-2:]], dim=0)
    assert duplicated.size(0) == model_max_length + 2
    return duplicated


def extend_opt_positions(model, model_max_length, mode="jenga_dup_scaled"):
    if mode not in POSITION_MODES:
        raise ValueError(f"unknown position mode '{mode}', expected one of {POSITION_MODES}")
    model_max_length = int(model_max_length)
    decoder = _find_decoder(model)
    old_emb = decoder.embed_positions
    old_weight = old_emb.weight.data
    if mode == "interpolate":
        body = old_weight[:-2].float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(body, size=(model_max_length, old_weight.size(1)),
                                mode="bilinear", align_corners=False)
        new_weight = torch.cat(
            [resized.squeeze(0).squeeze(0).to(old_weight.dtype), old_weight[-2:]], dim=0)
    else:
        new_weight = _dup_positions(old_weight, model_max_length, scale=(mode == "jenga_dup_scaled"))
    new_emb = OPTLearnedPositionalEmbedding(model_max_length, old_emb.embedding_dim)
    new_emb = new_emb.to(device=old_weight.device)
    new_emb.weight.data = new_weight
    new_emb.weight.requires_grad_(bool(old_emb.weight.requires_grad))
    decoder.embed_positions = new_emb
    return model


def ensure_positions(model, seq_len, mode="jenga_dup_scaled"):
    seq_len = int(seq_len)
    capacity = get_model_max_position(model)
    if capacity >= seq_len:
        print(f"[positions] mode={mode} capacity={capacity} -> seq_len={seq_len} (already ok)")
        return model
    model = extend_opt_positions(model, seq_len, mode=mode)
    print(f"[positions] mode={mode} capacity={capacity} -> seq_len={seq_len} "
          f"(extended to {get_model_max_position(model)})")
    return model
