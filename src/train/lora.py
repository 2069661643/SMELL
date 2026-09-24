# SMELL 3 lora NEW — LoRA 工厂与可训练参数状态字典一致性工具

import torch
from peft import LoraConfig, TaskType, get_peft_model


def build_lora_model(
    model,
    r=8,
    targets=("q_proj", "k_proj", "v_proj", "out_proj"),
    lora_alpha=16,
    dropout=0.0,
):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=list(targets),
        bias="none",
    )
    return get_peft_model(model, config)


def count_trainable(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# SMELL 3 lora rank_rotation BEGIN — LoRA 秩调度：从模型取秩数并按轮次生成激活秩集合
def num_lora_ranks(model) -> int:
    """从任一 `lora_A` 参数取秩（shape[0]）；无 LoRA 参数时报错。"""
    for name, param in model.named_parameters():
        if "lora_A" in name:
            return int(param.shape[0])
    raise ValueError("no lora_A parameter found; is the model LoRA-wrapped?")


def active_ranks_for(round_idx, rank_k, num_ranks, mode):
    """返回本轮激活秩的升序列表；mode=all 返回 None（保持旧的全参数行为）。"""
    rank_k = int(rank_k)
    num_ranks = int(num_ranks)
    assert rank_k > 0, f"rank_k must be > 0, got {rank_k}"
    assert num_ranks > 0, f"num_ranks must be > 0, got {num_ranks}"
    if mode == "all":
        return None
    assert mode in ("rotate", "lock"), f"unknown rank mode: {mode}"
    assert rank_k <= num_ranks, f"rank_k={rank_k} exceeds num_ranks={num_ranks}"
    if mode == "lock":
        ranks = range(rank_k)
    else:
        base = int(round_idx) * rank_k
        ranks = ((base + j) % num_ranks for j in range(rank_k))
    return sorted(int(rank) for rank in ranks)
# SMELL 3 lora rank_rotation END


def get_trainable_state_dict(model) -> dict:
    return {
        name: param.detach().to("cpu").clone()
        for name, param in sorted(model.named_parameters())
        if param.requires_grad
    }


def load_trainable_state_dict(model, sd) -> None:
    names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert set(sd.keys()) == names, (
        "trainable state_dict keys mismatch: "
        f"missing={sorted(names - set(sd.keys()))[:5]} "
        f"extra={sorted(set(sd.keys()) - names)[:5]}"
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(sd[name].to(device=param.device, dtype=param.dtype))
