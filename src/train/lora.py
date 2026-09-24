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
