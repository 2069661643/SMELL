# SMELL 3 lowrank_reparam NEW — 想法2：在 r=1 LoRA 之上叠加固定随机正交基的低秩可训练核 Z
# 对应 FwdLLM+/SubZero/LoRA-XS 路线：把可训练维数从 LoRA 的 (out+in)·r 压到每模块 k²（共 96·k²）。
# ΔW_eff = (alpha/r) B A + U Z V^T，U(out×k)/V(in×k) 为固定高斯 QR 正交基，Z(k×k) 为唯一可训练量。

import torch
from torch import nn


class _LowrankReparam(nn.Module):
    """固定正交基 U/V + 可训练核 Z 的子模块，forward(x) 返回 U Z Vᵀ 对输出的增量。

    extra = x @ V @ Zᵀ @ Uᵀ（float32 计算，避免 bf16 下 ZO 小扰动被量化）。
    """

    def __init__(self, in_features, out_features, k, u, v):
        super().__init__()
        self.k = int(k)
        self.register_buffer("U", u)
        self.register_buffer("V", v)
        self.Z = nn.Parameter(torch.zeros(self.k, self.k, dtype=torch.float32))

    def forward(self, x):
        xf = x.float()
        extra = (xf @ self.V.float()) @ self.Z.t() @ self.U.float().t()
        return extra


def _orthogonal(rows, cols, generator, dtype):
    """返回 rows×cols 的列正交矩阵（对高斯阵做 reduced QR）。"""
    assert cols <= rows, f"cannot build {cols} orthonormal columns in {rows} dims"
    gaussian = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(gaussian)
    return basis.to(dtype)


# SMELL 3 lowrank_reparam hooks BEGIN — 在 PEFT LoRA 层输出上叠加低秩核贡献
def _lowrank_forward_hook(module, inputs, output):
    x = inputs[0]
    extra = None
    for adapter in module._lowrank_keys:
        contribution = module._lowrank[adapter](x)
        extra = contribution if extra is None else extra + contribution
    return output + extra.to(output.dtype)
# SMELL 3 lowrank_reparam hooks END


def attach_lowrank_reparam(model, k, seed=0):
    """给每个含 lora_A/lora_B 的 PEFT 模块挂 U/V(固定正交, 不训练) 与 Z(k×k, 可训练)。

    同时冻结对应的 A/B，使 `flatten_trainable` 只看到 Z 集合（可训练维 = n_modules·k²）。
    通过 forward hook 把 `x @ V @ Zᵀ @ Uᵀ` 加到模块输出，不改 PEFT 的 state_dict/forward。
    返回每个模块的挂载记录列表。
    """
    k = int(k)
    assert k > 0, f"k must be > 0, got {k}"
    records = []
    module_index = 0
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        if getattr(module, "_lowrank", None) is not None:
            raise ValueError(f"{name} already has lowrank reparam attached")
        adapters = list(lora_a.keys())
        assert adapters, f"{name} has empty lora_A ModuleDict"
        submodules = {}
        for adapter_index, adapter in enumerate(adapters):
            a_weight = lora_a[adapter].weight
            b_weight = lora_b[adapter].weight
            in_features, out_features = int(a_weight.shape[1]), int(b_weight.shape[0])
            assert k <= min(in_features, out_features), (
                f"{name}[{adapter}] k={k} exceeds min(in={in_features}, out={out_features})")
            generator = torch.Generator().manual_seed(seed + 1009 * module_index + 17 * adapter_index)
            u = _orthogonal(out_features, k, generator, a_weight.dtype)
            v = _orthogonal(in_features, k, generator, a_weight.dtype)
            submodules[adapter] = _LowrankReparam(in_features, out_features, k, u, v)
            for param in (a_weight, b_weight):
                param.requires_grad_(False)
            records.append({
                "module": name,
                "adapter": adapter,
                "in_features": in_features,
                "out_features": out_features,
                "k": k,
                "z_name": f"{name}._lowrank.{adapter}.Z" if name else f"_lowrank.{adapter}.Z",
                "z_numel": k * k,
            })
        module.add_module("_lowrank", nn.ModuleDict(submodules))
        module._lowrank_keys = adapters
        module.register_forward_hook(_lowrank_forward_hook)
        module_index += 1
    assert records, "no LoRA module found (did build_lora_model run?)"
    return records


def lowrank_trainable_params(model):
    """返回 {Z} 可训练参数列表 (name, param)；用于断言可训练维 = Σ k²。"""
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "_lowrank." in name and name.endswith(".Z")
    ]


def load_lowrank_state_dict(model, state_dict):
    """把 {name: tensor} 载入 Z（PEFT save/load 不含 Z，需另行持久化）。"""
    params = dict(lowrank_trainable_params(model))
    assert set(state_dict.keys()) == set(params.keys()), (
        "lowrank state_dict keys mismatch: "
        f"missing={sorted(set(params) - set(state_dict))[:5]} "
        f"extra={sorted(set(state_dict) - set(params))[:5]}"
    )
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state_dict[name].to(device=param.device, dtype=param.dtype))


def get_lowrank_state_dict(model):
    return {name: param.detach().to("cpu").clone() for name, param in lowrank_trainable_params(model)}


# SMELL 3 lowrank_reparam selftest BEGIN — 玩具 LoRA 层上验证挂载/冻结/前向数学/正交性
def _make_fake_lora(in_features, out_features):
    class _FakeLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(in_features, out_features, bias=False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, 1, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(1, out_features, bias=False)})
            self.scaling = {"default": 2.0}

        def forward(self, x):
            out = self.base_layer(x)
            for adapter in self.lora_A:
                out = out + self.lora_B[adapter](self.lora_A[adapter](x)) * self.scaling[adapter]
            return out

    return _FakeLora()


def _selftest():
    torch.manual_seed(3)
    in_features, out_features, k = 6, 5, 2
    model = _make_fake_lora(in_features, out_features)
    a_before = model.lora_A["default"].weight.detach().clone()
    records = attach_lowrank_reparam(model, k=k, seed=11)
    results = {}
    results["one record"] = len(records) == 1
    results["A/B frozen"] = not any(
        param.requires_grad for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name)
    trainable = lowrank_trainable_params(model)
    results["only Z trainable"] = bool(
        len(trainable) == 1 and trainable[0][0].endswith("_lowrank.default.Z")
        and int(trainable[0][1].numel()) == k * k)
    results["A untouched"] = bool(torch.equal(model.lora_A["default"].weight, a_before))

    sub = model._lowrank["default"]
    results["U shape/orthonormal"] = bool(
        tuple(sub.U.shape) == (out_features, k)
        and torch.allclose(sub.U.t() @ sub.U, torch.eye(k), atol=1e-5))
    results["V shape/orthonormal"] = bool(
        tuple(sub.V.shape) == (in_features, k)
        and torch.allclose(sub.V.t() @ sub.V, torch.eye(k), atol=1e-5))
    results["Z shape"] = tuple(sub.Z.shape) == (k, k)

    x = torch.randn(4, in_features)
    base = model.base_layer(x) + model.lora_B["default"](model.lora_A["default"](x)) * 2.0
    with torch.no_grad():
        model._lowrank["default"].Z.copy_(torch.randn(k, k))
    out = model(x)
    expected_extra = x @ sub.V @ sub.Z.t() @ sub.U.t()
    results["hook adds U Z Vᵀ"] = bool(torch.allclose(out, base + expected_extra, atol=1e-5))

    results["state roundtrip"] = True
    sd = get_lowrank_state_dict(model)
    with torch.no_grad():
        model._lowrank["default"].Z.zero_()
    load_lowrank_state_dict(model, sd)
    results["state roundtrip"] = bool(torch.equal(model._lowrank["default"].Z.cpu(), sd[trainable[0][0]]))
    return results


if __name__ == "__main__":
    results = _selftest()
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("PASS" if all(results.values()) else "FAIL")
    raise SystemExit(0 if all(results.values()) else 1)
# SMELL 3 lowrank_reparam selftest END
