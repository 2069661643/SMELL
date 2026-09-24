# SMELL 3 zoo NEW — 中心差分前向梯度估计与 FedAvg 聚合工具

import torch


def _trainable_named_parameters(model):
    return sorted(
        ((name, param) for name, param in model.named_parameters() if param.requires_grad),
        key=lambda item: item[0],
    )


# SMELL 3 zoo rank_rotation BEGIN — 秩掩码：把 LoRA 的若干秩映射为平坦参数上的索引子集
def build_active_index(model, active_ranks):
    """按 `_trainable_named_parameters` 排序遍历，收集 active_ranks 对应的平坦索引。

    LoRA 约定：`lora_A.weight` 形状 (r, in)，秩 i = 行 i；`lora_B.weight` 形状 (out, r)，
    秩 i = 列 i。非 LoRA 可训练参数（若有）全部纳入。返回 CPU `torch.LongTensor`。
    """
    ranks = sorted(int(rank) for rank in active_ranks)
    assert ranks, "active_ranks must be non-empty"
    named = _trainable_named_parameters(model)
    assert named, "model has no trainable parameters"
    indices = []
    offset = 0
    lora_rank = None
    for name, param in named:
        if "lora_A" in name:
            rank = int(param.shape[0])
            if lora_rank is None:
                lora_rank = rank
            assert rank == lora_rank, f"LoRA rank mismatch: {name} has r={rank}, expected {lora_rank}"
            in_features = int(param.shape[1])
            for index in ranks:
                assert 0 <= index < rank, f"active rank {index} out of range for {name} (r={rank})"
                base = offset + index * in_features
                indices.extend(range(base, base + in_features))
        elif "lora_B" in name:
            rank = int(param.shape[1])
            if lora_rank is None:
                lora_rank = rank
            assert rank == lora_rank, f"LoRA rank mismatch: {name} has r={rank}, expected {lora_rank}"
            out_features = int(param.shape[0])
            for index in ranks:
                assert 0 <= index < rank, f"active rank {index} out of range for {name} (r={rank})"
                indices.extend(offset + j * rank + index for j in range(out_features))
        else:
            indices.extend(range(offset, offset + int(param.numel())))
        offset += int(param.numel())
    assert lora_rank is not None, "no LoRA trainable parameters found for rank masking"
    return torch.tensor(indices, dtype=torch.long)
# SMELL 3 zoo rank_rotation END


def flatten_trainable(model, index=None):
    named = _trainable_named_parameters(model)
    names = [name for name, _ in named]
    if not named:
        return names, torch.zeros(0)
    flat = torch.cat([param.detach().reshape(-1) for _, param in named])
    # SMELL 3 zoo rank_rotation ADD — index 非 None 时仅取激活子空间
    if index is not None:
        flat = flat[index.to(device=flat.device)]
    return names, flat


def apply_flat_delta(model, delta, index=None):
    named = _trainable_named_parameters(model)
    if not named:
        return
    delta = delta.detach()
    expected = sum(param.numel() for _, param in named)
    # SMELL 3 zoo rank_rotation BEGIN — 掩码：delta 为激活子空间长度，散射回全长（其余为 0）
    if index is not None:
        full = torch.zeros(expected, dtype=delta.dtype, device=delta.device)
        full[index.to(device=delta.device)] = delta
        delta = full
    assert delta.numel() == expected, f"delta has {delta.numel()} elements, expected {expected}"
    # SMELL 3 zoo rank_rotation END
    offset = 0
    with torch.no_grad():
        for _, param in named:
            numel = param.numel()
            chunk = delta[offset : offset + numel].reshape(param.shape).to(param.dtype)
            param.add_(chunk)
            offset += numel


def _restore_flat(model, flat, index=None):
    named = _trainable_named_parameters(model)
    if not named:
        return
    # SMELL 3 zoo rank_rotation BEGIN — 掩码：以当前长度全长为底，仅覆盖激活子空间（非激活保持原值）
    if index is not None:
        _, current = flatten_trainable(model)
        mask = index.to(device=current.device)
        current[mask] = flat.to(device=current.device, dtype=current.dtype)
        flat = current
    # SMELL 3 zoo rank_rotation END
    offset = 0
    with torch.no_grad():
        for _, param in named:
            numel = param.numel()
            param.copy_(flat[offset : offset + numel].reshape(param.shape))
            offset += numel


def zo_grad(model, loss_fn, eps, num_directions, generator=None, index=None) -> dict:
    names, full_flat = flatten_trainable(model)
    if not names:
        return {}
    device = full_flat.device
    # SMELL 3 zoo rank_rotation BEGIN — index 非 None 时中心差分只在激活子空间进行，dim 收缩为该子空间元素数
    if index is None:
        active = full_flat
        active_index = None
    else:
        active_index = index.to(device=device)
        active = full_flat[active_index]
    total = int(full_flat.numel())
    dim = active.numel()
    assert dim > 0, "active subspace is empty"
    grad = torch.zeros(dim, dtype=torch.float32, device=device)  # SMELL 3 zo_grad FIXED — grad 需与 direction 同设备
    for _ in range(num_directions):
        direction = torch.randn(dim, generator=generator, dtype=torch.float32).to(device)
        direction = direction / direction.norm()
        apply_flat_delta(model, eps * direction, index=active_index)
        with torch.no_grad():
            loss_plus = float(loss_fn().detach().cpu())
        apply_flat_delta(model, -2.0 * eps * direction, index=active_index)
        with torch.no_grad():
            loss_minus = float(loss_fn().detach().cpu())
        if active_index is None:
            _restore_flat(model, full_flat)
        else:
            _restore_flat(model, full_flat[active_index], index=active_index)
        assert torch.equal(flatten_trainable(model)[1], full_flat), "zo_grad failed to restore parameters"
        # SMELL 3 zo_grad：单位范数 v 满足 E[vvᵀ]=I/dim，按 dim 补偿后才是无偏梯度估计
        grad += dim * ((loss_plus - loss_minus) / (2.0 * eps)) * direction
    grad /= float(num_directions)
    if active_index is not None:
        full_grad = torch.zeros(total, dtype=grad.dtype, device=device)
        full_grad[active_index] = grad
        grad = full_grad
    # SMELL 3 zoo rank_rotation END
    named = _trainable_named_parameters(model)
    out = {}
    offset = 0
    for name, param in named:
        numel = param.numel()
        out[name] = grad[offset : offset + numel].reshape(param.shape).clone()
        offset += numel
    return out


def aggregate_fedavg(dicts, weights) -> dict:
    assert len(dicts) == len(weights), f"{len(dicts)} dicts vs {len(weights)} weights"
    assert len(dicts) > 0, "no state dicts to aggregate"
    total = float(sum(weights))
    assert abs(total - 1.0) < 1e-6, f"weights must sum to 1, got {total}"
    keys = set(dicts[0].keys())
    for index, state_dict in enumerate(dicts):
        assert set(state_dict.keys()) == keys, f"state_dict keys mismatch at index {index}"
    out = {}
    for key in sorted(keys):
        acc = torch.zeros_like(dicts[0][key], dtype=torch.float32)
        for state_dict, weight in zip(dicts, weights):
            acc += state_dict[key].to(torch.float32) * float(weight)
        out[key] = acc.to(dicts[0][key].dtype)
    return out


def _run_selftest():
    results = {}

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)

    names, flat = flatten_trainable(model)
    delta = torch.randn(flat.numel())
    apply_flat_delta(model, delta)
    _, flat_after = flatten_trainable(model)
    results["flatten/apply roundtrip"] = bool(torch.allclose(flat_after, flat + delta))
    _restore_flat(model, flat)
    results["restore exact"] = bool(torch.equal(flatten_trainable(model)[1], flat))

    def loss_fn():
        return ((model(x) - y) ** 2).mean()

    model.zero_grad()
    auto_loss = loss_fn()
    auto_loss.backward()
    grad_auto = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    model.zero_grad()

    generator = torch.Generator().manual_seed(1234)
    grad_zo = zo_grad(model, loss_fn, eps=1e-3, num_directions=2048, generator=generator)
    results["zo_grad vs autograd"] = all(
        bool(torch.allclose(grad_zo[name], grad_auto[name], atol=1e-1, rtol=1e-1))
        for name in grad_auto
    )
    results["zo_grad restore exact"] = bool(torch.equal(flatten_trainable(model)[1], flat))

    sd1 = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    sd2 = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([5.0])}
    sd3 = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([7.0])}
    agg = aggregate_fedavg([sd1, sd2, sd3], [0.2, 0.3, 0.5])
    expected_a = torch.tensor([3.6, 4.6])
    expected_b = torch.tensor([5.6])
    results["aggregate_fedavg"] = bool(
        torch.allclose(agg["a"], expected_a) and torch.allclose(agg["b"], expected_b)
    )

    # SMELL 3 zoo rank_rotation ADD — 掩码测试：active_index 取秩、apply/restore 往返与非激活冻结
    torch.manual_seed(7)
    r, in_features, out_features = 4, 6, 5
    lora_model = torch.nn.ModuleDict({
        "q_proj": torch.nn.ModuleDict({
            "lora_A": torch.nn.ModuleDict({"default": torch.nn.Linear(in_features, r, bias=False)}),
            "lora_B": torch.nn.ModuleDict({"default": torch.nn.Linear(r, out_features, bias=False)}),
        }),
    })
    index = build_active_index(lora_model, {2})
    a_weight = lora_model["q_proj"]["lora_A"]["default"].weight.detach()
    b_weight = lora_model["q_proj"]["lora_B"]["default"].weight.detach()
    expected_index = torch.tensor(
        [2 * in_features + j for j in range(in_features)]
        + [r * in_features + j * r + 2 for j in range(out_features)]
    )
    results["build_active_index offsets"] = bool(torch.equal(index, expected_index))
    _, flat_masked = flatten_trainable(lora_model, index=index)
    expected_masked = torch.cat([a_weight[2, :].reshape(-1), b_weight[:, 2].reshape(-1)])
    results["build_active_index rows_cols"] = bool(torch.equal(flat_masked, expected_masked))

    before = flatten_trainable(lora_model)[1].clone()
    inactive_mask = torch.ones(before.numel(), dtype=torch.bool)
    inactive_mask[index] = False
    delta = torch.randn(index.numel())
    apply_flat_delta(lora_model, delta, index=index)
    after = flatten_trainable(lora_model)[1]
    results["masked apply active only"] = bool(
        torch.allclose(after[index], before[index] + delta)
        and torch.equal(after[inactive_mask], before[inactive_mask])
    )
    _restore_flat(lora_model, before[index], index=index)
    results["masked restore roundtrip"] = bool(torch.equal(flatten_trainable(lora_model)[1], before))

    return results


if __name__ == "__main__":
    results = _run_selftest()
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("PASS" if all(results.values()) else "FAIL")
    raise SystemExit(0 if all(results.values()) else 1)
