# SMELL 3 zoo NEW — 中心差分前向梯度估计与 FedAvg 聚合工具

import torch


def _trainable_named_parameters(model):
    return sorted(
        ((name, param) for name, param in model.named_parameters() if param.requires_grad),
        key=lambda item: item[0],
    )


def flatten_trainable(model):
    named = _trainable_named_parameters(model)
    names = [name for name, _ in named]
    if not named:
        return names, torch.zeros(0)
    flat = torch.cat([param.detach().reshape(-1) for _, param in named])
    return names, flat


def apply_flat_delta(model, delta):
    named = _trainable_named_parameters(model)
    if not named:
        return
    delta = delta.detach()
    expected = sum(param.numel() for _, param in named)
    assert delta.numel() == expected, f"delta has {delta.numel()} elements, expected {expected}"
    offset = 0
    with torch.no_grad():
        for _, param in named:
            numel = param.numel()
            chunk = delta[offset : offset + numel].reshape(param.shape).to(param.dtype)
            param.add_(chunk)
            offset += numel


def _restore_flat(model, flat):
    named = _trainable_named_parameters(model)
    if not named:
        return
    offset = 0
    with torch.no_grad():
        for _, param in named:
            numel = param.numel()
            param.copy_(flat[offset : offset + numel].reshape(param.shape))
            offset += numel


def zo_grad(model, loss_fn, eps, num_directions, generator=None) -> dict:
    names, flat = flatten_trainable(model)
    if not names:
        return {}
    device = flat.device
    dim = flat.numel()
    grad = torch.zeros(dim, dtype=torch.float32, device=device)  # SMELL 3 zo_grad FIXED — grad 需与 direction 同设备
    for _ in range(num_directions):
        direction = torch.randn(dim, generator=generator, dtype=torch.float32).to(device)
        direction = direction / direction.norm()
        apply_flat_delta(model, eps * direction)
        with torch.no_grad():
            loss_plus = float(loss_fn().detach().cpu())
        apply_flat_delta(model, -2.0 * eps * direction)
        with torch.no_grad():
            loss_minus = float(loss_fn().detach().cpu())
        _restore_flat(model, flat)
        assert torch.equal(flatten_trainable(model)[1], flat), "zo_grad failed to restore parameters"
        # SMELL 3 zo_grad：单位范数 v 满足 E[vvᵀ]=I/dim，按 dim 补偿后才是无偏梯度估计
        grad += dim * ((loss_plus - loss_minus) / (2.0 * eps)) * direction
    grad /= float(num_directions)
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

    return results


if __name__ == "__main__":
    results = _run_selftest()
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("PASS" if all(results.values()) else "FAIL")
    raise SystemExit(0 if all(results.values()) else 1)
