# SMELL 3 serial_fedavg NEW — 串行 FedAvg：ClientRunner（ZOO 本地更新）+ ServerAggregator + JSONL 指标

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class ClientRunner:
    """在 model 的可训练参数上做 local_steps 次 ZOO 更新。

    selector（CATV/Jenga sparse）由调用方或模型内部应用，这里只保存引用。
    """

    def __init__(self, model, local_steps=20, lr=1e-3, zo_eps=1e-3,
                 zo_directions=8, selector=None):
        self.model = model
        self.local_steps = int(local_steps)
        self.lr = float(lr)
        self.zo_eps = float(zo_eps)
        self.zo_directions = int(zo_directions)
        self.selector = selector

    def _loss_fn(self, input_ids):
        return lambda: self.model(input_ids, labels=input_ids).loss

    def run(self, input_ids_iterable):
        from src.train.lora import get_trainable_state_dict
        from src.train.zoo import apply_flat_delta, flatten_trainable, zo_grad

        names, sample_flat = flatten_trainable(self.model)
        assert names, "model has no trainable parameters (did LoRA wrapping run?)"
        device = sample_flat.device
        initial = get_trainable_state_dict(self.model)

        iterator = iter(input_ids_iterable)
        losses = []
        steps = 0
        for _ in range(self.local_steps):
            try:
                input_ids = next(iterator)
            except StopIteration:
                break
            if isinstance(input_ids, dict):
                input_ids = input_ids["input_ids"]
            input_ids = input_ids.to(device)
            with torch.no_grad():
                losses.append(float(self.model(input_ids, labels=input_ids).loss.detach().cpu()))
            grad = zo_grad(self.model, self._loss_fn(input_ids), self.zo_eps, self.zo_directions)
            grad_flat = torch.cat([grad[name].reshape(-1).float() for name in names])
            apply_flat_delta(self.model, (-self.lr * grad_flat).to(device))
            steps += 1

        final = get_trainable_state_dict(self.model)
        delta = {name: final[name] - initial[name] for name in initial}
        return {
            "state_dict": final,
            "delta": delta,
            "train_loss": (sum(losses) / len(losses)) if losses else None,
            "steps": steps,
            "lr": self.lr,
        }


class ServerAggregator:
    """FedAvg 聚合：samples（w_i=n_i/Σn）或 equal（w_i=1/N）。"""

    def __init__(self, weighting="samples"):
        assert weighting in ("samples", "equal"), f"unknown weighting: {weighting}"
        self.weighting = weighting

    @staticmethod
    def consistency_check(state_dicts):
        assert len(state_dicts) > 0, "no state dicts to aggregate"
        keys = set(state_dicts[0].keys())
        for index, state_dict in enumerate(state_dicts[1:], start=1):
            assert set(state_dict.keys()) == keys, (
                f"state_dict[{index}] key mismatch: "
                f"missing={sorted(keys - set(state_dict.keys()))[:5]} "
                f"extra={sorted(set(state_dict.keys()) - keys)[:5]}"
            )
            for key in keys:
                assert state_dict[key].shape == state_dicts[0][key].shape, (
                    f"state_dict[{index}][{key}] shape {tuple(state_dict[key].shape)} "
                    f"!= {tuple(state_dicts[0][key].shape)}"
                )
        return True

    def _weights(self, num_clients, counts):
        if self.weighting == "equal":
            return [1.0 / num_clients] * num_clients
        assert counts is not None and len(counts) == num_clients, (
            "samples weighting requires one count per state dict"
        )
        total = float(sum(counts))
        assert total > 0, "sum(counts) must be positive"
        return [float(count) / total for count in counts]

    def aggregate(self, state_dicts, counts=None):
        self.consistency_check(state_dicts)
        num_clients = len(state_dicts)
        weights = self._weights(num_clients, counts)
        aggregated = {}
        for key in sorted(state_dicts[0].keys()):
            reference = state_dicts[0][key]
            acc = torch.zeros_like(reference, dtype=torch.float32)
            for state_dict, weight in zip(state_dicts, weights):
                acc += state_dict[key].to(torch.float32) * weight
            aggregated[key] = acc.to(reference.dtype) if reference.dtype.is_floating_point else acc
        return aggregated


def append_metrics(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _selftest():
    import tempfile

    torch.manual_seed(0)
    sd1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
    sd2 = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([1.5])}
    sd3 = {"w": torch.tensor([5.0, 6.0]), "b": torch.tensor([2.5])}
    counts = [1, 2, 4]

    samples = ServerAggregator("samples").aggregate([sd1, sd2, sd3], counts)
    manual_w = (1.0 * sd1["w"] + 2.0 * sd2["w"] + 4.0 * sd3["w"]) / 7.0
    manual_b = (1.0 * sd1["b"] + 2.0 * sd2["b"] + 4.0 * sd3["b"]) / 7.0
    samples_ok = bool(torch.allclose(samples["w"], manual_w) and torch.allclose(samples["b"], manual_b))

    equal = ServerAggregator("equal").aggregate([sd1, sd2, sd3])
    equal_manual_w = (sd1["w"] + sd2["w"] + sd3["w"]) / 3.0
    equal_manual_b = (sd1["b"] + sd2["b"] + sd3["b"]) / 3.0
    equal_ok = bool(torch.allclose(equal["w"], equal_manual_w) and torch.allclose(equal["b"], equal_manual_b))

    mismatch_raised = False
    try:
        ServerAggregator.consistency_check([sd1, {"w": torch.tensor([1.0])}])
    except AssertionError:
        mismatch_raised = True

    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = Path(tmp) / "metrics.jsonl"
        append_metrics(jsonl_path, {"event": "round", "round": 0})
        append_metrics(jsonl_path, {"event": "round", "round": 1})
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    jsonl_ok = len(records) == 2 and records[1]["round"] == 1

    print(f"samples  agg w={samples['w'].tolist()} b={samples['b'].tolist()}")
    print(f"         manual w={manual_w.tolist()} b={manual_b.tolist()}")
    print(f"equal    agg w={equal['w'].tolist()} b={equal['b'].tolist()}")
    print(f"         manual w={equal_manual_w.tolist()} b={equal_manual_b.tolist()}")
    print(f"consistency_check mismatch raised: {mismatch_raised}")
    print(f"append_metrics jsonl records: {len(records)}")
    results = {
        "samples weighting": samples_ok,
        "equal weighting": equal_ok,
        "consistency_check": mismatch_raised,
        "append_metrics": jsonl_ok,
    }
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return all(results.values())


if __name__ == "__main__":
    ok = _selftest()
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
