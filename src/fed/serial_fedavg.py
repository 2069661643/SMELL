# SMELL 3 serial_fedavg NEW — 串行 FedAvg：ClientRunner（ZOO 本地更新）+ ServerAggregator + JSONL 指标

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# SMELL 3 CATV votes BEGIN — 模型 config 定位 + predictor 原始分数逐层累积器
def resolve_model_config(model):
    """定位 PeftModel 内部的 OPTConfig（vote_callback/consensus_mask 挂在该共享 config 上）。"""
    candidates = [model]
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        candidates.append(get_base())
    base = getattr(model, "base_model", None)
    if base is not None:
        candidates.append(getattr(base, "model", None))
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "num_hidden_layers"):
            return config
    raise AttributeError(f"cannot locate model config on {type(model).__name__}")


class _VoteAccumulator:
    """逐层累积 CATV 块投票：每次 forward 累加 batch 维总和与行数，最后取均值。"""

    def __init__(self, first_layer, last_layer):
        self.first_layer = int(first_layer)
        self.last_layer = int(last_layer)
        self.sums = {}
        self.counts = {}

    def __call__(self, layer_idx, block_scores):
        layer_idx = int(layer_idx)
        if not (self.first_layer <= layer_idx <= self.last_layer):
            return
        values = block_scores.detach().float()
        if values.dim() == 1:
            values = values.unsqueeze(0)
        total = values.sum(dim=0)
        if layer_idx in self.sums:
            self.sums[layer_idx] = self.sums[layer_idx] + total
            self.counts[layer_idx] += int(values.size(0))
        else:
            self.sums[layer_idx] = total.clone()
            self.counts[layer_idx] = int(values.size(0))

    def average(self):
        return {
            layer: (self.sums[layer] / float(self.counts[layer])).detach().to(
                device="cpu", dtype=torch.float32
            ).clone()
            for layer in sorted(self.sums)
        }
# SMELL 3 CATV votes END


class ClientRunner:
    """在 model 的可训练参数上做 local_steps 次本地更新（ZOO 或 BP）。

    # SMELL 3 ClientRunner bp MODIFIED — trainer="bp" 时用标准反传（autocast bf16 + AdamW + 梯度裁剪）替代中心差分
    selector（CATV/Jenga sparse）由调用方或模型内部应用，这里只保存引用。
    collect_votes=True 时在本地训练期间收集 CATV 块投票（每次 forward，含 ZOO 中心差分两次探测）。
    """

    def __init__(self, model, local_steps=20, lr=1e-3, zo_eps=1e-3,
                 zo_directions=8, selector=None, collect_votes=False,
                 # SMELL 3 ClientRunner trainer ADD — zoo=前向梯度（默认，字节兼容）；bp=标准反传本地训练
                 trainer="zoo", bp_clip=1.0,
                 # SMELL 3 ClientRunner rank_rotation ADD — 激活秩掩码/秩集合（None = 旧行为，全参数）
                 active_index=None, active_ranks=None):
        assert trainer in ("zoo", "bp"), f"unknown trainer: {trainer}"
        self.model = model
        self.local_steps = int(local_steps)
        self.lr = float(lr)
        self.zo_eps = float(zo_eps)
        self.zo_directions = int(zo_directions)
        self.selector = selector
        # SMELL 3 ClientRunner collect_votes ADD — CATV 投票开关
        self.collect_votes = bool(collect_votes)
        self.trainer = str(trainer)
        self.bp_clip = float(bp_clip)
        # SMELL 3 ClientRunner rank_rotation ADD — 秩掩码：ZOO 子空间收缩 / BP 非激活快照-还原
        self.active_index = active_index
        self.active_ranks = list(active_ranks) if active_ranks is not None else None

    def _loss_fn(self, input_ids):
        return lambda: self.model(input_ids, labels=input_ids).loss

    def run(self, input_ids_iterable):
        from src.train.lora import get_trainable_state_dict
        from src.train.zoo import _restore_flat, apply_flat_delta, flatten_trainable, zo_grad

        # SMELL 3 ClientRunner votes BEGIN — CATV: 本地训练期间安装逐层投票累积回调
        vote_config = None
        vote_accumulator = None
        if self.collect_votes:
            vote_config = resolve_model_config(self.model)
            num_layers = int(getattr(vote_config, "num_hidden_layers"))
            vote_accumulator = _VoteAccumulator(num_layers // 2 - 1, num_layers - 2)
            vote_config.vote_callback = vote_accumulator
        try:
            names, sample_flat = flatten_trainable(self.model)
            assert names, "model has no trainable parameters (did LoRA wrapping run?)"
            device = sample_flat.device
            initial = get_trainable_state_dict(self.model)
            # SMELL 3 ClientRunner bp optimizer ADD — BP: 对全部可训练参数建 AdamW（--lr 可调）
            optimizer = None
            bp_params = None
            if self.trainer == "bp":
                bp_params = [param for param in self.model.parameters() if param.requires_grad]
                optimizer = torch.optim.AdamW(bp_params, lr=self.lr, weight_decay=0.0)
            # SMELL 3 ClientRunner rank_rotation ADD — BP: 记录初始全长，step 后把非激活位置逐位还原
            init_full = sample_flat.clone() if self.active_index is not None else None

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
                # SMELL 3 ClientRunner bp BEGIN — zero_grad → autocast bf16 前向 → backward → clip → step
                if self.trainer == "bp":
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        loss = self.model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
                    loss.backward()
                    losses.append(float(loss.detach().float().cpu()))
                    if self.bp_clip > 0:
                        torch.nn.utils.clip_grad_norm_(bp_params, self.bp_clip)
                    optimizer.step()
                    # SMELL 3 ClientRunner rank_rotation BEGIN — 非激活秩还原为初始值（AdamW 动量不漂移）
                    if self.active_index is not None:
                        assert init_full is not None, "active_index requires an initial snapshot"
                        current = flatten_trainable(self.model)[1]
                        mask = self.active_index.to(device=current.device)
                        inactive = torch.ones(current.numel(), dtype=torch.bool, device=current.device)
                        inactive[mask] = False
                        current[inactive] = init_full.to(device=current.device, dtype=current.dtype)[inactive]
                        _restore_flat(self.model, current)
                    # SMELL 3 ClientRunner rank_rotation END
                    steps += 1
                    continue
                # SMELL 3 ClientRunner bp END
                with torch.no_grad():
                    losses.append(float(self.model(input_ids, labels=input_ids).loss.detach().cpu()))
                # SMELL 3 ClientRunner rank_rotation ADD — ZOO 只在 active_index 子空间估计/扰动
                grad = zo_grad(self.model, self._loss_fn(input_ids), self.zo_eps, self.zo_directions,
                               index=self.active_index)
                grad_flat = torch.cat([grad[name].reshape(-1).float() for name in names])
                apply_flat_delta(self.model, (-self.lr * grad_flat).to(device))
                steps += 1
        finally:
            # SMELL 3 ClientRunner votes END — 训练结束移除回调，避免污染其它 client/轮次
            if vote_config is not None:
                vote_config.vote_callback = None

        final = get_trainable_state_dict(self.model)
        delta = {name: final[name] - initial[name] for name in initial}
        result = {
            "state_dict": final,
            "delta": delta,
            "train_loss": (sum(losses) / len(losses)) if losses else None,
            "steps": steps,
            "lr": self.lr,
            # SMELL 3 ClientRunner trainer ADD — 记录本 client 使用的本地训练器
            "trainer": self.trainer,
            # SMELL 3 ClientRunner rank_rotation ADD — 本轮激活秩（None = 无掩码全参数）
            "active_ranks": sorted(self.active_ranks) if self.active_ranks is not None else None,
        }
        # SMELL 3 ClientRunner votes ADD — CATV: 返回逐层 CPU fp32 平均块投票
        if vote_accumulator is not None:
            result["votes"] = vote_accumulator.average()
        return result


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

    # SMELL 3 ServerAggregator accumulate_votes ADD — CATV: 逐层累加客户端投票（服务器端求和）
    @staticmethod
    def accumulate_votes(vote_sum, votes):
        for layer_idx, tensor in votes.items():
            tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
            if layer_idx in vote_sum:
                vote_sum[layer_idx] = vote_sum[layer_idx] + tensor
            else:
                vote_sum[layer_idx] = tensor.clone()
        return vote_sum

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
