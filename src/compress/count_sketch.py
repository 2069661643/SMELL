# SMELL 3 compress count_sketch NEW — Count Sketch（third_party/VQ 复刻）用于 ZOO client delta 上行压缩

import math

import numpy as np


class CountSketch:
    """Count Sketch 压缩/解压（与 third_party/VQ/main.py 的 hash/sign 语义一致）。

    参数
    ----
    dim       : 原始向量维度（活跃可训练参数数）
    channels  : channel 数 C（ZOO 默认 local_steps * zo_directions）
    c_dim     : 每个 channel 的桶数（c_dim * channels / dim = 压缩率）
    seed      : hash/sign seed 序列的基 seed；client 间必须共享
    device    : torch device（CPU/GPU 均可）

    说明
    ----
    - 每个 channel 使用独立的 hash/sign seed；`seed` 只决定 seed 序列，
      所有 client 用同一个 `seed` 才能在线性聚合时对齐。
    - `h_j(i) = (i * a_j) % c_dim`，`s_j(i) = +1/-1`（VQ 原实现的整数版本）。
    - 额外约束 `gcd(a_j, c_dim) = 1`，避免小 c_dim 时桶覆盖退化。
    - 同一实例会缓存各 channel 的 h/s 表；不同 dim/c_dim 需新建实例。
    """

    def __init__(self, dim, channels, c_dim, seed, device="cpu"):
        import torch

        dim = int(dim)
        channels = int(channels)
        c_dim = int(c_dim)
        seed = int(seed)
        assert dim > 0, f"dim must be > 0, got {dim}"
        assert channels > 0, f"channels must be > 0, got {channels}"
        assert c_dim > 0, f"c_dim must be > 0, got {c_dim}"
        self.dim = dim
        self.channels = channels
        self.c_dim = c_dim
        self.seed = seed
        self.device = torch.device(device)
        self.hash_seeds, self.sigma_seeds = self._generate_seeds()
        self._indices = torch.arange(dim, dtype=torch.int64, device=self.device)
        self._hashes = None
        self._signs = None

    def _generate_seeds(self):
        rng = np.random.default_rng(self.seed)
        hash_seeds = []
        sigma_seeds = []
        for _ in range(self.channels):
            a = int(rng.integers(2, 1_000_000))
            if self.c_dim > 1:
                while math.gcd(a, self.c_dim) != 1:
                    a = int(rng.integers(2, 1_000_000))
            b = int(rng.integers(10_000, 1_000_000))
            hash_seeds.append(a)
            sigma_seeds.append(b)
        return hash_seeds, sigma_seeds

    def _ensure_tables(self):
        import torch

        if self._hashes is not None:
            return
        hashes = torch.empty((self.channels, self.dim), dtype=torch.int64, device=self.device)
        signs = torch.empty((self.channels, self.dim), dtype=torch.float32, device=self.device)
        for index, (hash_seed, sigma_seed) in enumerate(zip(self.hash_seeds, self.sigma_seeds)):
            hashes[index] = (self._indices * hash_seed) % self.c_dim
            sign_raw = (self._indices * sigma_seed) % 1_000_000
            signs[index] = torch.where(
                sign_raw <= 500_000,
                torch.ones((), dtype=torch.float32, device=self.device),
                -torch.ones((), dtype=torch.float32, device=self.device),
            )
        self._hashes = hashes
        self._signs = signs

    def compress(self, vector):
        """把一维向量压缩为 (channels, c_dim) 的 float32 sketch。"""
        import torch

        self._ensure_tables()
        if not torch.is_tensor(vector):
            vector = torch.as_tensor(vector)
        vector = vector.detach().reshape(-1).to(device=self.device, dtype=torch.float32)
        assert vector.numel() == self.dim, f"vector has {vector.numel()} elements, expected {self.dim}"
        sketch = torch.zeros((self.channels, self.c_dim), dtype=torch.float32, device=self.device)
        for index in range(self.channels):
            sketch[index].scatter_add_(0, self._hashes[index], vector * self._signs[index])
        return sketch

    def decompress(self, sketch):
        """把 (channels, c_dim) sketch 还原为一维 float32 向量（channel 平均）。"""
        import torch

        self._ensure_tables()
        if not torch.is_tensor(sketch):
            sketch = torch.as_tensor(sketch)
        sketch = sketch.detach().to(device=self.device, dtype=torch.float32)
        assert tuple(sketch.shape) == (self.channels, self.c_dim), (
            f"sketch shape {tuple(sketch.shape)} != {(self.channels, self.c_dim)}")
        out = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        for index in range(self.channels):
            out += sketch[index][self._hashes[index]] * self._signs[index]
        return out / float(self.channels)

    def roundtrip(self, vector):
        sketch = self.compress(vector)
        return sketch, self.decompress(sketch)

    @staticmethod
    def cosine(left, right):
        import torch

        left = left.detach().reshape(-1).float()
        right = right.detach().reshape(-1).float().to(device=left.device)
        denom = float(left.norm() * right.norm())
        if denom <= 0.0:
            return 0.0
        return float(torch.dot(left, right) / denom)

    @property
    def sketch_numel(self):
        return int(self.channels * self.c_dim)

    def sketch_bytes(self, bytes_per_element=4):
        return int(self.sketch_numel * int(bytes_per_element))


def _selftest():
    import torch

    results = {}
    torch.manual_seed(0)
    dim = 4096
    vector = torch.randn(dim)

    # 50% 预算：cos 应显著高于 0，且低于 1（Count Sketch 有损）
    compressor = CountSketch(dim, channels=8, c_dim=256, seed=1234)
    sketch, restored = compressor.roundtrip(vector)
    cos_half = CountSketch.cosine(vector, restored)
    results["50% finite"] = bool(torch.isfinite(restored).all())
    results["50% cos in (0.5, 0.95)"] = bool(0.5 < cos_half < 0.95)
    results["50% shape"] = tuple(sketch.shape) == (8, 256)

    # 100% 预算：Count Sketch 仍是有损的（N=d 时理论 cos≈1/sqrt(2)）；
    # CLI 在 rate=1 时直接绕过 VQ，所以这里只做健全性检查。
    full = CountSketch(dim, channels=8, c_dim=512, seed=1234)
    _, restored_full = full.roundtrip(vector)
    cos_full = CountSketch.cosine(vector, restored_full)
    results["100% cos > 0.6"] = bool(cos_full > 0.6)

    # seed 可复现 / 不同 seed 不同
    same = CountSketch(dim, channels=8, c_dim=256, seed=1234)
    other = CountSketch(dim, channels=8, c_dim=256, seed=1235)
    results["seed deterministic"] = bool(torch.equal(compressor.compress(vector), same.compress(vector)))
    results["seed varies"] = bool(not torch.equal(compressor.compress(vector), other.compress(vector)))

    # 线性性：S(x+y) 的还原 == S(x)+S(y) 的还原（同一 hash 实例）
    left = torch.randn(dim)
    right = torch.randn(dim)
    restored_sum = compressor.decompress(compressor.compress(left + right))
    restored_parts = compressor.decompress(compressor.compress(left) + compressor.compress(right))
    results["linear"] = bool(torch.allclose(restored_sum, restored_parts, atol=1e-4, rtol=1e-4))

    # 桶覆盖：gcd(a, c_dim)=1 时 h 应覆盖全部桶
    small = CountSketch(dim, channels=1, c_dim=17, seed=7)
    small._ensure_tables()
    results["bucket coverage"] = int(torch.unique(small._hashes[0]).numel()) == 17

    # sign 平衡
    compressor._ensure_tables()
    sign_mean = float(compressor._signs.mean())
    results["sign balanced"] = bool(abs(sign_mean) < 0.05)

    # c_dim=1 不崩
    one = CountSketch(dim, channels=2, c_dim=1, seed=9)
    _, restored_one = one.roundtrip(vector)
    results["c_dim=1 finite"] = bool(torch.isfinite(restored_one).all())

    print(f"[count_sketch] dim={dim} 50% cos={cos_half:.4f} 100% cos={cos_full:.4f} "
          f"sign_mean={sign_mean:.4f}")
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return all(results.values())


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
