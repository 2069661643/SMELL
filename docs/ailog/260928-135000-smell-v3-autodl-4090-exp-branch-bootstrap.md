# Ailog 260928-135000 — Phase N: AutoDL 4090 新机接入：exp 分支下拉 + AGENTS.md 多机化

> 新服务器（AutoDL 容器，1×RTX 4090 24G，host `autodl-container-861840a478-12f9c2aa`）首次接入 SMELL-v3。
> 本 session 只做交接核查与文档更新，不跑训练。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 13:35 | 校验交接权重（只读） | `checkpoints/posemb_step1/a01_pos_only_500step/`、`checkpoints/predictor/step4_a01_pos_only_causal/` | — | 3 个 sha256 与 `docs/weight-manifest.md` 全部一致 |
| 13:36 | 下拉实验分支 | git | `exp/rank-rotation-lora` | 原 clone 停在 `main`（落后 19 个提交）；已建本地跟踪分支并 checkout |
| 13:37 | 修 submodule 拉取 | 本机 `.git/config` | 3 个 submodule URL 改为 SSH | 本机 GitHub HTTPS 挂死、SSH 正常；Jenga/VQ 初始化成功 |
| 13:40–13:55 | 更新 AGENTS.md | `AGENTS.md` | 头部、进度表、最新进展、AutoDL 工作机节、Git/ailog 索引 | 多机化：换机先确认分支/机器；补 step4 predictor、k-scan、TD-3 停止等最新结论 |
| 13:45 | 静态检查 | `src/**/*.py` | `python3 -m py_compile` | 全通过（base py3.12 有 torch 2.8.0+cu128，无 jenga） |
| 13:50+ | 重试 FwdLLM submodule | `third_party/FwdLLM` | 清陈旧状态后 `git submodule update`（SSH） | 本 ailog 提交时仍在下载（~90M） |

## 本机环境（已核验）

- 1×**RTX 4090 24GB**（sm_89，driver 595 / CUDA 13.2）、20 CPU、系统盘 30G（无 `nvidia-smi` 争卡问题）。
- conda：`~/miniconda3`（base py3.12 + torch 2.8.0+cu128）；transformers/peft/flash-attn/jenga 均未装，无 `SMELL` env。
- `scripts/setup_server.sh` 默认 `$HOME/miniconda3` + env `SMELL` 正好适配本机（尚未执行）。
- 已传输：pos_only 位置表、step4 causal predictor（sha256 全对）；缺：`third_party/Jenga/checkpoints/opt-350m/`、`dataset_v3/`（warmup 池）。
- GitHub：HTTPS git 挂死，SSH 正常（submodule URL 已在本机改 SSH，注意别 `git submodule sync`）。

## 待办

- [ ] FwdLLM submodule 下载完成；`bash scripts/setup_server.sh` 建 env（先 `df -h`，盘仅 30G）。
- [ ] 拉/传 opt-350m 与 warmup 池（或按 `build_discovery_16k.py`/`build_warmup_16k.py` 重建）。
- [ ] 用 `step4_a01_pos_only_causal` 复现 TD-2'（`sdpa_prune` fp32），过门控后再议 TD-3。
- [ ] AGENTS.md 中云端运行态（`f4e879d` 时点）如与 amax 实际不符，以云端 ailog 为准。
