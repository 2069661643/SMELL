# Ailog 261014-145500 — Phase N: 停 BERT a03-on + T_D 补录（D32 补评 r7/r8，统一 8 轮窗口）

> 用户指令：① 停掉当前的 a03 on 实验；② 补 D 对照表 T_D（8–9 轮窗口，D22 也进表、统一到最短 r，
> 解决 D32 缺 r8 点的 ⚠）。

## 1. 停止 BERT a03-on

- 识别：正在跑的是另一 session 的 BERT 链第 4 臂 `logs/fed/bert_da/r30_a03_catv_on/a03`
  （`temp/run_bert_d_a_chain_bg.sh`，PID 793046，14:21 启动）。
- 操作：`kill 793046`；父链脚本按设计收尾（写 `stage_a.done` → 生成 `bert_r30_report.txt` → `DA_CHAIN_ALL_DONE`，无新训练）。
- 损失：该臂在 round 0 内被杀（`metrics.jsonl` 尚无 round 事件），无可用数据；不影响的旧产物保留在
  `logs/fed/bert_da/r30_a03_{catv_off,catv_on}` 等目录。GPU 释放（1 MiB / 0%）。
- 注：另一 session 已备 `temp/run_bert_da_fixed_chain_bg.sh`（CATV vote NaN + block0 修复版），本次未触碰。

## 2. D32 事后补评（解决 T_D 窗口不对齐）

- 语义澄清：`adapter_roundNNN` 在 round NNN 结束后保存 ⇒ `adapter_round007` = 训练 8 轮后，
  与 D16/D22 pilot（`rounds=8`，in-run eval r0..r7）的端点 r7 **严格对齐**；用户注里的 "round008" 系 off-by-one，
  但为稳妥两个都评了。
- 命令（`temp/run_d32_r7r8_bg.sh`，与训练 run 同口径：fp32 + sdpa_prune + sparse 0.4 + step5 predictor + a03 全局 500 条）：

```bash
$PY src/eval/ppl.py --tag a03 --attn sdpa_prune --dtype fp32 --sparse 0.4 \
  --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
  --predictor checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth \
  --pruned-config checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth \
  --adapter logs/fed/zoo_k24_d32_lr0.30_r30/a03/adapter_round007 \
  --out temp/eval_d32_r7.json --per-sample-out temp/eval_d32_r7_persample.json
```

- 结果（full G-PPL, n=500）：r0 33.5296 → **r7 32.5071** → r8 32.4151 → r9 32.3408（in-run 32.3408）。
- **口径验证**：r9 事后复评 = 32.3408，与 in-run 值差 −0.0000 ⇒ 补评可信。

## 3. T_D（统一 8 轮窗口）

| D | 窗口 | r0 | 端点 | Δ | Δ% |
|---|---|---|---|---|---|
| 16 | r0→r7 | 33.5332 | 32.7144 | −0.8189 | −2.44% |
| 22 | r0→r7 | 33.5211 | 32.5335 | −0.9875 | −2.95% |
| 32 | r0→r7 | 33.5296 | 32.5071 | −1.0225 | **−3.05%** |

- 产物：`reports/opt/tables/table7_d_comparison.{md,tex}` + `data/table7_d_comparison.csv`；
  对账断言 +6（D16/D22 r7、D32 r0/r7/r8/r9），总 38/38 PASS。
- 顺带：`save_fig` 改为 PDF 确定性输出（`metadata={"CreationDate": None}`），两次重跑哈希一致。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 14:32 | 停止 | PID 793046（BERT a03-on） | GPU 释放；该臂无数据损失 |
| 14:32 | 新增 | `temp/run_d32_r7r8_bg.sh` | 三点补评（r7/r8/r9），产物 `temp/eval_d32_r*.json` |
| 14:50 | 修改 | `reports/opt/make_report.py` | 新增 `table7_d_comparison` + `d_checks`（+6 断言）+ PDF 确定性输出 |
| 14:52 | 生成 | `reports/opt/tables/table7_d_comparison.*`、`data/table7_d_comparison.csv` | T_D 表（8 轮窗口统一） |
| 14:53 | 新增 | 本 ailog | 记录 |
