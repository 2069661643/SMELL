# Ailog 261012-183000 — SMELL-v3 DistilBERT：lr 扫描 + LD 表「单一串行链」交接（无人值守）

> 用户决定：停掉此前两进程布置（lr 扫描 + 等待型 LD 脚本），改为**单脚本串行链**一次性执行；TUI 将关闭，
> 全流程无人值守跑完，产物落盘，回来后按本文件指示查阅。
> 承接：`261012-151805`（fixed-60docs 缺陷 + LD 计划）、`261012-153000`（doc-order 实现 + BP cycle 补测）。

## 1. 运行方式（唯一入口）
- 脚本：`temp/run_bert_lr_ld_chain_bg.sh`（setsid nohup 后台；单一进程串行四阶段）
- 启动：2026-10-12 18:27:31；PID 文件 `temp/logs/last_bert_chain_lrld.pid`；运行目录 `temp/logs/bert_chain_lrld_<ts>/`
- `driver.log` 每阶段 `[HH:MM:SS] STEP ...` 打点；**阶段哨标**：
  - `stage_lr_scan.done`（lr 扫描完 + lr* 已定，~18:50）
  - `stage_ld.done`（LD 表 10 格完成，预计 ~09:00 次日）
  - `stage_chain.done`（报告生成完，全链结束）
- 自动报告：`temp/logs/bert_ld_report.txt`（固定路径，链尾生成；含 lr 扫描表 + LD 全表 + BP 参照 + 建议）

## 2. lr 处理（复刻 OPT 方法论，见 260929-094100）
1. 探针 δ 锚定：`probe_zoo_L8 @ lr4e-6` 实测 δ=6.871e-4（cycle、16k、sparse+step5 predictor）；目标 δ = **稀疏路径 BP 参考 0.243**（cycle L8 lr3e-4 实测）→ **lr\* ≈ 1.41e-3**（2× = 2.83e-3 储备档）；
2. lr 扫描（真实 16k、1 client × L8 × D22 × 3 轮，cycle）：`{5e-4, 1.4e-3, 2.8e-3, 4e-3, 1e-2}`，验证 δ∝lr 线性与 loss 随 lr 单调性（OPT 结论：单调降 → 取 δ 匹配档）；
3. clamp 规则：`[1e-4, 1.0]` 上限（OPT 曾因 1e-4 上限夹低 0.15 空转 13 轮）；`--delta-clip 1.0` 护离群 client。

## 3. LD 表（P1）格子（c30、cycle、6 轮、评测 250@bs8、全 6 层、sparse=0.4、CATV off）
| 序 | 格 | 参数 | 目的 |
|---|---|---|---|
| 1 | L8D4 | L8 D4 lr* | 便宜锚 |
| 2 | L8D8 | L8 D8 lr* | iso-M ↔ L16D4 |
| 3 | fixed_L8D8 | doc-order fixed | H2 对照 A（vs 格 2） |
| 4 | L16D4 | L16 D4 lr* | iso-M 对照 |
| 5 | L16D8 | L16 D8 lr* | iso-M ↔ L8D16 |
| 6 | L8D16 | L8 D16 lr* | iso-M 对照 |
| 7 | L8D8_lr2x | 2lr* | lr 储备档（OPT：单调性） |
| 8 | dense_L8D8 | attn=sdpa | 剥离剪枝惩罚锚 |
| 9 | B1_L8D4_r5 | 5 轮（40 篇唯一） | 等语料对照（vs 10） |
| 10 | B2_L2D4_r20 | 20 轮（40 篇唯一） | 低成本等语料对照 |

## 4. 前置事实（本链依赖，全部已实测）
- 基线（冻结头，前 250 样本）：acc 0.4880 / NLL 1.5317；集中对照 +11.5% NLL（天花板）；
- BP cycle 参考：dense L8 lr3e-4 +8.90%/10r（%/h≈+20.9）；sparse L8 +4.57%（自身 +14.58%）；
- H1+H2 均成立（rotation 6×、L 本身 2.4×，等语料等时长对照）；旧 fixed 数字仅作对照。

## 5. 回来后怎么做
1. 读 `temp/logs/bert_ld_report.txt`（或 `temp/logs/last_bert_chain_lrld_dir.txt` → `driver.log`）；
2. 按表中 **%/h × 相对下降 × δ 归一化** 选 r30 四臂配置（α×CATV、cycle、评测 500）；
   - r30 成本参考（0.2s/前向）：L8D4≈3.6h/臂、L8D8≈6.8h/臂、L8D16≈13h/臂（×4 臂）；
3. 若链中任何格 rc≠0，看对应 `temp/logs/bert_chain_lrld_*/<cell>.log`；
4. r30 启动脚本尚未创建（等选定配置）。

## 6. 变更列表
| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 18:26 | 停止 | 旧两进程布置（lr_scan + 等待型 ld） | 用户决定改单链 |
| 18:27 | 新增 | `temp/run_bert_lr_ld_chain_bg.sh`、`temp/bert_ld_report.py`（均 gitignore） | 单一串行链 + 自动报告 |
| 18:30 | 新增 | 本 ailog | 无人值守交接 |
