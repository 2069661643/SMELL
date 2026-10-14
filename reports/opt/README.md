# OPT-350M results package

由 `reports/opt/make_report.py` 从原始产物自动生成（纯 CPU、确定性；禁止手工改本目录数字）。

## 产物

- `figures/fig1_a03_convergence.{pdf,png}` — a03 主曲线：k=4/D=22(30r)、k=24/D=16(60r)、k=24/D=32(74r)、k=24/D=32+CATV(74r)，full 与 answer G-PPL。同口径（无掩码）。
- `figures/fig2_catv_alpha.{pdf,png}` — CATV × alpha：a03/a01 的 off / on(同口径) / on(部署) 双指标曲线。
- `figures/fig3_efficiency.{pdf,png}` — 左：full G-PPL vs 实测墙钟小时（完整链）；右：PPL/h 条形（统一取 0–29 轮口径）。
- `figures/fig4_pos_adaptation.{pdf,png}` — 位置适配 B/C 的 16k/2k G-PPL。
- `figures/fig5_catv_alpha01.{pdf,png}` — α=0.1 的 CATV 三条曲线（off (baseline) / on (eval off) / on (eval on)），16k full G-PPL；Times New Roman、无标题、图例白底黑框。
- `figures/fig6_catv_alpha03.{pdf,png}` — α=0.3 同款。
- `tables/table1_main_results.{md,tex}` — 主结果汇总（同口径）。
- `tables/table2_catv_alpha.{md,tex}` — CATV × alpha × 口径，含逐样本配对 ratio/p。
- `tables/table3_sparse_quality.{md,tex}` — predictor/随机/dense 对照（n=4）+ dense vs sparse（n=500）。
- `tables/table4_pos_adaptation.{md,tex}` — 位置适配判据表。
- `tables/table5_micro_scans.{md,tex}` — 覆盖度微扫（k 扫描 + L×D 先导）。
- `tables/table6_catv_r44_r74.{md,tex}` — CATV 三口径（off / on 同口径 / on 部署）在 r44/r74 的实际值与相对 r0 下降率。
- `tables/table7_d_comparison.{md,tex}` — D 对照（D16/D22/D32，统一 8 轮窗口；**以训练前共享基线 33.6653 起算**，行内保留各臂 own-r0 备注）。
- `data/*.csv` — 上述图表底层 tidy 数据（`arm_points.csv` 为全部曲线点）。

## 口径与数据来源

- 运行均在 AutoDL 1×RTX 4090：ZOO+LoRA、fp32 + `sdpa_prune`、`sparse=0.4`、
  `--predictor step5_a01a03_clients_causal`、c=30 串行 FedAvg、16k、a03/a01。
- 曲线取 `metrics.jsonl` 的 eval 事件（token 加权 `full_ppl_token`/`answer_ppl_token`）；
  r0 锚点与 ON 臂同口径点取 `temp/eval_*.json` 事后评测（脚本内 `PERSAMPLE`/`R0_*` 注册表）。
- 墙钟小时：每个 run block `config.json` mtime → 最后一个 `adapter_round*` mtime；链式
  实验为各 block 之和（不含 block 间探测间隙），与 ailog 中的约数略有出入属定义差异。
- PPL/h = (full PPL 起点 − 终点) / 墙钟小时；F3 右图统一 0–29 轮窗口。
- fig5/fig6 按用户指定样式生成：Times New Roman（serif）、无标题、**图例**白底黑框、
  α 作为图例标题；表中 α 一律用希腊字母（Markdown 用 `α`，TeX 源用 `$lpha$`）。
- 主运行目录：`logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip`、`zoo_k24_d16_lr0.30_r30`、
  `zoo_k24_d16_r60_cont_lr0.10`、`zoo_k24_d32[_catv]_lr0.30_r30`、
  `a03_zoo_k24_d32[_catv]_lr0.10_cont15{,b,c}`、`a01_zoo_k24_d32[_catv]_lr0.30_r30`、
  `a01_zoo_k24_d32[_catv]_lr0.10_cont15`。
- 对账：脚本内置 ailog 已发表关键值断言，运行输出 32/32 PASS 才写本文件。

## 结论摘要

- 覆盖度是 ZOO 的主杠杆：k=4→24 使每轮 full-PPL 降幅约提高一个数量级，
  0–29 轮速率 k4 0.019、D16 0.101、D32 0.087、D32+CATV 0.094、a01 0.084 PPL/h。
- CATV 的收益依赖 alpha：a03（0.3）同口径全程领先约 0.20 PPL 且 answer 早期显著改善
  （r29 answer ratio 0.83，p=5e-31）；a01（0.1）训练效应不显著（r29 full p=0.97）。
- a03 部署口径（训练+评测均带掩码）在 r64–r69 打平（r74 full ratio 1.0001，p=0.65），
  同口径仍领先 0.20；answer 的同口径优势在长训后反转（r74 ratio 1.13，p=6.7e-8），
  部署口径残余 answer 优势（0.93）主要来自评测掩码本身。
- 位置适配：16k G-PPL 398.8 → 31.4（B）→ 18.1（C），2k 无回归，选 C。

## 已知 caveat（写盘时保留，勿在正文中抹平）

- T3a 的 predictor/random/dense 对照只有 n=4（历史小样本，且未加载位置适配），仅作支持行；
  需要正式口径可用 GPU 空闲时以 500 样本重评（当前 GPU 被 BERT 链占用）。
- pos_lora 训练曲线与权重不在本机（cloud A40 产物），T4/F4 用 `posemb_step1/MANIFEST.json`
  的终值；本机仅有 pos_only 的 metrics。
- 云端 A40 的 BP 对照与 4 卡 alpha×CATV 全协议消融未纳入本包（按用户决定暂不做）。
- 260929 审计前的历史 ZOO run 与 lr clamp bug run 已排除；`zoo_k4_sparse_causal_a03`
  （lr≈9e-6）与 `..._step5`（lr=1e-4 被 clamp）不作结果使用。
- clipped 列是 30 client-轮累计被 `--delta-clip` 裁剪的数量（按 3×median 探测标定），
  表示稀疏路径 FD 噪声水平，不是失败计数。

## 复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY reports/opt/make_report.py            # 全量 + 对账闸门
$PY reports/opt/make_report.py --only figures
$PY reports/opt/make_report.py --only tables
$PY reports/opt/make_report.py --no-check # 仅调试用
```

字体：fig5/fig6 指定 Times New Roman，本机经
`apt-get install -y ttf-mscorefonts-installer` 安装（华为云镜像）；换机缺失时脚本按
`Times New Roman → Nimbus Roman → Liberation Serif → STIXGeneral → DejaVu Serif` 自动回退。

## 对账结果

- 断言：43/43 PASS
