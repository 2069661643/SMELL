# Ailog 261013-161052 — Phase N: OPT 结果包增补（CATV on/off 单图 ×2 + r44/r74 端点表）

> 承接 `261013-155317`（OPT 结果包）。用户追加指定三项：
> ① α=0.1 的 CATV on/off full-PPL 图；② α=0.3 同款；③ a01/a03 × {on, on-deploy, off} 在
> **仅 r44/r74** 的 PPL 表（实际值 + 相对 r0 下降率）。
> 图样式要求：**Times New Roman、无标题、注释白底黑框、α 用希腊字母（TeX `\alpha`）**。

## 变更

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:58 | 安装字体 | `ttf-mscorefonts-installer`（apt, 华为云镜像） | Times New Roman 可用（此前系统只有 DejaVu/STIX） |
| 16:02 | 新增 | `make_report.py`：`_times_plt()`、`fig5_catv_onoff()`、`table6_endpoints()` | serif 栈 Times→Nimbus→Liberation→STIX；mathtext=stix |
| 16:05 | 生成 | `figures/fig5_catv_alpha01.{pdf,png}`、`figures/fig6_catv_alpha03.{pdf,png}` | 单面板 off(solid)/on(dashed)，注释框 `square,pad=0.4` 白底黑框 |
| 16:05 | 生成 | `tables/table6_catv_r44_r74.{md,tex}` + `data/table6_catv_r44_r74.csv` | 6 行×{r44,r74 实际值,下降率}；α 列用 `α`/`$\alpha$` |
| 16:08 | 更新 | `reports/opt/README.md` | 产物索引 + 样式口径说明 |

## Table 6 数值（现算，full G-PPL）

| α | 口径 | r44 | Δ | r74 | Δ |
|---|---|---|---|---|---|
| 0.1 | off | 31.1653 | −7.04% | — | — |
| 0.1 | on (same) | 31.1572 | −7.06% | — | — |
| 0.1 | on (deploy) | 31.3749 | −7.51% | — | — |
| 0.3 | off | 31.1189 | −7.19% | 30.8090 | −8.11% |
| 0.3 | on (same) | 30.9369 | −7.73% | 30.6095 | −8.71% |
| 0.3 | on (deploy) | 31.1563 | −8.20% | 30.8118 | −9.22% |

- a01 链只到 r44（r74 不存在，表中以 `-` 标出，脚注说明）；Δ 对各自 r0 锚点
  （同口径 33.5296/33.5253；部署 33.9393/33.9211）。
- fig5/fig6 为同口径（无掩码）两曲线；部署数只在表中出现。

## 验证

- 全量重跑：32/32 对账 PASS（未动断言）；`py_compile` 通过。
- 图目检：Times New Roman 生效、无标题、注释框白底黑框、`α=0.1/0.3` 正确渲染；
  图内数字与 table1/2 一致。
