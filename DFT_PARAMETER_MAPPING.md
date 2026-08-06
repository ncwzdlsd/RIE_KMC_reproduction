# DFT 参数映射

本文档记录 Shi 等人在 *Science* 387, 791–796 (2025)，DOI
`10.1126/science.adr3149` 中的参数与当前代码模型之间的映射。

## 已直接应用

| 物理量 | 论文值 | 代码参数 | 状态 |
| --- | ---: | --- | --- |
| Ce–O 单配位结合能 | 0.30 eV | `CeOxParameters.ce_o_binding_energy_ev` | 已应用 |

该数值现在统一定义在 `paper_parameters.py`，并作为
`CeOxParameters` 的默认值。此前数据类默认值为 0.50 eV，虽然三个主要运行脚本会显式覆盖为
0.30 eV，但直接使用数据类或新增入口时可能误用；此次修改消除了这一不一致。

## 论文设置，但不是 DFT 能量

温度 453 K、Ce/O 化学势 −0.60 eV、20 nm 模拟盒、5 nm 颗粒、S34
快照步数，以及 1 nm 超声腐蚀半径和 10% 溶解概率，也集中定义在
`paper_parameters.py`。集中管理的目的是记录来源，不表示这些数值均由 DFT 得到。

## 暂不能直接替换

当前可访问的论文记录没有提供能与本代码事件模型一一对应的以下 DFT 数值：

- Ir–Ir 和 Ir–O 最近邻结合能；
- Ir 离子化学势与还原自由能；
- Ir 吸附、脱附、扩散、还原和氧化势垒；
- 上述过程的频率因子。

因此 `run_ir_visualization.py` 中的 Ir 参数仍是可视化参数，没有冒充论文/DFT
参数。当前项目只复现球形 CeOₓ 颗粒，并使用各向同性 Ce–O 配位能；不能把其他类型的
表面能直接填入 Ir–O 结合能或扩散势垒。

## 来源与限制

- 论文：W. Shi et al., “Ultrastable supported oxygen evolution electrocatalyst
  formed by ripening-induced embedding,” *Science* 387, 791–796 (2025),
  DOI `10.1126/science.adr3149`。
- 论文正文说明其模拟采用 DFT-based KMC；本项目当前仅保留球形颗粒部分。
- Science 补充材料下载端当前受到访问验证限制；论文引用的 Dryad DOI 在公共 API 中也未
  返回可查看的数据文件。因此，本轮没有从无法核实的二手资料抄录 Ir 数值。
