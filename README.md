# 超声前后 CeOₓ 载体上 Ir 嵌入状态的动力学蒙特卡洛

本项目实现 Shi 等人在 *Science* 387 (2025) 补充材料中描述的 CeO₂-Ir 晶格气体 KMC 模型，唯一复现目标是在相同物理时间点比较无超声和施加超声时，Ir 颗粒位于载体表面还是嵌入载体。

为隔离超声效应，程序把同一个随机初始构型复制为两支：一支不施加超声，另一支施加超声，并在相同物理时间点比较；因此这里的“前后”是匹配条件对照，不是把同一个已经演化过的状态再顺序施加超声。

论文来源：Shi 等，*Science* 387 (2025)，DOI `10.1126/science.adr3149`；模型依据补充材料 “Kinetic monte carlo (KMC) simulations” 部分实现。

## 事件模型

两组条件共享五个可逆反应：Ce 吸附/脱附、O 吸附/脱附、Ir 离子在盒子边缘 M 位点吸附/脱附、Ir 离子沿相邻空 M 位点扩散，以及 Ir 离子还原/金属 Ir 氧化。超声组额外包含论文定义的界面腐蚀事件：随机选择界面中心，1 nm 球内的每个表面 Ce/O 原子以 10% 概率溶解。

五个可逆反应与代码事件的对应关系如下：

| 补充材料反应 | 正向事件 | 反向事件 |
| --- | --- | --- |
| `Ce(l) + *@M ⇌ Ce*@M` | `CE_ADSORPTION` | `CE_DESORPTION` |
| `O(l) + *@O ⇌ O*@O` | `O_ADSORPTION` | `O_DESORPTION` |
| `Ir(ion,l) + *@M ⇌ Ir(ion)*@M` | `IR_ION_ADSORPTION` | `IR_ION_DESORPTION` |
| `Ir(ion)*@M + *@M ⇌ *@M + Ir(ion)*@M` | `IR_ION_DIFFUSION` | 反向扩散由同一事件表示 |
| `Ir(ion)*@M ⇌ Ir*@M` | `IR_REDUCTION` | `IR_OXIDATION` |

每个当前纳米颗粒-溶液界面 Ce/O 位点都贡献一个独立超声条件事件的危险率。超声采用独立泊松时钟，不进入化学 KMC 反应目录，也不增加 KMC 步数；总频率等于“每界面位点频率 × 当前界面位点数”，触发后再等概率确定中心。当前诊断频率为 `1.0e-5 s^-1/位点`，是原设置的 5 倍。补充材料未公开该频率，因此仍需标定。

溶液本体采用充分混合的隐式表示。有、无超声组的 Ce/O 巨正则化学势均固定为 `-0.60 eV`。超声只作为独立外界条件触发显式表面腐蚀；脱离原子只改变晶格占位与溶解统计，不反馈到 Ce/O 化学势。Ir 使用守恒的有限前驱体储库，并按论文描述从盒子边缘吸附到 M 位点，再沿整个溶液可达的空 M 位点网络扩散。Ir 只有到达载体或已有金属 Ir 附近才能还原，因此盒内迁移态是 `Ir(ion)`，不会直接形成液相金属团簇。

Ir 储库把“前驱体投料”和“最终负载目标”分开处理。补充材料 Table S9 的最终 RIE-Ir/CeOₓ 组成为 Ir 15.60 wt%、Ce 74.96 wt%，对应 `Ir/Ce = 0.15170`；标准 5 nm 初始载体含约 1632 个 Ce 原子，最终连接载体的目标因而约为 248 个 Ir。默认前驱体库存现扩展为约 600 个，等价于约 `248/600 = 41.3%` 的名义捕获率；600 是诊断投料设置，并非论文公开值。Ir 吸附消耗一个储库原子，Ir 离子脱附归还一个；耗尽后吸附自动停止。高级入口仍可用 `--ir-precursor-atoms` 覆盖总投料。

Ir 形貌采用论文五类反应内的“扩散后还原”机制：离子态 Ir 的诊断扩散频率因子为 `5.0 s^-1`，异相还原频率因子为 `0.5 s^-1`。前者由已完成标准结果的约 16.5% 捕获率和 600 原子投料达到 248 原子目标所需的约 41.3% 捕获率作一阶反推，仍属于论文未公开、需要完整 20 nm 复核的初值。O 位点的局部能量同时计入 Ce-O 和 Ir-O 邻接，因此 CeOₓ 能够在 Ir 周围继续生长。金属 Ir 表面扩散不在论文列出的五类反应中，当前没有擅自加入。

所有正式速率文件使用 `s^-1`。KMC 时钟按 `dt = -ln(u) / sum(k_i)` 推进，不再使用事件数代替实验时间。

## XPK 加速

`--method xpk` 启用 XPK 加速。实现严格采用 XPK 的两阶段结构：在 Ce/O、金属 Ir 和各物种数目固定的化学状态下，只运行 `IR_ION_DIFFUSION` 以采样扩散系综；随后对 Ce/O 交换、Ir 吸附/脱附、Ir 氧化还原和独立超声条件事件的倾向率取系综平均，并以 `dt = -ln(u) / <sum(k_slow)>` 在化学空间推进。扩散采样不推进物理时间、不进入论文反应事件计数，也不会修改任何速率、势垒或化学势。

XPK 比较论文在其 40×40 加氢测试模型中，每个插值点运行 `1.6×10^7` 个纯扩散步，以 `0.04` 覆盖度间隔做线性插值。这些数值会写入元数据作为论文参照，但它们属于该二维测试模型的数值收敛设置，不是本 CeOₓ-Ir 模型的物理参数。本项目的输出包含 CeOₓ 形貌、金属 Ir 位置和包埋关系；把覆盖度相同但形貌不同的状态强行插值会改变复现模型。因此当前实现在每个实际化学状态上直接采样，不跨不同形貌插值。这保留了 XPK 的扩散系综平均和化学空间演化，并消除了额外的覆盖度插值误差。

`--xpk-equilibration-sweeps`、`--xpk-samples` 和 `--xpk-decorrelation-sweeps` 只控制统计收敛，不是物理参数。正式结果必须增大这些值做收敛检查；每次运行采用的值和扩散采样步数都会写入 `run_metadata.json` 与 `metrics.csv`。当前默认仍为 `--method kmc`，只有 XPK 对照达到统计收敛后才应把它用于正式复现，避免未经验证就改变论文结果。

参数文件若给出非 `-0.60 eV` 的 Ce/O 化学势或非零超声化学势增量，主程序现在会直接拒绝运行，而不是静默覆盖输入或在反应过程中改变化学势。

## 论文参数与代码映射

| 物理量 | 论文值 | 代码常量/参数 |
| --- | ---: | --- |
| Ce-O 结合能（具体形貌模拟设置） | 0.30 eV | `DFT_CE_O_BINDING_ENERGY_EV` |
| Ir-Ir 平均结合能 | 0.32 eV | `DFT_IR_IR_BINDING_ENERGY_EV` |
| Ir-O 界面值（论文符号） | -0.05 eV | `PAPER_REPORTED_IR_O_BINDING_ENERGY_EV` |
| Ir-O 稳定化强度（内部符号） | +0.05 eV | `DFT_IR_O_BINDING_ENERGY_EV` |
| 温度 | 453 K | `PAPER_TEMPERATURE_K` |
| 模拟盒 | 20 × 20 × 20 nm³ | `PAPER_BOX_NM` |
| 初始 CeO₂ 颗粒直径 | 5 nm | `PAPER_PARTICLE_DIAMETER_NM` |
| 超声腐蚀半径 | 1 nm | `PAPER_SONICATION_RADIUS_NM` |
| 局部溶解概率 | 10% | `PAPER_DISSOLUTION_PROBABILITY` |
| 独立超声条件频率 | 未公开，当前诊断值 `1.0e-5 s^-1/位点` | `sonication_event_rate_s` |
| 超声条件 Ce/O 化学势增量 | 固定为 0；两组均为 -0.60 eV | `sonication_chemical_potential_shift_ev` |
| RIE 催化剂 Ir 质量分数 | 15.60 wt% | `PAPER_RIE_IR_MASS_PERCENT` |
| RIE 催化剂 Ce 质量分数 | 74.96 wt% | `PAPER_RIE_CE_MASS_PERCENT` |
| 由 ICP-OES 换算的 Ir/Ce 原子比 | 0.15170 | `PAPER_RIE_IR_TO_CE_ATOM_RATIO` |

补充材料在 DFT 段落还报告 Ce-O 平均结合能约 0.33 eV；其后描述具体粗糙度/超声模拟时明确采用 0.30 eV，因此当前复现使用后者。

代码的局部结合表达式把正数作为稳定化强度，并在成键时从自由能变化中减去。论文的 Ir-O 界面值以 `-0.05 eV` 给出，因此进入该表达式前转换为 `+0.05 eV`；否则程序会错误地排斥 Ir-O 接触，使 Ir 远离载体向外枝化。O 吸附/脱附同时统计相邻 Ce 和 Ir，这是载体围绕 Ir 继续生长所需的界面项。

嵌入判定只针对与最大 CeOₓ 主载体连通的 Ir：仍处于溶液可达外表面的记为表面 Ir，不属于该外表面的记为嵌入 Ir。XYZ 文件中的 `embedded=0/1` 给出逐原子判定；`ir_embedding_comparison.csv` 汇总两种条件下的表面数量、嵌入数量和嵌入比例。

## 环境

- Python 3.10+
- NumPy
- OVITO（可选）

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy
```

## Python 文件说明

| 文件 | 类型 | 功能 |
| --- | --- | --- |
| `run_traditional_kmc.py` | 传统 KMC 直接入口 | 直接运行标准 180 min 逐跳 KMC，不需要填写命令行参数。 |
| `run_xpk_optimized.py` | XPK 直接入口 | 直接运行标准 180 min XPK 模拟，不需要填写命令行参数。 |
| `run_sonication_comparison.py` | 高级公共入口 | 两个直接入口共用的运行与输出逻辑；需要覆盖几何、参数或采样设置时才直接调用。 |
| `xpk_kmc.py` | XPK 加速引擎 | 在固定化学状态下执行 Ir 纯扩散采样，对慢事件倾向率取系综平均并在化学空间推进。 |
| `local_kmc.py` | 基准 KMC 引擎 | 实现完整局部更新、n-fold-way 逐跳抽样、物理时钟、有限 Ir 储库、独立超声条件时钟、检查点及形貌统计。 |
| `kinetic_parameters.py` | 参数模块 | 定义可序列化的动力学参数集，读取/写入标定文件，并转换为 Ce/O、Ir 和超声子参数。 |
| `paper_parameters.py` | 论文常量 | 集中保存论文或补充材料可追溯的温度、能量、盒尺寸、时间点、超声参数和 Ir/Ce 比例。 |
| `constants.py` | 基础枚举 | 定义晶格位点类型、Ce/O/Ir 占据状态和玻尔兹曼常数。 |
| `lattice_build.py` | 晶格模块 | 构建萤石 CeO₂ 晶格、两类格点、邻接表、坐标及盒边缘储库边界。 |
| `generation.py` | 几何与输出 | 初始化/粗糙化球形 CeOₓ，计算溶液连通性和外表面，识别载体连接 Ir，并写出 XYZ。 |
| `ceox_events.py` | Ce/O 参数 | 定义 Ce/O 事件类型、参数对象和基于自由能变化的跃迁速率公式。 |
| `ir_events.py` | Ir 参数与局域能 | 定义 Ir 事件类型和参数，计算 Ir–Ir、Ir–O 局域结合、异相还原条件及活化速率。 |
| `sonication_events.py` | 超声参数 | 定义超声腐蚀事件类型，以及 1 nm 作用半径、10% 溶解概率和单位界面位点倾向。 |
| `preparation/calibrate_parameters.py` | 一次性标定入口 | 直接运行即可按正式默认配置完成参数标定，并写出参数文件及标定报告。 |
| `preparation/calibration.py` | 一次性标定算法 | 使用补充材料 Table S3/S5 的粒径和溶解数据，在两组化学势固定为 `-0.60 eV` 且超声化学势增量固定为零时，拟合 Ce/O 时间尺度和独立超声倾向。 |

主入口为 `run_sonication_comparison.py`；其余模块文件不应单独运行。

## 1. 参数标定（可选）

补充材料没有公开完整动力学参数。准备阶段可运行一次 `preparation/calibrate_parameters.py`；程序默认执行 40 次搜索、每个候选参数运行 5 个随机重复，并用 Table S3 的相对粒径曲线和 Table S5 的 Ce 溶解曲线，在两组均固定为 `-0.60 eV` 的 Ce/O 化学势下拟合 Ce/O 交换时间尺度和每界面位点的独立超声条件频率。

仍需显式校准的输入包括：

- Ce/O 与 Ir 的事件频率因子；
- Ir 离子化学势、还原自由能；
- 吸附、脱附、扩散、还原和氧化势垒；
- 每个纳米颗粒-溶液界面位点的超声腐蚀事件倾向。

```powershell
py -3 preparation/calibrate_parameters.py
```

也可以在 IDE 中直接运行该文件，不需要填写命令行参数。结果始终默认写入项目根目录的 `calibrated_parameters.json` 和 `calibrated_parameters_report.json`，供 comparison 程序自动加载。命令行选项仅用于需要覆盖默认配置的高级场景。

生成：

- `calibrated_parameters.json`：带单位、标定范围和目标函数的参数文件；
- `calibrated_parameters_report.json`：接受的搜索过程和最终误差。

只有目标函数低于接受阈值的文件才会标记为 `calibrated: true`。当前实验表格不足以唯一标定 Ir 各事件速率，因此 Ir 参数仍在文件中明确标注为初始估计；若要定量解释 Ir 粒径和包埋比例，还需加入对应时间序列。

当前 `5.0 s^-1` 的单跳扩散频率与 `0.5 s^-1` 的载体接触还原频率保留了显式逐跳输运。600 原子储库的吸附活度按 `剩余前驱体/248 原子目标` 计算，因此初始活度约为 2.42，而不再因除以自身库存错误地归一化为 1。该尺度仍需完整标准盒结果校准。

## 2. 运行 0-180 min 对照

标定完成后，可直接在 IDE 中分别运行下面两个文件。两者都会自动加载合格的 `calibrated_parameters.json`，并使用 20 nm 盒、5 nm 初始载体、0/5/30/60/120/180 min 取样点和自动缩放的有限 Ir 储库完成标准对照，不需要输入任何额外参数：

```powershell
py -3 run_traditional_kmc.py
py -3 run_xpk_optimized.py
```

传统 KMC 结果写入带时间戳的 `kmc_output/traditional_kmc_180min_*` 目录；XPK 结果写入 `kmc_output/xpk_180min_*`。若需要提高 XPK 采样强度等高级控制，仍可使用公共入口：

```powershell
py -3 run_sonication_comparison.py `
  --method xpk `
  --xpk-equilibration-sweeps 4 `
  --xpk-samples 16 `
  --xpk-decorrelation-sweeps 1
```

脚本只会自动加载标记为 `calibrated: true` 的 `calibrated_parameters.json`；失败或过期的标定文件会被忽略，并改用内置诊断初值且打印警告。显式传入 `--parameter-file` 仍可检查任意候选文件；需要强制使用通过验收的参数时添加 `--require-calibrated`。

每次运行都会在 `kmc_output` 下原子式创建独立目录，例如 `comparison_180min_20260809_153012_123456`。时间戳精确到微秒，程序绝不会复用或清空已有运行目录。`--output` 只作为目录名前缀使用，实际目录同样自动追加时间戳。

紧凑晶格：

```powershell
py -3 run_sonication_comparison.py `
  --target-times-min 0,5,30,60,120,180 `
  --box-nm 4.8 `
  --particle-diameter-nm 4.0 `
  --output kmc_output/time_comparison
```

需要保留续算检查点时添加 `--keep-checkpoints`。所有输出目录都会自动追加时间戳，不会覆盖已有结果。

## 输出

每次运行默认保留以下三套轨迹及其逐时间点快照。标准设置下，每套均包含 0、5、30、60、120、180 min 六帧，且每帧都把无超声组和有超声组并排写入同一个 XYZ 文件：

- `trajectory.xyz` 与 `snapshots/snapshot_<序号>_<时间>min.xyz`：结构轨迹及对应的单帧文件，只显示最大 CeOₓ 连通主晶体以及与该主晶体连接的 Ir；脱落的 Ce/O/Ir 碎片和仍在盒内迁移的 Ir 不混入主颗粒图像；
- `trajectory_ir_emphasis.xyz` 与 `snapshots_ir_emphasis/snapshot_<序号>_<时间>min.xyz`：相同物理状态的 Ir 强调版本，写入 OVITO 可识别的 `Radius` 和 `Transparency` 属性，使 Ce/O 半透明并放大 Ir，不改变任何 KMC 状态；
- `trajectory_ir_only.xyz` 与 `snapshots_ir_only/snapshot_<序号>_<时间>min.xyz`：只保留与主晶体连接 Ir 的诊断版本，用于直接清点团簇和检查背面/包埋 Ir；
- `ir_embedding_comparison.csv`：最精简的目标结果，逐时间点对比无超声/有超声的载体连接 Ir、表面 Ir、嵌入 Ir、嵌入比例及超声造成的差值；
- `metrics.csv`：相同物理时间下的两组指标及差值，包含 Ir 储库、守恒误差、团簇数、盒内未连接 Ir、负载/嵌入 Ir、相对 Table S9 目标的主晶体 Ir 到达比例、`detached_support_atoms`、XPK 扩散采样步数与化学空间步数，以及主晶体径向离散度、轴向尺寸比和形状各向异性；
- `run_metadata.json`：参数、单位、事件计数、最终状态和 Ir 嵌入趋势检查。

嵌入趋势检查关注超声后的 Ir 嵌入比例是否高于无超声对照，同时保留 Ir 库存守恒和首个非零时间点可见性的基本一致性检查。

默认情况下，用于组装上述成品文件的 `_raw` 临时目录会在运行结束时删除。添加 `--keep-checkpoints` 后，程序会保留 `_raw/no_sonication/checkpoint_latest.npz` 和 `_raw/sonication/checkpoint_latest.npz` 两组最新检查点，并删除其中已经汇总到成品轨迹的原始快照。OVITO 中可直接打开任一 `trajectory*.xyz`；按 `condition` 着色时，`0` 为无超声，`1` 为有超声。

## 局部更新引擎

`local_kmc.py` 分别维护 Ce/O、盒边缘 Ir 交换、Ir 氧化还原和 Ir 离子扩散速率桶，并另外维护不属于这些速率桶的超声条件时钟。普通反应只刷新两层邻域；超声条件事件只刷新局部腐蚀区域且不增加 KMC 步数。外部溶液连通性按 `--reconcile-every` 周期全局校正，避免封闭孔洞造成长期误差。有、无超声条件的 Ce/O 化学势均固定为 `-0.60 eV`，超声脱离不反馈化学势。Ir 交换桶保存满浓度下的本征速率；事件抽样时仅对吸附桶乘以实时储库余量比例，因此储库变化不需要全局扫描，同时严格保持 `溶液 Ir + 晶格 Ir = 初始投料`。

20 nm 论文晶格包含 607,836 个位点。在当前环境中，完整局部引擎初始化约需 14 秒；实际运行时间取决于标定后的总速率、事件数和连通性校正频率。

## 模型边界

Table S3/S5 只能约束 Ce/O 时间尺度和超声腐蚀，不能证明唯一微观参数组。如果标定目标函数无法通过阈值，说明当前单颗粒/隐式液相模型不足，应扩展多颗粒共享溶液储库或多颗粒熟化模型，而不能强行把事件步数映射为分钟。
