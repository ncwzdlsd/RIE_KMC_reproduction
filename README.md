# CeOₓ-Ir 时间分辨动力学蒙特卡洛

本项目实现 Shi 等人在 *Science* 387 (2025) 补充材料中描述的 CeO₂ 晶格气体
KMC 模型，并在相同物理时间点比较有、无超声条件。

## 事件模型

两组条件共享五个可逆反应：Ce 吸附/脱附、O 吸附/脱附、Ir 离子在盒子边缘 M 位点
吸附/脱附、Ir 离子沿相邻空 M 位点扩散，以及 Ir 离子还原/金属 Ir 氧化。超声组额外
包含论文定义的界面腐蚀事件：随机选择界面中心，1 nm 球内的每个表面 Ce/O 原子
以 10% 概率溶解。

每个当前纳米颗粒-溶液界面 Ce/O 位点都对应一个候选 KMC 超声事件。程序用
n-fold-way 将这些等速事件聚合：总倾向等于“每界面位点倾向 × 当前界面位点数”，
事件被选中后再等概率确定中心。这与显式列出全部中心严格等价，不是给整个颗粒设置
一个与表面积无关的恒定总事件率。补充材料未公开每界面位点倾向，因此该值仍需标定。

溶液本体采用充分混合的隐式表示。Ce/O 液相采用固定为 `-0.60 eV` 的巨正则化学势；Ir 使用
守恒的有限前驱体储库，并按论文描述从盒子边缘吸附到 M 位点，再沿整个溶液可达的
空 M 位点网络扩散。Ir 只有到达载体或已有金属 Ir 附近才能还原，因此盒内迁移态是
`Ir(ion)`，不会直接形成液相金属团簇。
超声随机移除、普通脱附和再吸附仍会进入诊断记账，但这些数量不再反馈改变 Ce/O
事件速率。因此两组的 Ce/O 化学势始终相同，唯一额外机制是超声界面腐蚀事件。

Ir 储库默认按补充材料 Table S9 的 RIE-Ir/CeOₓ ICP-OES 组成缩放：Ir 为
15.60 wt%，Ce 为 74.96 wt%，对应 `Ir/Ce = 0.15170` 的原子数比。标准 5 nm
初始载体含 1632 个 Ce 原子，因此每个条件从相同的 248 个 Ir 前驱体原子开始。
Ir 吸附消耗一个储库原子，Ir 离子脱附归还一个；吸附速率乘以储库剩余比例，耗尽后
自动变为零。高级入口可用 `--ir-precursor-atoms` 覆盖总投料，但一般无需修改。

Ir形貌采用论文五类反应内的“扩散后还原”机制：离子态Ir的初始扩散频率因子为
`2.0 s^-1`，还原频率因子为`5.0e-3 s^-1`，使Ir离子在还原冻结前能够穿过盒内并沿界面
寻找较高Ir配位位置。O位点的局部能量同时计入Ce-O和Ir-O邻接，因此CeOₓ能够在
Ir周围继续生长。金属Ir表面扩散不在论文列出的五类反应中，当前没有擅自加入。

所有正式速率文件使用 `s^-1`。KMC 时钟按
`dt = -ln(u) / sum(k_i)` 推进，不再使用事件数代替实验时间。

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
| `run_comparison.py` | 推荐入口 | 一键启动论文几何下 0–180 min 的有/无超声对照，并为每次运行创建独立输出目录。 |
| `run_sonication_comparison.py` | 高级入口 | 解析高级参数、构建相同初态的两组模拟、按物理时间取样，并写出轨迹、快照、指标和元数据。 |
| `local_kmc.py` | 主引擎 | 实现完整局部更新、n-fold-way 事件抽样、物理 KMC 时钟、有限 Ir 储库、超声事件、检查点及形貌统计。 |
| `kinetic_parameters.py` | 参数模块 | 定义可序列化的动力学参数集，读取/写入标定文件，并转换为 Ce/O、Ir 和超声子参数。 |
| `paper_parameters.py` | 论文常量 | 集中保存论文或补充材料可追溯的温度、能量、盒尺寸、时间点、超声参数和 Ir/Ce 比例。 |
| `constants.py` | 基础枚举 | 定义晶格位点类型、Ce/O/Ir 占据状态和玻尔兹曼常数。 |
| `lattice_build.py` | 晶格模块 | 构建萤石 CeO₂ 晶格、两类格点、邻接表、坐标及盒边缘储库边界。 |
| `generation.py` | 几何与输出 | 初始化/粗糙化球形 CeOₓ，计算溶液连通性和外表面，识别载体连接 Ir，并写出 XYZ。 |
| `ceox_events.py` | Ce/O 参数 | 定义 Ce/O 事件类型、参数对象和基于自由能变化的跃迁速率公式。 |
| `ir_events.py` | Ir 参数与局域能 | 定义 Ir 事件类型和参数，计算 Ir–Ir、Ir–O 局域结合、异相还原条件及活化速率。 |
| `sonication_events.py` | 超声参数 | 定义超声腐蚀事件类型，以及 1 nm 作用半径、10% 溶解概率和单位界面位点倾向。 |
| `calibrate_parameters.py` | 标定入口 | 从命令行启动参数标定并写出参数文件及标定报告。 |
| `calibration.py` | 标定算法 | 使用补充材料 Table S3/S5 的粒径和溶解数据，在固定 −0.60 eV 下拟合 Ce/O 时间尺度与超声倾向。 |
| `run_s34_reproduction.py` | 补充图入口 | 按补充材料 Fig. S34 的 20 nm 盒、5 nm 颗粒和固定事件步数运行 Ce/O-only 复现，支持续算。 |
| `fast_ceox_kmc.py` | S34 专用引擎 | 为 S34 的百万级格点/事件计算提供不含 Ir 与超声账本的轻量局部速率桶和检查点。 |

正常生成对照结果只需运行 `run_comparison.py`；其余入口用于标定、调参或独立复现
补充图。模块文件不应单独运行。

## 1. 参数标定

补充材料没有公开完整动力学参数。先用 Table S3 的相对粒径曲线和 Table S5 的
Ce 溶解曲线，在固定 `-0.60 eV` 化学势下拟合 Ce/O 交换时间尺度和每界面位点超声事件倾向：

```powershell
py -3 calibrate_parameters.py `
  --iterations 40 `
  --replicates 5 `
  --output calibrated_parameters.json
```

生成：

- `calibrated_parameters.json`：带单位、标定范围和目标函数的参数文件；
- `calibrated_parameters_report.json`：接受的搜索过程和最终误差。

只有目标函数低于接受阈值的文件才会标记为 `calibrated: true`。当前实验表格不足以
唯一标定 Ir 各事件速率，因此 Ir 参数仍在文件中明确标注为初始估计；若要定量解释
Ir 粒径和包埋比例，还需加入对应时间序列。

## 2. 运行 0-180 min 对照

直接运行 `run_comparison.py` 即使用 20 nm 盒、5 nm 初始载体和自动缩放的有限 Ir
储库完成标准对照：

```powershell
py -3 run_comparison.py
```

每次运行都会在 `kmc_output` 下原子式创建独立目录，例如
`comparison_180min_20260809_153012_123456`。时间戳精确到微秒，程序绝不会复用或
清空已有运行目录。高级入口中的 `--output` 也只作为目录名前缀使用，实际目录同样
自动追加时间戳。

紧凑晶格：

```powershell
py -3 run_sonication_comparison.py `
  --parameter-file calibrated_parameters.json `
  --target-times-min 0,5,30,60,120,180 `
  --output kmc_output/time_comparison
```

论文的 20 nm 盒和 5 nm 初始颗粒：

```powershell
py -3 run_sonication_comparison.py `
  --parameter-file calibrated_parameters.json `
  --target-times-min 0,5,30,60,120,180 `
  --box-nm 20 `
  --particle-diameter-nm 5 `
  --keep-checkpoints `
  --output kmc_output/paper_geometry
```

未通过标定的参数会被正式入口拒绝。仅用于检查程序时可添加
`--allow-uncalibrated`。

## 输出

每次运行默认只保留：

- `trajectory.xyz`：0、5、30、60、120、180 min 的有/无超声并排轨迹，只显示与载体连接的 Ir；
- `snapshots/*.xyz`：每个取样时间点可单独打开的相同过滤结果；
- `metrics.csv`：相同物理时间下的两组指标及差值，包含 Ir 储库、守恒误差、团簇数、
  盒内未连接 Ir、负载 Ir、最大团簇、平均Ir-Ir配位、回转半径和形状各向异性；
- `run_metadata.json`：参数、单位、事件计数和最终状态。

添加 `--keep-checkpoints` 时会额外保留两组最新检查点。OVITO 中直接打开
`trajectory.xyz`，按 `condition` 着色：`0` 为无超声，`1` 为有超声。

## 局部更新引擎

`local_kmc.py` 分别维护 Ce/O、盒边缘 Ir 交换、Ir 氧化还原和 Ir 离子扩散速率桶。
普通事件只刷新两层邻域；超声事件只刷新局部腐蚀区域。外部溶液连通性按
`--reconcile-every` 周期全局校正，避免封闭孔洞造成长期误差。
Ce/O 化学势固定后，各配位速率桶的速率在整次运行中保持不变；局部事件只增删桶成员。
Ir 交换桶保存满浓度下的本征速率；事件抽样时仅对吸附桶乘以实时储库余量比例，
因此储库变化不需要全局扫描，同时严格保持 `溶液 Ir + 晶格 Ir = 初始投料`。

20 nm 论文晶格包含 607,836 个位点。在当前环境中，完整局部引擎初始化约需
14 秒；实际运行时间取决于标定后的总速率、事件数和连通性校正频率。

## 模型边界

事件与论文参数的逐项映射见 [DFT_PARAMETER_MAPPING.md](DFT_PARAMETER_MAPPING.md)。
Table S3/S5 只能约束 Ce/O 时间尺度和超声腐蚀，不能证明唯一微观参数组。如果标定
目标函数无法通过阈值，说明当前单颗粒/隐式液相模型不足，应扩展多颗粒共享溶液储库
或多颗粒熟化模型，而不能强行把事件步数映射为分钟。
