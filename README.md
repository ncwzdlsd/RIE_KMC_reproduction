# CeOₓ-Ir 时间分辨动力学蒙特卡洛

本项目实现 Shi 等人在 *Science* 387 (2025) 补充材料中描述的 CeO₂ 晶格气体
KMC 模型，并在相同物理时间点比较有、无超声条件。

## 事件模型

两组条件共享五个可逆反应：Ce 吸附/脱附、O 吸附/脱附、Ir 离子在固液界面
吸附/脱附、Ir 离子沿载体或已锚定 Ir 团簇扩散，以及 Ir 离子还原/金属 Ir 氧化。超声组额外
包含论文定义的界面腐蚀事件：随机选择界面中心，1 nm 球内的每个表面 Ce/O 原子
以 10% 概率溶解。

每个当前纳米颗粒-溶液界面 Ce/O 位点都对应一个候选 KMC 超声事件。程序用
n-fold-way 将这些等速事件聚合：总倾向等于“每界面位点倾向 × 当前界面位点数”，
事件被选中后再等概率确定中心。这与显式列出全部中心严格等价，不是给整个颗粒设置
一个与表面积无关的恒定总事件率。补充材料未公开每界面位点倾向，因此该值仍需标定。

溶液采用充分混合的隐式表示，不在 20 nm 空盒中放置显式溶质原子。Ce/O 液相仍由
化学势和超声溶出记账控制；Ir 则使用守恒的有限前驱体储库。晶格只表示固态 CeOₓ
与附着 Ir；吸附仅发生在
溶液可达且与载体或已锚定 Ir 接触的位点，避免在液相中产生虚假的 Ir 团簇。
超声随机移除的 Ce/O 同时进入过量溶质记账；其累计溶出量作为宏观反应浴的浓度
信号，将有效 Ce/O 化学势从低浓度基线逐渐提高，最高限制为论文考察的 -0.60 eV。
因此随机腐蚀后会发生再吸附和表面重构，而不是只删除载体原子。

Ir 储库默认按补充材料 Table S9 的 RIE-Ir/CeOₓ ICP-OES 组成缩放：Ir 为
15.60 wt%，Ce 为 74.96 wt%，对应 `Ir/Ce = 0.15170` 的原子数比。标准 5 nm
初始载体含 1632 个 Ce 原子，因此每个条件从相同的 248 个 Ir 前驱体原子开始。
Ir 吸附消耗一个储库原子，Ir 离子脱附归还一个；吸附速率乘以储库剩余比例，耗尽后
自动变为零。高级入口可用 `--ir-precursor-atoms` 覆盖总投料，但一般无需修改。

Ir形貌采用论文五类反应内的“扩散后还原”机制：离子态Ir的初始扩散频率因子为
`2.0e-2 s^-1`，还原频率因子为`5.0e-4 s^-1`，使Ir离子在还原冻结前能够沿界面
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

## 1. 参数标定

补充材料没有公开完整动力学参数。先用 Table S3 的相对粒径曲线和 Table S5 的
Ce 溶解曲线拟合 Ce/O 交换时间尺度、共享化学势和每界面位点超声事件倾向：

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

- `trajectory.xyz`：0、5、30、60、120、180 min 的有/无超声并排轨迹；
- `snapshots/*.xyz`：每个取样时间点可单独打开的有/无超声并排结构；
- `metrics.csv`：相同物理时间下的两组指标及差值，包含 Ir 储库、守恒误差、团簇数、
  最大团簇、平均Ir-Ir配位、回转半径和形状各向异性；
- `run_metadata.json`：参数、单位、事件计数和最终状态。

添加 `--keep-checkpoints` 时会额外保留两组最新检查点。OVITO 中直接打开
`trajectory.xyz`，按 `condition` 着色：`0` 为无超声，`1` 为有超声。

## 局部更新引擎

`local_kmc.py` 分别维护 Ce/O、Ir 界面交换、Ir 氧化还原和 Ir 表面扩散速率桶。
普通事件只刷新两层邻域；超声事件只刷新局部腐蚀区域。外部溶液连通性按
`--reconcile-every` 周期全局校正，避免封闭孔洞造成长期误差。
当溶液化学势随超声累计溶出量改变时，Ce/O 各配位速率桶会整体重算，但不重新
扫描全部位点。
Ir 交换桶保存满浓度下的本征速率；事件抽样时仅对吸附桶乘以实时储库余量比例，
因此储库变化不需要全局扫描，同时严格保持 `溶液 Ir + 晶格 Ir = 初始投料`。

20 nm 论文晶格包含 607,836 个位点。在当前环境中，完整局部引擎初始化约需
14 秒；实际运行时间取决于标定后的总速率、事件数和连通性校正频率。

## 模型边界

事件与论文参数的逐项映射见 [DFT_PARAMETER_MAPPING.md](DFT_PARAMETER_MAPPING.md)。
Table S3/S5 只能约束 Ce/O 时间尺度和超声腐蚀，不能证明唯一微观参数组。如果标定
目标函数无法通过阈值，说明当前单颗粒/隐式液相模型不足，应扩展多颗粒共享溶液储库
或多颗粒熟化模型，而不能强行把事件步数映射为分钟。
