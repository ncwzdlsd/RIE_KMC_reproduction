# CeOₓ / Ir 动力学蒙特卡洛复现

本项目使用 Python 实现晶格气体动力学蒙特卡洛（Kinetic Monte Carlo, KMC）模型，用于复现和演示 CeOₓ 载体的形貌演化、Ir 物种在载体表面的迁移/氧化还原，以及超声辅助腐蚀与熟化过程。

当前代码主要包含三类计算：

- 复现 *Science* 387 (2025) 补充材料图 S34 中公开描述的球形粗糙 CeO₂ 颗粒演化；
- 生成 Ir 吸附、扩散、还原和氧化过程的短程 OVITO 可视化轨迹；
- 对比有、无超声条件下的 CeOₓ–Ir 结构演化和 Ir 包埋程度。

> [!IMPORTANT]
> 论文未公开全部动力学参数。代码中的部分频率因子、Ir 动力学参数、表面粗糙度及超声事件速率是显式假设或可视化参数。S34 计算适合比较形貌趋势和事件步数下的结构，不应在未经标定时将 KMC 时间解释为真实物理时间。

## 环境要求

- Python 3.10 或更高版本
- NumPy
- OVITO（可选，用于查看 `.xyz` 轨迹）

建议在虚拟环境中运行：

```bash
python -m venv .venv
```

如果 Windows 中未配置 `python` 命令，可将下文命令中的 `python` 替换为 Python Launcher 的 `py`（例如 `py -m venv .venv`）。

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy
```

Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

## 快速开始

### 1. 小规模检查

先运行一个很短的 S34 计算，确认环境和输出流程正常：

```bash
python run_s34_reproduction.py --steps 1000 --snapshot-steps 0,500,1000 --progress-every 100
```

结果默认写入 `kmc_output/s34_reproduction/`。如果该目录已有检查点，程序会自动从不超过目标步数的最新检查点继续；若希望忽略已有检查点，可添加 `--no-resume`，或通过 `--output` 指定新的目录。

### 2. 完整 S34 复现

```bash
python run_s34_reproduction.py
```

默认设置对应代码中整理出的论文公开条件：

| 参数 | 默认值 |
| --- | ---: |
| 晶格盒尺寸 | 20 × 20 × 20 nm³ |
| CeO₂ 颗粒直径 | 5 nm |
| 温度 | 453 K |
| Ce / O 化学势 | −0.60 eV |
| Ce–O 结合能 | 0.30 eV |
| 总 KMC 事件数 | 5,000,000 |
| 快照步数 | 0、1,000,000、3,000,000、5,000,000 |
| 随机种子 | 2026 |

完整计算包含约 60 万个晶格位点和 500 万次事件，运行时间及磁盘占用会明显高于快速检查。程序使用局部更新的 `FastCeOxKMC` 引擎，并定期保存可恢复的 `.npz` 检查点。

常用参数示例：

```bash
python run_s34_reproduction.py \
  --steps 5000000 \
  --snapshot-steps 0,1000000,3000000,5000000 \
  --roughness-fraction 0.05 \
  --checkpoint-every 250000 \
  --reconcile-every 250000 \
  --output kmc_output/s34_reproduction
```

在 PowerShell 中可将反斜杠续行改为一行命令，或使用反引号 `` ` `` 续行。查看全部选项：

```bash
python run_s34_reproduction.py --help
```

### 3. Ir 动力学可视化

```bash
python run_ir_visualization.py
```

该脚本执行 3,000 步演示计算，每 50 步输出一帧，结果写入带时间戳的 `kmc_output/ir_visualization_YYYYMMDD_HHMMSS/` 目录。脚本内的 Ir 参数经过刻意调整，以便在短轨迹中展示各类事件，仅用于可视化。

### 4. 超声条件对照

```bash
python run_sonication_comparison.py
```

默认分别计算 200 步无超声和有超声轨迹，每 50 步保存一帧。也可调整：

```bash
python run_sonication_comparison.py --steps 1000 --snapshot-every 100 --seed 2026 --sonication-rate 20
```

结果写入带时间戳的 `kmc_output/sonication_comparison_YYYYMMDD_HHMMSS/`。其中 1 nm 腐蚀半径和 10% 溶解概率来自论文描述；超声事件速率和平均场生长脉冲属于可视化设定，不能直接用于定量物理解释。

## 输出文件

不同入口会生成以下文件中的一部分：

| 文件 | 内容 |
| --- | --- |
| `snapshot_XXXXXXXX.xyz` | 指定 KMC 步数下的原子结构，可作为 OVITO 文件序列打开 |
| `metrics.csv` | 粒子数、表面原子数、Ir 包埋比例、事件累计数等随步数的变化 |
| `event_counts.json` | 最终状态、停止原因和各类事件计数 |
| `checkpoint_XXXXXXXX.npz` | S34 高性能引擎的可恢复检查点 |
| `run_metadata.json` | 运行参数、论文设定、显式假设和模型说明 |
| `comparison_summary.csv` | 有/无超声条件的最终指标对比 |
| `comparison_process_metrics.csv` | 有/无超声条件在每个输出帧的并列指标及差值 |
| `OVITO_README.txt` | 当前结果对应的 OVITO 查看提示 |

扩展 XYZ 文件除元素和坐标外，还包含以下属性：

- `surface`：是否位于外表面；
- `ir_state`：`0` 为非 Ir，`1` 为离子态 Ir，`2` 为金属态 Ir；
- `embedded`：Ir 是否已不再暴露于外表面；
- `support_contacts`：Ir 邻近的 Ce/O 载体原子数；
- `condition`：配对轨迹中的实验条件编号。

### 在 OVITO 中查看

球形颗粒轨迹可打开：

```text
kmc_output/s34_reproduction/particle/snapshot_00000000.xyz
```

并选择将同目录快照作为文件序列载入。

超声对照推荐打开结果目录中的：

- `comparison_final.xyz`：最终并排结构；
- `comparison_ir_environment.xyz`：Ir 周围 1 nm 的局部环境；
- `paired_process/snapshot_00000000.xyz`：完整结构的配对动画；
- `paired_ir_environment/snapshot_00000000.xyz`：Ir 局部环境的配对动画。

可按 `species` 或 `ir_state` 着色，并利用 `embedded`、`support_contacts` 和 Slice modifier 检查 Ir 的包埋与载体覆盖情况。

## 模型概览

论文/DFT 参数与代码参数的逐项对应及尚未解决的参数，见
[`DFT_PARAMETER_MAPPING.md`](DFT_PARAMETER_MAPPING.md)。

晶格由萤石型 CeO₂ 的金属子晶格和氧子晶格组成。通用 KMC 引擎采用基于总速率的无拒绝事件选择，并以指数分布采样时间增量。模型支持：

- Ce/O 吸附与脱附；
- Ir 离子吸附、脱附和表面扩散；
- Ir 离子还原与金属 Ir 氧化；
- 超声诱导的随机界面腐蚀；
- 用于短程演示的平均场 Ce/O 再沉积；
- 外部溶液连通性、表面识别及形貌/包埋指标统计。

S34 入口使用按“事件类型 × 局部配位数”分桶的局部更新引擎，以避免在每次事件后重建全局事件目录。吸附还要求目标位点与相反子晶格的晶体原子接触，从而抑制有限盒子中的非物理均匀成核。

## 项目结构

```text
.
├── constants.py                    # 位点、物种和通道枚举
├── lattice_build.py                # 萤石晶格与邻接表构建
├── generation.py                   # 球形颗粒初始化、表面分析与 XYZ 输出
├── ceox_events.py                  # Ce/O 事件与速率
├── ir_events.py                    # Ir 事件与速率
├── sonication_events.py            # 超声腐蚀和平均场熟化生长
├── kmc_engine.py                   # 通用 KMC 引擎及指标输出
├── fast_ceox_kmc.py                # S34 使用的局部更新高性能引擎
├── run_s34_reproduction.py         # S34 复现入口
├── run_ir_visualization.py         # Ir 短程可视化入口
├── run_sonication_comparison.py    # 有/无超声配对对照入口
├── test_fast_ceox_kmc.py           # 高性能引擎测试
└── test_sonication_events.py        # 超声事件测试
```

## 测试

项目使用 Python 标准库 `unittest`：

```bash
python -m unittest discover -v
```

测试覆盖高性能引擎与完整事件目录的一致性、检查点恢复，以及超声事件的构建、腐蚀和平均场生长行为。

## 结果解释注意事项

- 固定随机种子可提高可重复性，但随机模型仍应通过多次独立运行评估统计波动。
- `KMC_time` 由当前相对速率计算；在频率因子未物理标定时，它不是经过验证的实验时间。
- S34 中未披露的吸附/脱附频率因子比经过拟合，默认目标是重现近似形貌范围，而非证明唯一的微观机制。
- Ir 可视化和超声对照脚本优先保证机制在短轨迹中可见，不应直接用于论文级定量结论。
- 每次正式计算都应保留 `run_metadata.json`，并在报告结果时同时记录随机种子和所有非论文参数。
