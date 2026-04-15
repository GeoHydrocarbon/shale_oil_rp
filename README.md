# 裂缝型页岩油岩石物理建模流程说明

本项目基于本地下载的 `rockphypy-main`，建立了一套适用于裂缝型页岩油工区的轻量岩石物理建模流程。

当前流程已经包含：

- 数据前处理
- 无裂缝基线岩石物理建模
- 参数敏感性分析
- 裂缝正演
- 裂缝密度与裂缝倾角联合反演
- 深度综合图
- 交会图

根目录主程序为 [main.py](/D:/项目/2济阳凹陷/17-岩石物理建模/main.py)。

## 1. 目录说明

当前实际在用的核心文件如下：

- [config.yaml](/D:/项目/2济阳凹陷/17-岩石物理建模/config.yaml)：统一配置文件
- [main.py](/D:/项目/2济阳凹陷/17-岩石物理建模/main.py)：主程序入口
- [preprocess_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/preprocess_v2.py)：数据前处理
- [rp_model_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/rp_model_v2.py)：无裂缝基线岩石物理模型
- [sensitivity_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/sensitivity_v2.py)：参数敏感性分析
- [fracture_analysis_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/fracture_analysis_v2.py)：裂缝正演与裂缝密度-倾角敏感性分析
- [fracture_inversion.py](/D:/项目/2济阳凹陷/17-岩石物理建模/fracture_inversion.py)：裂缝密度和裂缝倾角联合反演
- [plotting.py](/D:/项目/2济阳凹陷/17-岩石物理建模/plotting.py)：深度综合图
- [crossplotting.py](/D:/项目/2济阳凹陷/17-岩石物理建模/crossplotting.py)：交会图模块
- [rockphypy-main](/D:/项目/2济阳凹陷/17-岩石物理建模/rockphypy-main)：本地开源建模库

说明：

- `fracture_analysis.py` 是早期尝试文件，当前主程序不再调用，可忽略。
- `preprocess.py`、`rp_model.py` 为旧版文件，当前主程序不再调用。

## 2. 一键运行

在项目根目录执行：

```powershell
python .\main.py
```

如果使用其他配置文件：

```powershell
python .\main.py --config .\your_config.yaml
```

## 3. 当前完整流程

### 3.1 数据前处理

由 [preprocess_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/preprocess_v2.py) 完成。

主要功能：

- 将 `DTC/DTS` 转换为 `Vp/Vs`
- 统一密度、孔隙度、饱和度等单位
- 将矿物质量分数中的负值截断为 `0`
- 对矿物质量分数重新归一化
- 将矿物质量分数转换为体积分数
- 将 `TOC` 转换为干酪根固相
- 计算基质密度、测井反演模量和质量控制标记

### 3.2 无裂缝基线岩石物理模型

由 [rp_model_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/rp_model_v2.py) 完成。

当前建模链条为：

`矿物+干酪根基质 -> 双孔隙 DEM 干岩骨架 -> 孔隙流体 -> Gassmann -> Vp/Vs/rho`

当前采用的主要物理假设：

- 干酪根作为低模量固相进入基质
- 总孔隙拆分为硬孔和软孔
- 使用 `rockphypy` 的 `EM.Berryman_DEM()` 进行双孔隙干岩建模
- 使用 `BW` 模块计算油水流体性质
- 使用 `Fluid.Gassmann()` 做低频饱和替换

### 3.3 参数敏感性分析

由 [sensitivity_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/sensitivity_v2.py) 完成。

当前支持：

- 单参数一维扫描
- 双参数二维网格扫描
- 自动输出误差统计表和热图

目前主要用于标定：

- 干酪根参数
- 软孔占比
- 软孔和硬孔纵横比

### 3.4 裂缝正演

由 [fracture_analysis_v2.py](/D:/项目/2济阳凹陷/17-岩石物理建模/fracture_analysis_v2.py) 完成。

当前裂缝正演基于：

- `rockphypy.EM.hudson_cone()`
- `rockphypy.Anisotropy.vel_azi_VTI()`

当前裂缝模块支持：

- 基准裂缝正演结果表
- 裂缝密度-倾角二维敏感性分析

### 3.5 裂缝联合反演

由 [fracture_inversion.py](/D:/项目/2济阳凹陷/17-岩石物理建模/fracture_inversion.py) 完成。

当前反演思路为：

- 候选裂缝密度网格
- 候选裂缝倾角网格
- 每个深度点做裂缝正演
- 使用 `Vp` 和 `Vs` 残差作为目标函数
- 用全井平滑约束寻找一条最优裂缝密度-倾角路径

这意味着：

- 裂缝密度随深度变化
- 裂缝倾角也可随深度变化
- 但不是逐点独立乱跳，而是带路径平滑约束

### 3.6 绘图与交会图

当前已经集成：

- 基线速度与误差四联深度图
- 裂缝反演参数深度图
- 裂缝反演后速度与误差四联深度图
- 页岩油解释交会图总面板

## 4. 配置文件怎么改

所有控制参数都在 [config.yaml](/D:/项目/2济阳凹陷/17-岩石物理建模/config.yaml)。

最常改的几个部分如下。

### 4.1 数据路径和列名

如果换井或换数据，优先改这些：

- `paths.input_file`
- `paths.sheet_name`
- `columns.*`
- `units.*`

### 4.2 矿物和干酪根参数

如果换工区或要重新标定基质：

- `minerals.*`
- `kerogen.*`
- `pore_system.*`
- `fluid.*`

### 4.3 裂缝正演开关

位于 `fracture` 段：

- `enabled`
- `run_forward_model`
- `run_sensitivity_analysis`

用途如下：

- `run_forward_model: true`：输出一版基准裂缝正演结果
- `run_sensitivity_analysis: true`：输出裂缝密度-倾角热图和敏感性表

### 4.4 裂缝联合反演

位于 `fracture.inversion` 段。

最重要的参数：

- `density_grid`：裂缝密度候选值
- `dip_grid_deg`：允许搜索的裂缝倾角候选值
- `smooth_lambda_density`：裂缝密度平滑权重
- `smooth_lambda_dip`：裂缝倾角平滑权重
- `max_density_change`：相邻深度密度最大跳变
- `max_dip_change_deg`：相邻深度倾角最大跳变

强烈建议：

- 不要让 `dip_grid_deg` 过宽
- 应根据地质认识只给合理倾角范围

如果反演结果总贴在上下边界，说明：

- 候选倾角范围可能不合理
- 应重新调整 `dip_grid_deg`

### 4.5 交会图

位于 `crossplots` 段。

关键项：

- `enabled`
- `elastic_source`

其中：

- `elastic_source: fracture_inv` 表示用裂缝反演后的 `Vp/Vs`
- `elastic_source: log` 表示用实测 `Vp/Vs`

## 5. 当前输出文件

当前主要输出位于：
[rockphypy-main/data/Jorlin](/D:/项目/2济阳凹陷/17-岩石物理建模/rockphypy-main/data/Jorlin)

常见结果包括：

- `jorlin_preprocessed.xlsx`：前处理结果
- `jorlin_modeled.xlsx`：无裂缝基线建模结果
- `sensitivity_summary.xlsx`：参数敏感性分析结果
- `jorlin_fracture_reference.xlsx`：裂缝正演参考结果
- `fracture_sensitivity.xlsx`：裂缝密度-倾角敏感性分析结果
- `jorlin_fracture_inversion.xlsx`：裂缝联合反演结果

图件位于：

- [rockphypy-main/data/Jorlin/figures](/D:/项目/2济阳凹陷/17-岩石物理建模/rockphypy-main/data/Jorlin/figures)
- [rockphypy-main/data/Jorlin/figures/fracture](/D:/项目/2济阳凹陷/17-岩石物理建模/rockphypy-main/data/Jorlin/figures/fracture)
- [rockphypy-main/data/Jorlin/figures/crossplots](/D:/项目/2济阳凹陷/17-岩石物理建模/rockphypy-main/data/Jorlin/figures/crossplots)

## 6. 推荐使用顺序

如果以后换一个新井或新工区，建议按下面顺序使用。

1. 修改 `config.yaml` 里的输入路径、列名和单位
2. 运行 `python .\main.py`
3. 先检查 `jorlin_preprocessed.xlsx` 是否正常
4. 再检查 `jorlin_modeled.xlsx` 和 `velocity_depth_panel.png`
5. 再看 `sensitivity_summary.xlsx` 调整基线参数
6. 再开启或调整裂缝正演和裂缝反演参数
7. 最后看裂缝反演图和交会图做地质解释

## 7. 当前适用范围

这套流程当前更适合：

- 裂缝型页岩油
- 有测井 `Vp/Vs/rho`
- 有矿物组分和 `TOC`
- 需要做“基线模型 + 裂缝补偿解释”

## 8. 当前需要注意的地方

1. 当前裂缝模型采用的是 `hudson_cone` 近似，因此反演得到的更适合解释为“等效裂缝密度”和“等效裂缝倾角”。
2. 如果裂缝倾角反演结果总贴边界，不要直接拿去做强解释，应优先调整 `dip_grid_deg`。
3. 当前裂缝模块更适合做井曲线层面的弹性补偿和解释，不等同于严格的真实裂缝几何反演。
4. 如果后续要做更严格的单组倾斜裂缝模型，可以在当前流程上继续升级旋转刚度张量或正交各向异性建模。

## 9. 最常用命令

```powershell
python .\main.py
git status
git add .
git commit -m "update rock physics workflow"
git push -u origin main
```

## 10. 一句话总结

如果只记一件事：

先改 [config.yaml](/D:/项目/2济阳凹陷/17-岩石物理建模/config.yaml)，再运行 [main.py](/D:/项目/2济阳凹陷/17-岩石物理建模/main.py)，然后按“前处理 -> 基线模型 -> 裂缝反演 -> 交会图”的顺序检查结果。
