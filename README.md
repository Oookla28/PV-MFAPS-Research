# Feedback Geometry and Stage-Aware Readout Refinement

这是一个面向读者复现实验的最小研究包。公开内容只包含可运行源码、配置、数据接口、evaluation 入口、必要的结果摘要和理论说明。原始图像、checkpoint、encoder state、服务器信息和内部研究记录均不在仓库中。

## 快速导航

- [研究对象与数据来源](#1-研究对象) · [PlantSeg / PlantSPD](#plantseg-与-plantsdp) · [CIFAR-LT](#cifar-lt)
- [数学对象](#3-数学对象) · [反馈风险](#32-feedback-gap) · [阶段精化](#36-stage-trr-n)
- [模块地图](#4-模块地图) · [环境](#5-环境) · [合同测试](#6-合同测试)
- [PlantSPD 主链](#7-plantsdp-主链) · [插件](#8-balanced-softmax-插件)
- [CIFAR-100-LT](#9-cifar-100-lt-stage-trr-10) · [CIFAR-10-LT](#10-cifar-10-lt-stage-trr-10) · [标准 200 epoch 矩阵](#12-cifar-100-lt-ir100-的-200-epoch-矩阵)
- [一次性评估](#11-一次性-evaluation) · [消融设计](#13-消融设计) · [参考结果](#14-参考结果)

## 从零开始的最短路径

```bash
git clone https://github.com/Oookla28/PV-MFAPS-Research.git
cd PV-MFAPS-Research
conda env create -f environment.yml
conda activate pv-mfaps-repro
pip install -r requirements.txt
python tests/test_pv_mfaps_dp.py
python tests/test_pcfa_policy_contract.py
```

然后准备外部数据目录，按 [PlantSPD 主链](#7-plantsdp-主链) 或
[CIFAR-LT 入口](#9-cifar-100-lt-stage-trr-10) 运行训练，再按
[一次性 evaluation](#11-一次性-evaluation) 读取冻结后的 evaluation split。

完整文件索引见 [`docs/file_map.md`](docs/file_map.md)，理论与模块对应关系见
[`docs/theory_and_module_map.md`](docs/theory_and_module_map.md)，图表重建说明见
[`figures/README.md`](figures/README.md)。

## 1. 研究对象

项目包含两个相互衔接的研究模块。

1. SPD 协方差表示中的反馈风险。矩阵函数的 Frechet 导数通过 Loewner 差商影响读出误差如何传回 encoder。
2. 长尾识别中的阶段感知读出精化。Stage-TRR-N 在前 N 个 epoch 使用一次 detached readout refinement，之后回到标准 Logit Adjustment 联合训练。

PlantSPD 提供真实场景 SPD 分支，CIFAR-100-LT 和 CIFAR-10-LT 提供标准长尾识别复现入口。

## 2. 数据来源

### PlantSeg 与 PlantSPD

PlantSeg 原始资源面向真实场景植物病害分割，包含复杂背景、视角和光照变化。数据论文为 Wei et al., A Large-Scale In-the-wild Dataset for Plant Disease Segmentation, Scientific Data 13, 205 (2026), [DOI 10.1038/s41597-025-06513-4](https://doi.org/10.1038/s41597-025-06513-4)。

本项目不重新发布 PlantSeg 图像。PlantSPD 是从该场景整理出的多视图 SPD 协方差分类协议，使用 69 个疾病类别，FIT/HELD 为 4,184/1,046，FIT 类计数范围为 5–344。

外部数据接口：

~~~text
DATA_ROOT/
├── E01/
│   ├── input/rgb_views_uint8.npy
│   ├── input/meta.npz
│   ├── history_origin/prepared/SPLITS.npz
│   ├── source/runner_v2.py
│   └── assets/resnet18-f37072fd.pth
└── Plant_SPD/
    └── 11_f1_f4_A_forensics_v1/inputs/FIT_LABELS.npy
~~~

### CIFAR-LT

CIFAR-10 和 CIFAR-100 使用公开官方数据。代码按 imbalance ratio 100 构造长尾训练集，并为每个数据文件写入组合 SHA256。

- CIFAR-100-LT：100 类，IR=100，30/50 epoch 历史复现，以及独立的 ResNet-32、200 epoch 标准矩阵配置。
- CIFAR-10-LT：10 类，IR=100，30 epoch。
- 训练 seeds：3407、3408、3409。

## 3. 数学对象

### 3.1 SPD 谱映射

令
$$
C=U\operatorname{diag}(\lambda_1,\ldots,\lambda_d)U^\top.
$$

谱函数作用为
$$
\phi(C)
=
U\operatorname{diag}(\phi(\lambda_1),\ldots,\phi(\lambda_d))U^\top.
$$

对称扰动 H 的 Frechet 导数为
$$
D\phi_C[H]
=
U\left(G_\phi(\lambda)\odot(U^\top H U)\right)U^\top,
$$
其中
$$
(G_\phi)_{ij}
=
\begin{cases}
\dfrac{\phi(\lambda_i)-\phi(\lambda_j)}
{\lambda_i-\lambda_j}, & i\ne j,\\
\phi'(\lambda_i), & i=j.
\end{cases}
$$

前向特征依赖 $\phi(\lambda)$，反馈敏感性依赖 $G_\phi$。

### 3.2 Feedback gap

当前读出为 $W_t$，FIT 上的参考读出为 $\widetilde W_t$。定义
$$
e_t
=
\nabla_\psi L(\psi_t,W_t)
-
\nabla_\psi L(\psi_t,\widetilde W_t),
$$
以及
$$
\kappa_t
=
\frac{\lVert e_t\rVert}
{\lVert\nabla_\psi L(\psi_t,\widetilde W_t)\rVert+\delta}.
$$

这些量是 FIT 反馈诊断，不是总体泛化常数。

### 3.3 有限动作风险

类别 c、分区 p、内层步数 k、reference mixing 系数 α 的风险代理为
$$
r_{c,p}(k,\alpha)
=
(1-\alpha)^2A_{c,p}^2\rho_{c,p}^{2k}
+
\alpha^2\frac{\sigma_{c,p}^2}{m q_c}.
$$

A、ρ、σ、n 和 q 都从 FIT 校准得到。稳定性项和 empirical-Bernstein 项在风险包络代码中组合。

### 3.4 KL 分区鲁棒性

两个配对分区的名义分布为 $\bar u=(1/2,1/2)$：
$$
\mathcal U_\varepsilon
=
\{u\in\Delta^2:
\operatorname{KL}(u\|\bar u)\le\varepsilon\}.
$$

动作 a 的鲁棒风险为
$$
\mathcal E_c(a)
=
\sup_{u\in\mathcal U_\varepsilon}
\sum_{p=0}^1u_p\widehat R_{c,p}(a).
$$

### 3.5 离散 Pareto DP

动作集合为 $\mathcal A$，离散成本为 $\bar d_c(a)$。DP 递推保存分区风险向量的 Pareto frontier：
$$
\mathcal P_j(b)
=
\operatorname{Pareto}
\left(
\bigcup_{a\in\mathcal A}
\left\{
(r_0+R_{j,0}(a),r_1+R_{j,1}(a);\pi\cup\{a\})
\right\}
\right).
$$

该结论只保证给定风险表、动作网格和量化成本下的离散最优性。

### 3.6 Stage-TRR-N

前 N 个 epoch 使用
$$
W_t^{(1)}
=
W_t-\eta_h\nabla_W L_{\mathrm{LA}}(\psi_t,W_t),
$$
并以 detached refined readout 更新 encoder：
$$
\psi_{t+1}
=
\psi_t-
\eta_\psi
\nabla_\psi
L_{\mathrm{LA}}(\psi_t,\operatorname{stopgrad}(W_t^{(1)})).
$$

第 N 个 epoch 之后恢复联合 Logit Adjustment：
$$
(\psi_{t+1},W_{t+1})
=
(\psi_t,W_t)
-
\eta_t\nabla_{\psi,W}L_{\mathrm{LA}}(\psi_t,W_t).
$$

阶段策略的条件性估计界为
$$
J(\widehat{k}_{1:T})-J(k^*_{1:T})
\le
2\sum_{t,c}\omega_c\epsilon_{t,c}
+
\epsilon_{\mathrm{quant}}.
$$

## 4. 模块地图

| 模块 | 入口 | 作用 |
|---|---|---|
| SPD 反馈风险 | [`src/model/mechanism_synthetic.py`](src/model/mechanism_synthetic.py) | 构造谱反馈诊断 |
| 风险包络 | [`src/model/pv_mfaps_risk_envelope.py`](src/model/pv_mfaps_risk_envelope.py) | 组合 bias、稳定性与统计项 |
| 有限动作 DP | [`src/model/pv_mfaps_dp.py`](src/model/pv_mfaps_dp.py) | 运行离散预算分配 |
| PC-FA 策略 | [`src/model/pcfa_policy.py`](src/model/pcfa_policy.py) | 参数化长尾动作策略 |
| Fused 运行时 | [`src/validated_runtime/pcfa_equivalence_smoke.py`](src/validated_runtime/pcfa_equivalence_smoke.py) | 检查 Legacy/Fused 语义等价 |
| PlantSPD 训练 | [`src/validated_runtime/pr_pv_mfaps_runner.py`](src/validated_runtime/pr_pv_mfaps_runner.py) | 运行 PlantSPD 主链 |
| Balanced Softmax 插件 | [`src/validated_runtime/bs_plugin_runner.py`](src/validated_runtime/bs_plugin_runner.py) | 运行插件矩阵 |
| Stage-TRR | [`experiments/cifar100_lt/trr_audited_runner.py`](experiments/cifar100_lt/trr_audited_runner.py) | CIFAR-100-LT 训练 |
| CIFAR-10-LT Stage-TRR | [`experiments/cifar100_lt/trr_audited_cifar10_runner.py`](experiments/cifar100_lt/trr_audited_cifar10_runner.py) | CIFAR-10-LT 训练 |
| 一次性评估 | [`experiments/cifar100_lt/trr_audited_evaluate.py`](experiments/cifar100_lt/trr_audited_evaluate.py) | 冻结后读取 evaluation split |
| DP 合同测试 | [`tests/test_pv_mfaps_dp.py`](tests/test_pv_mfaps_dp.py) | 检查 DP 离散最优性 |
| PC-FA 合同测试 | [`tests/test_pcfa_policy_contract.py`](tests/test_pcfa_policy_contract.py) | 检查动作空间和预算 |

更完整的公式与模块说明见 [`docs/theory_and_module_map.md`](docs/theory_and_module_map.md)。从 SPD 反馈诊断到 Stage-TRR 的逐步推导见 [`docs/stage_trr_derivation.md`](docs/stage_trr_derivation.md)。

## 5. 环境

参考硬件：

- NVIDIA A100-SXM4-40GB
- CUDA 13.0
- PyTorch 2.12.1+cu130
- Python 3.12
- 256 CPU cores
- 约 1 TB RAM

安装：

~~~bash
conda env create -f environment.yml
conda activate pv-mfaps-repro
pip install -r requirements.txt
~~~

## 6.1 读者侧脚本入口

仓库提供四个不含私有路径的最小脚本：

- [`scripts/download_cifar100.sh`](scripts/download_cifar100.sh)：下载并解压官方 CIFAR-100 Python archive。
- [`scripts/run_cifar100_minimal.sh`](scripts/run_cifar100_minimal.sh)：运行预检与 1 epoch、2 batch smoke。
- [`scripts/run_cifar100_standard_200.sh`](scripts/run_cifar100_standard_200.sh)：运行四个标准基线并在冻结后读取 test split。
- [`scripts/run_stage_trr_200.sh`](scripts/run_stage_trr_200.sh)：运行同一 ResNet-32 合同下的 Stage-TRR-10。

脚本只使用 `CIFAR_ROOT`、`OUTPUT_ROOT`、`GPU_ID` 和 `SEEDS` 环境变量，不包含服务器地址、账号、token 或个人目录。

## 6. 合同测试

~~~bash
python tests/test_pv_mfaps_dp.py
python tests/test_pcfa_policy_contract.py
python -m compileall -q src experiments figures
~~~

预期输出包含 PV_MFAPS_DP_TEST_PASS 和 status: PASS。

## 7. PlantSPD 主链

设置外部数据目录：

~~~bash
export REPO_ROOT=/path/to/PV-MFAPS-Research
export DATA_ROOT=/path/to/plantspd-data
export OUTPUT_ROOT=$REPO_ROOT/artifacts/results/plantspd_run
~~~

运行 FIT-only 训练：

~~~bash
export ACTION_POLICY=$REPO_ROOT/configs/policies/PR_POLICY_eps001.json
export PARTITIONS=0,1
export TRAINING_SEEDS=3407,3408,3409
export STEPS=8
export BATCH_SIZE=8
PYTHONPATH=src/validated_runtime python src/validated_runtime/pr_pv_mfaps_runner.py --root "$DATA_ROOT" --out "$OUTPUT_ROOT" --action-policy "$ACTION_POLICY" --partitions 0,1 --seeds 3407,3408,3409 --steps 8 --batch-size 8
~~~

冻结、FIT evaluation 和 HELD evaluation 使用：

~~~bash
PYTHONPATH=src/validated_runtime python src/validated_runtime/make_freeze_manifest.py --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/FIT_FREEZE_MANIFEST.json"
PYTHONPATH=src/validated_runtime python src/validated_runtime/held_eval.py --freeze-manifest "$OUTPUT_ROOT/FIT_FREEZE_MANIFEST.json" --output "$OUTPUT_ROOT/HELD_EVAL.json"
~~~

训练阶段必须保持 held_reads=0。只有所有 checkpoint 写入后才能执行 evaluator。

## 8. Balanced Softmax 插件

插件使用独立输出目录：

~~~bash
python -m src.validated_runtime.bs_plugin_runner --data-root "$DATA_ROOT" --output "$OUTPUT_ROOT/plugin" --arms balanced_softmax,bs_fixed_k_matched_B02,bs_pcfa,native_balanced,effective_number,logit_adjustment --seeds 3407,3408,3409
~~~

插件同时作用于 native、reference、solver 和 Fused correction loss。它是独立可复现模块。

## 9. CIFAR-100-LT Stage-TRR-10

~~~bash
export CIFAR_ROOT=/path/to/cifar-cache
export OUTPUT_ROOT=$REPO_ROOT/artifacts/results/cifar100_stage10
python experiments/cifar100_lt/trr_audited_runner.py --data-root "$CIFAR_ROOT" --output "$OUTPUT_ROOT/seed_3407" --epochs 30 --trr-epochs 10 --seed 3407 --gpu 0
~~~

对三个 seed 重复运行。LA 对照将 trr-epochs 设为 0。50 epoch 只需把 epochs 改为 50，保持 trr-epochs 10。

## 10. CIFAR-10-LT Stage-TRR-10

~~~bash
python experiments/cifar100_lt/trr_audited_cifar10_runner.py --data-root "$CIFAR_ROOT/cifar-10-batches-py" --output "$OUTPUT_ROOT/cifar10_seed_3407" --epochs 30 --trr-epochs 10 --seed 3407 --gpu 0
~~~

该 runner 会记录五个训练 batch 的组合 hash 和类别计数。

## 11. 一次性 evaluation

所有训练 JSON 和 checkpoint 完成后，运行：

~~~bash
python experiments/cifar100_lt/trr_audited_evaluate.py --run-dir "$OUTPUT_ROOT/seed_3407" --data-root "$CIFAR_ROOT" --output "$OUTPUT_ROOT/HELD_EVAL.json" --gpu 0
~~~

多个 seed 需要使用同一个 frozen group 目录。评估脚本只在训练结束后读取 evaluation split，并把读取数量写入结果。

## 12. CIFAR-100-LT IR=100 的 200 epoch 矩阵

标准矩阵使用 ResNet-32（16/32/64 channels，每个 stage 五个 block）、zero-padding crop、SGD、5 epoch warmup 和 160/180 epoch 学习率衰减。IR=100 的训练集包含 10,899 张图，计数从 500 递减到 5。命令、参数合同和 Stage-TRR-10 的同 horizon 入口集中在 [`configs/cifar100_lt/ir100_200epoch.yaml`](configs/cifar100_lt/ir100_200epoch.yaml) 与 [`docs/cifar100_lt_protocol.md`](docs/cifar100_lt_protocol.md)。

矩阵包含 ERM、Balanced Softmax、train-time Logit Adjustment 和 LDAM-DRW。Balanced Softmax 与 $\tau=1$ 的 Logit Adjustment 在该实现中只相差一个类别无关常数，因此作为两个读者侧名称保留，结果解读时不把它们当作两次独立的算法增益。训练阶段只读 FIT，全部 checkpoint 写入后再由 evaluation 入口一次性读取官方 test split。

## 13. 消融设计

建议保持以下消融顺序：

1. LA-only 与 Stage-TRR-10。
2. 全程 TRR 与 prefix5、prefix15、prefix20。
3. 30 epoch 与 50 epoch。
4. Legacy 与 Fused one-step equivalence。
5. Balanced Softmax 与 fixed-k。
6. Raw/Log SPD 反馈风险。
7. FTRR 统计门与 gate-off 诊断。

每个消融使用独立输出目录。动作表、训练 seed、batch size、optimizer 和成本口径必须保持可追踪。

## 14. 已整理的读者侧结果

短程跨数据集结果按数据集分别报告：

| 数据集与 horizon | 方法 | Macro recall | Tail-10% recall | 相对 LA 的 Tail 变化 |
|---|---:|---:|---:|---:|
| CIFAR-100-LT IR=100，30 epoch | LA | 0.4264 | 0.1273 | — |
| CIFAR-100-LT IR=100，30 epoch | Stage-TRR-10 | 0.4440 | 0.1820 | +43.0% |
| CIFAR-10-LT IR=100，30 epoch | LA | 0.7428 | 0.7400 | — |
| CIFAR-10-LT IR=100，30 epoch | Stage-TRR-10 | 0.7799 | 0.7797 | +5.4% |

这两组相对变化来自不同数据集，分别保留，不合并为一个跨数据集百分比。50 epoch 和 200 epoch 结果用于观察阶段窗口，命令和指标定义见 CIFAR-LT 协议文档。

系统执行轴单独记录：

| 指标 | 来源实验 | 数值 | 口径 |
|---|---|---:|---|
| Fused/Legacy 一步等价性 | corrected CIFAR one-step smoke | state max delta $4.47\times10^{-8}$ | loss/action semantics matched, `held_reads=0` |
| 物理 encoder forward | CIFAR-100-LT 30 epoch，seed 3407 | 2,580 vs 54,180 | Fused vs Legacy，同一动作序列 |
| FLOPs | CIFAR-100-LT 30 epoch，seed 3407 | $8.72\times10^{10}$ vs $1.83\times10^{12}$ | Fused / Legacy，约 0.048x |
| wall-clock | CIFAR-100-LT 30 epoch，seed 3407 | 73.175 s vs 316.376 s | 同 GPU 顺序计时，约 4.32x 加速 |

系统数字描述的是执行图与物理代价。它们与动态干预的精度指标分开使用。

## 14. 参考结果

公开包中的 JSON 摘要用于检查复现结果的量级：

- CIFAR-100-LT 30 epoch：Stage-TRR-10 Macro 0.4440，Tail-10% 0.1820；LA Macro 0.4264，Tail-10% 0.1273。
- CIFAR-10-LT 30 epoch：Stage-TRR-10 Macro 0.7799，Tail-10% 0.7797；LA Macro 0.7428，Tail-10% 0.7400。
- CIFAR-100-LT 50 epoch：Macro 差值 +0.46pp，Tail-10% 差值 -0.33pp。
- PlantSPD PR-PV-MFAPS：Macro 0.4841，Tail-10% 0.2976。
- Fused real-checkpoint equivalence：physical encoder forward 3→1，最大状态误差约 $5.77\times10^{-6}$。

这些摘要用于复现核对。原始数据、checkpoint 和大规模 encoder state 需要自行准备。

## 15. 协议说明

- 公开脚本把训练和 evaluation split 分开。训练 JSON 的 `held_reads` 保持为 0，所有 checkpoint 写入后才调用 evaluator。
- 30/50 epoch 的历史结果使用项目早期的宽浅 CIFAR 模型，200 epoch 矩阵使用 ResNet-32。两套配置分别复现，不把它们混成同一张标准表。
- Balanced Softmax 与 train-time Logit Adjustment 在 $\tau=1$ 时是同一个 prior-aware objective 的不同写法，因此结果阅读时保留名称但不重复计算算法增益。
- PaCo、RIDE、MiSLAS 等需要额外对比学习分支、专家路由或两阶段校准的重型方法没有在轻量 runner 中用简化版本替代。需要比较时应接入其完整官方实现和对应训练预算。
- Fused/Legacy 是执行图消融。算法指标、物理 encoder forwards、FLOPs 和 wall-clock 分列记录。

## 15. 公开内容边界

公开仓库只保留：

- 可运行源码和 evaluation 入口。
- 配置、数据接口和合同测试。
- 小型结果摘要、静态图和 LaTeX 表。
- 面向读者的理论与模块说明。

研究流水账、内部审计过程、阶段注册记录、GPU 编排脚本和 AAAI 初稿保留在独立本地目录，不属于公开最小复现单元。

