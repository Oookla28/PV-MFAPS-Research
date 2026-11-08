# Feedback Geometry and Stage-Aware Readout Refinement

这是一个面向长尾识别、SPD 几何表示和训练系统执行的完整研究复现包。仓库按研究推进顺序组织，从数据来源、反馈机制诊断、有限动作分配、系统融合，到 CIFAR-LT 上的阶段感知读出精化，读者可以沿着 README 从最小合同测试一直运行到 FIT freeze、一次性评估、消融和图表生成。

项目的核心对象是读出头与编码器之间的时间尺度差异。读出头的近似误差会通过表示 Jacobian 传回编码器，因而“当前表示是否可读出”和“当前读出是否给编码器提供了合适的反馈”是两个不同的训练问题。研究先在 SPD 协方差表示上测量这一差异，再把它写成有限动作的风险分配问题，最后得到一个更轻量的阶段策略。

## 1. 数据来源与任务定义

### PlantSeg 与 PlantSPD

PlantSeg 是研究中的真实场景植物病害数据来源。原始数据面向植物病害分割，包含复杂背景、视角变化和光照变化。数据论文为：

> Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P., and Huang, Z. *A Large-Scale In-the-wild Dataset for Plant Disease Segmentation*. Scientific Data 13, 205 (2026). DOI: [10.1038/s41597-025-06513-4](https://doi.org/10.1038/s41597-025-06513-4)

本项目使用从 PlantSeg 场景中整理出的 PlantSPD E01 协方差表示任务。每个样本由多视图特征形成 SPD 协方差矩阵，再经过 Raw、Log 或其他矩阵函数映射后送入读出头。E01 的研究接口包含 69 个类别，FIT/HELD 样本数为 4184/1046，FIT 类计数范围为 5 到 344，头尾计数比约为 68.8。FIT 用于训练和动作校准，HELD 在冻结之后统一读取。

仓库不包含原始图像、私有 checkpoint 或大规模 encoder state。数据提供方需要通过外部路径挂载下列接口。

```text
DATA_ROOT/
├── E01/
│   ├── input/rgb_views_uint8.npy
│   ├── input/meta.npz
│   ├── history_origin/prepared/SPLITS.npz
│   ├── source/runner_v2.py
│   └── assets/resnet18-f37072fd.pth
└── Plant_SPD/
    └── 11_f1_f4_A_forensics_v1/inputs/FIT_LABELS.npy
```

### CIFAR-100-LT 与 CIFAR-10-LT

标准长尾扩展使用 CIFAR-100 和 CIFAR-10 官方数据。数据由 [Toronto CIFAR repository](https://www.cs.toronto.edu/~kriz/cifar.html) 提供，代码在首次运行时按公开 URL 下载并校验。

- CIFAR-100-LT，imbalance ratio 100，100 类，30 epoch。
- CIFAR-10-LT，imbalance ratio 100，10 类，30 epoch。
- 类别计数由指数长尾规则生成，每个训练 seed 固定类别采样和数据顺序。
- CIFAR-10-LT 的组合数据哈希写入对应 manifest，避免把 CIFAR-10 和 CIFAR-100 文件混用。

## 2. 研究过程

### 2.1 SPD 表示与反馈机制

第一阶段比较 Raw 与 Log SPD 映射。矩阵函数的局部导数写成

$$
D\phi(C)[H]
=
U\left(G_\phi(\Lambda)\odot(U^\top H U)\right)U^\top,
$$

其中 $C=U\Lambda U^\top$，$G_\phi$ 是 Loewner 差商矩阵。Raw 映射的差商为常数，Log 映射会在小特征值方向放大部分局部谱扰动。研究不把这个局部放大直接等同于总体泛化结论，而是记录 readout residual、encoder feedback error、谱调制和类别支持数，形成可测的反馈风险对象。

相关代码位于：

```text
src/model/mechanism_synthetic.py
src/model/pv_mfaps_risk_envelope.py
src/model/pv_mfaps_risk_plot.py
docs/mechanism_feedback_risk.md
```

### 2.2 PV-MFAPS 有限动作分配

第二阶段把每个类别的 readout 精度写成有限动作。基本反馈风险代理为

$$
r_c(k_c,\alpha_c)
=
(1-\alpha_c)^2A_c^2\rho_c^{2k_c}
+
\alpha_c^2\frac{\sigma_c^2}{m q_c},
$$

其中 $k_c$ 是 readout 内层步数，$\alpha_c$ 是 reference feedback 混合比例，$\rho_c$ 是内层收缩率，$\sigma_c$ 是批次波动尺度，$q_c$ 是类别采样质量，$m$ 是 batch size。

有限动作 DP 处理预算、类别风险、尾部保护和动作量化。DP 的正确性是相对于给定风险表和动作网格的离散最优性。它不会自动变成连续优化或总体泛化定理。

核心入口：

```text
src/model/pv_mfaps_dp_solver.py
src/model/pv_mfaps_dp.py
src/model/pcfa_policy.py
tests/test_pv_mfaps_dp.py
```

### 2.3 分区鲁棒策略

PlantSPD 的配对分区使用共享动作策略。PR-PV-MFAPS 将两个分区的风险向量放入 KL 鲁棒目标，epsilon sweep 用来观察动作如何随着分区不确定性改变。由于两个分区的类频率配对固定，动作变化主要由分区特定的风险曲率而非频率差异驱动。

策略文件放在：

```text
configs/policies/PR_POLICY_eps001.json
configs/policies/PR_POLICY_eps010.json
configs/policies/PV_MFAPS_ACTION_POLICY_p0_s3407.json
configs/policies/PV_MFAPS_ACTION_POLICY_p1_s3407.json
```

### 2.4 Fused 执行路径

第三阶段将逻辑动作和物理 encoder 计算拆开。Legacy 路径为每个 correction group 重新计算 encoder，Fused 路径只执行一次 encoder forward，再在 detached feature/head 上完成逻辑 correction。真实 checkpoint 等价性 smoke 检查动作序列、逻辑计数、状态差异和 HELD 读取数。

```text
src/validated_runtime/pcfa_equivalence_smoke.py
src/validated_runtime/pcfa_cost_model.py
src/validated_runtime/pcfa_cost_smoke.py
src/validated_runtime/pivot_runner.py
docs/fused_system.md
```

A100 上的 one-step real-checkpoint smoke 记录了相同动作序列和逻辑计数，encoder physical forward 从 3 次变为 1 次，状态最大绝对差约为 $5.77\times10^{-6}$。Fused 结果描述的是执行路径和重复计算削减，FLOPs 与 wall-clock 必须分开记录。

### 2.5 Balanced Softmax 插件

插件把 Balanced Softmax 同时接入 native、reference、detached inner solver 和 Fused correction loss。它使用独立 arm 列表和独立输出目录，便于观察几何调度在另一种长尾目标下的行为。

插件模块：

```text
src/validated_runtime/bs_plugin_runner.py
src/validated_runtime/longtail_losses.py
src/engine/plugin_runner.py
configs/plugin/
docs/plugin_protocol.md
scripts/run_plugin.sh
```

默认插件 arm 包括：

- balanced_softmax
- bs_fixed_k_matched_B02
- bs_pcfa
- native_balanced
- effective_number
- logit_adjustment

### 2.6 CIFAR-LT 与阶段感知 TRR

早期 FD-SDPA、FTRR 和复杂统计门实验主要用于定位问题。真实长尾轨迹中，逐类经验 Bernstein 半径容易过大，统计门会拒绝非零动作。随后研究转向更稳定的 Two-Timescale Readout Refinement。

Stage-TRR-10 的规则是：

1. 前 10 个 epoch 进行一次 detached inner readout refinement，并用 refined readout 更新 encoder。
2. 后续 epoch 回到标准 Logit Adjustment 联合优化。
3. 训练只使用 FIT，prefix 长度在训练前固定。
4. 同一 batch、同一 cosine schedule、同一优化器基础学习率用于 matched LA 对照。

训练入口：

```text
experiments/cifar100_lt/trr_audited_runner.py
experiments/cifar100_lt/trr_audited_cifar10_runner.py
experiments/cifar100_lt/trr_audited_evaluate.py
docs/trr_stage10_registration_20261108.md
docs/trr_stage10_claims_evidence_audit_20261108.md
```

阶段切换的条件性 surrogate-risk 界为

$$
J(\widehat{k}_{1:T})-J(k^*_{1:T})
\le
2\sum_{t,c}\omega_c\epsilon_{t,c}
+
\epsilon_{\rm quant},
$$

它描述有限动作估计误差、切换成本和量化误差下的动态策略差距，不直接声称测试集收益。

## 3. 目录结构

```text
PV-MFAPS/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/
│   ├── core/
│   ├── plugin/
│   ├── policies/
│   └── ablations/
├── src/
│   ├── model/
│   ├── engine/
│   ├── data/
│   ├── evaluation/
│   ├── validated_runtime/
│   └── utils/
├── experiments/
│   └── cifar100_lt/
├── scripts/
├── tests/
├── docs/
├── artifacts/
│   ├── results/
│   ├── figures/
│   └── manifests/
└── figures/
```

公开包不保存原始数据、checkpoint、服务器路径和 SSH 信息。所有大文件通过外部 DATA_ROOT 接入。

## 4. 环境与硬件

主要 GPU 实验使用：

- NVIDIA A100-SXM4-40GB
- CUDA 13.0
- PyTorch 2.12.1+cu130
- Python 3.12
- 256 CPU cores
- 约 1 TB RAM

早期 PlantSPD Fused 复核也使用过 RTX 3090。不同 GPU 的绝对 wall-clock 不直接横向比较，成本表同时记录设备名称、CUDA 版本、PyTorch 版本、FLOPs 和 CUDA event wall-clock。

安装环境：

```bash
conda env create -f environment.yml
conda activate pv-mfaps-repro
pip install -r requirements.txt
```

## 5. 从零复现

### 5.1 设置路径

```bash
export REPO_ROOT=/path/to/PV-MFAPS
export DATA_ROOT=/path/to/external-data
export OUTPUT_ROOT=$REPO_ROOT/artifacts/results/run_$(date +%Y%m%d_%H%M%S)
```

这些变量只在当前 shell 存在，不写入 artifact。

### 5.2 环境检查和离散 solver

```bash
cd "$REPO_ROOT"
bash scripts/preflight.sh
bash scripts/run_dp_tests.sh
python tests/test_pv_mfaps_dp.py
python tests/test_pcfa_policy_contract.py
```

### 5.3 PlantSPD 主链

```bash
export ACTION_POLICY=$REPO_ROOT/configs/policies/PR_POLICY_eps001.json
export PARTITIONS=0,1
export TRAINING_SEEDS=3407,3408,3409
export STEPS=8
export BATCH_SIZE=8
bash scripts/train_core.sh
```

冻结和评估：

```bash
export RUN_DIR=$OUTPUT_ROOT/core
bash scripts/freeze.sh
SPLIT=fit bash scripts/evaluate.sh
SPLIT=held bash scripts/evaluate.sh
bash scripts/audit.sh
```

### 5.4 Balanced Softmax 插件

```bash
export PLUGIN_ARMS=balanced_softmax,bs_fixed_k_matched_B02,bs_pcfa,native_balanced,effective_number,logit_adjustment
bash scripts/run_plugin.sh
export RUN_DIR=$OUTPUT_ROOT/plugin
bash scripts/freeze.sh
SPLIT=held bash scripts/evaluate.sh
```

### 5.5 CIFAR-100-LT

```bash
export DATA_ROOT=/path/to/cifar-cache
bash scripts/cifar100_lt/smoke.sh
export OUTPUT_ROOT=$REPO_ROOT/artifacts/results/cifar100_lt/ir100_30epoch
bash scripts/cifar100_lt/train_30epoch.sh
RUN_DIR=$OUTPUT_ROOT bash scripts/cifar100_lt/evaluate.sh
```

### 5.6 Audited Stage-TRR-10

CIFAR-100-LT：

```bash
python experiments/cifar100_lt/trr_audited_runner.py \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/trr_c100_prefix10/seed_3407" \
  --epochs 30 --trr-epochs 10 --seed 3407 --gpu 0
```

CIFAR-10-LT：

```bash
python experiments/cifar100_lt/trr_audited_cifar10_runner.py \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/trr_c10_prefix10/seed_3407" \
  --epochs 30 --trr-epochs 10 --seed 3407 --gpu 0
```

LA 对照使用相同 runner，把 `--trr-epochs` 设为 0。训练阶段 JSON 要保持 `held_reads: 0`，所有 seed 完成后才生成 freeze manifest。

### 5.7 一次性评估

```bash
python experiments/cifar100_lt/trr_audited_evaluate.py \
  --run-dir "$OUTPUT_ROOT/trr_c100_prefix10" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/trr_c100_prefix10/HELD_EVAL.json" \
  --gpu 0
```

CIFAR-10-LT runner 会在 DATA_ROOT/extract/cifar-10-batches-py 下读取五个训练 batch，使用对应 evaluator 和该 DATA_ROOT。评估脚本只在冻结之后读取测试集，并将每次读取数写入 summary。

## 6. 消融实验

建议按以下顺序运行：

1. LA-only 与 Stage-TRR-10。
2. 全程 TRR 与 prefix5、prefix15、prefix20。
3. 30 epoch 与 50 epoch horizon。
4. Fused 与 Legacy one-step equivalence。
5. Balanced Softmax 插件与 fixed-k。
6. Raw/Log SPD 反馈风险诊断。
7. FTRR 统计门、pooled certificate 和 gate-off 对照。

每个消融使用独立输出目录。不可把 pilot TEST 结果覆盖到 freeze summary。旧的 optimizer 语义不一致矩阵、错误 CIFAR-10 数据哈希 smoke 和缺少必要状态的结果都保留在历史说明中，不进入新的聚合表。

## 7. 已整理的结果

### PlantSPD 主链与插件

原 PlantSPD 主链保存于 `artifacts/results/` 的 summary、freeze manifest、policy 和 claims audit。主链和插件的配置、预算、FLOPs、wall-clock、FIT/HELD 读取数均可从 JSON 追溯。

### Stage-TRR-10

CIFAR-100-LT 30 epoch，三 seed：

- Stage-TRR-10：Macro 0.4440，Tail-10% 0.1820。
- LA：Macro 0.4264，Tail-10% 0.1273。
- 配对差值：Macro +0.0176，Tail +0.0547。

CIFAR-10-LT IR=100，三 seed：

- Stage-TRR-10：Macro 0.7799，Tail-10% 0.7797。
- LA：Macro 0.7428，Tail-10% 0.7400。
- 配对差值：Macro +0.0372，Tail +0.0397。

CIFAR-100-LT 50 epoch：

- Macro 差值 +0.0046。
- Tail-10% 差值 -0.0033。

这些数字来自冻结后的三 seed 组汇总。30 epoch prefix 长度来自先前 pilot，因此对应结果按 exploratory 研究过程记录。CIFAR-10-LT 使用固定的 (N=10) 进行跨基准验证。

## 8. 图表与可视化

可直接从 JSON 生成以下图表：

- 两个 CIFAR 数据集的 Macro/Tail grouped bar。
- 30/50 epoch 的阶段边界折线图。
- C100 三个 seed 的 paired delta dot plot。
- Fused/Legacy physical forwards 与 wall-clock 柱状图。

图表脚本和输出放在 `figures/`，所有图都从 JSON/CSV 读取，不在绘图脚本中硬编码实验结果。

## 9. 研究记录和引用

研究过程的完整流水账按下面顺序阅读：

1. `docs/related_work.md`
2. `docs/research_timeline.md`
3. `docs/original_execution_protocol.md`
4. `docs/mechanism_feedback_risk.md`
5. `docs/pv_mfaps_algorithm.md`
6. `docs/fused_system.md`
7. `docs/plugin_protocol.md`
8. `docs/fd_sdpa_tracker.md`
9. `docs/ftrr_derivation_20261108.md`
10. `docs/trr_stage10_registration_20261108.md`
11. `docs/trr_stage10_claims_evidence_audit_20261108.md`

引用 PlantSeg 时请使用 DOI。引用 CIFAR 时请说明官方 CIFAR repository 与长尾构造规则。涉及 Balanced Softmax、Logit Adjustment、LDAM-DRW 和 SPD 几何的背景说明，应在论文或扩展文档中给出对应原始文献，而不是把仓库内的实现当作文献来源。
