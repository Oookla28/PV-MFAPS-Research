# Current experiment scope

主清链展示的是 PlantSPD E01 上的一个 8-step、双分区、三 seed 运行矩阵。它适合复现算法动作、成本和 held evaluation 的关系。

50-step 补充目录目前只保留 metadata，缺少完整 matched rows 和 state files。CIFAR-100-LT IR=100 的 30-epoch 扩展已经执行并收录在 `artifacts/results/cifar100_lt/`，它是短训练预算的标准数据集迁移检查。ImageNet-LT、第二个 SPD 项目、可学习 Power(p) 几何和修正后的 A100 扩展链保存为协议入口。

机制材料中的 `A`, `rho`, `sigma` 是 FIT proxy。稳定性和 empirical-Bernstein 公式给出的是声明条件下的 envelope，不能直接转写成 Macro-F1 或总体泛化保证。Fused 结果是 real-checkpoint equivalence 与 measured micro-profile，系统复用路径和算法尾部指标需要分别解读。
