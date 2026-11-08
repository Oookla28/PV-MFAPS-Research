# Research timeline

## 1. 从 SPD 读出开始

最初的对象是多视图图像的 SPD 协方差统计。Raw 协方差和 Log-Euclidean 协方差都能被固定 encoder 后的读出头利用，但同样的前向可读出性并不意味着 encoder 收到相同的反馈。研究先把 readout residual、谱函数 Frechet derivative 和 encoder gradient 分开记录。

## 2. 反馈风险诊断

在固定 FIT checkpoint 上，native head 和 affine reference head 对同一批 FIT 样本产生两组 encoder gradient。二者的差定义为 feedback error。Loewner 差商把矩阵函数的前向谱映射和反向谱调制连接起来。这个阶段形成了 `A`, `rho`, `sigma`, `n` 四类 FIT proxy，后续动作分配都从这里开始。

## 3. 长尾调度原型

均值调度、CVaR 调度、importance-corrected sampling、trust region 和 K-MFAPS 依次作为原型。每个原型都保留 native、reference、solver、action 和成本计数。它们共同揭示了平均风险、最差类别风险和 wall-clock 预算之间的冲突，于是研究对象从单一 k 变成了 `(k_c, alpha_c, q_c)` 的离散动作。

## 4. PV-MFAPS 与离散 DP

风险 envelope 把 bias、稳定性、经验 Bernstein 置信项和量化成本放入有限动作 DP。小规模穷举用于检查 solver，规模实验用于记录 frontier。这个阶段的数学对象是量化后的有限优化问题，训练网络的总体泛化需要另行评估。

## 5. Fused 计算图

初始实现把不同 correction group 分开计算，encoder 被重复调用。Fused 路径在同一 outer step 复用 encoder 输出，并把 correction 放到 detached head。真实 checkpoint smoke 验证了动作和逻辑计数保持一致，随后才进入 matched 训练链。

## 6. 分区鲁棒策略

E01 的两个分区按类别配对，类频率一致。PR-PV-MFAPS 使用共享动作向量和 KL ambiguity objective，在分区风险向量上做 Pareto frontier 选择。epsilon sweep 观察到少数类的动作翻转，这些翻转来自风险曲率和支配分区，而不是分区频率变化。

## 7. 插件线

Balanced Softmax 被作为独立目标函数插件接入 native、reference、solver 和 fused correction loss。插件有独立 registration、arm 矩阵和一次性 HELD 流程，主链和插件链的输出目录保持分开。

## 8. 结果收束

主清链使用 8-step micro-proxy、两个分区和三个 training seeds，得到一个 PlantSPD aggregate matched-budget observation。补充的 McNemar 分析和 50-step metadata 被单独保存，跨域和第二 SPD 扩展保留为后续运行协议。

## 9. 阶段反馈精化线

在 CIFAR-LT 迁移实验中，直接把内层精化固定到整个训练过程会把早期反馈修正延续到读出已经接近收敛的阶段。于是研究把动作写成阶段策略。Stage-TRR-10 在前 10 个 epoch 使用一次 detached readout refinement，随后切换到标准 logit-adjusted 联合训练。切换策略对应一个带阶段误差和切换代价的有限动作目标，文档中的界只针对同时成立的 FIT 证书事件。

## 10. 跨数据集扩展

Stage-TRR-10 与匹配的 Logit Adjustment 在 CIFAR-100-LT 和 CIFAR-10-LT（均为 imbalance ratio 100）上各运行三个 seeds。两个数据集的训练状态均先完成 FIT freeze，再进行一次性测试评估。CIFAR-100-LT 的 30-epoch exploratory comparison 为 Macro-F1 0.4440 对 0.4264、Tail-10% 0.1820 对 0.1273；CIFAR-10-LT 为 0.7799 对 0.7428、0.7797 对 0.7400。50-epoch CIFAR-100-LT 边界实验显示 Macro 的优势缩小、Tail 优势消失，这个阶段边界与 30-epoch 结果分开保存。

## 11. 公开复现整理

最终公开包把算法模块、训练入口、一次性评估入口、插件线、配置、图表脚本和证据摘要放在同一目录。原始图像、模型 checkpoint、服务器路径和访问凭据不进入公开仓库。所有数据文件通过环境变量接入，结果摘要使用相对路径和哈希记录。
