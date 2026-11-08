# Protocols

## FIT/HELD boundary

所有校准、动作选择、成本预算和训练过程只使用 FIT。训练 JSON 的 `held_reads` 必须为 0。全部 FIT states 写入 freeze manifest 后，运行一次 HELD 评估。HELD 不参与动作选择、early stopping 或阈值搜索。

## Matched cost

每个 arm 记录实际 reference action、solver step、native/reference forward/backward、FLOPs 和 CUDA event wall-clock。额外 probe 计入该 arm 的成本。Fused 与 legacy 的 physical encoder forward 单独记录，不能用逻辑 action 数替代物理调用数。

## Shared partition policy

PR-PV-MFAPS 为两个分区使用同一个类别动作向量。风险向量先按分区聚合，再由 KL ambiguity objective 选择策略。分别为分区求策略会改变研究问题，因此不放进主协议。

## Reproducibility manifest

运行 manifest 至少包含数据和 checkpoint 哈希、代码清单哈希、配置哈希、GPU 名称、CUDA/PyTorch 版本、partition、checkpoint seed、training seeds、steps、batch size、action grid、Fused 开关和输出文件路径。
