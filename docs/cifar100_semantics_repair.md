# CIFAR-100-LT implementation semantics repair

The first 30-epoch CIFAR matrix is retained as an executed artifact but is excluded from PV/Fixed algorithm and Fused-vs-Legacy claims. Legacy accumulated one correction loss per active group, whereas Fused reused one head output but added only one correction term. Therefore the paths had different objective weights.

The repaired Fused path multiplies the shared correction loss by the number of active logical groups. The repair is run in a fresh output directory with a one-step equivalence smoke before any matched matrix. ERM and Logit Adjustment rows are unaffected because they do not enter the correction branch.
