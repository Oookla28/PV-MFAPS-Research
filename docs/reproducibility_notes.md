# Reproducibility notes

原始远程 run 的 PR runner 曾经把 sibling 目录加入 `sys.path`，而 freeze manifest 记录了另一份代码目录。新的复现包把实际运行模块合并到 `src/validated_runtime/`，并把公开算法入口放在 `src/model/`。因此，新运行应生成新的 code SHA256 manifest，不直接复用历史 freeze hash。

结果包不复制大数据和 checkpoint。所有外部输入由 `DATA_ROOT` 和 `CHECKPOINT_ROOT` 提供，runner 会把输入路径和哈希写进 manifest。`._*` macOS 元数据文件被排除。

状态文件可以在服务器或共享存储中保存。为了让结果包保持可迁移，默认 artifacts 只保留 JSON、Markdown、校准和小型行级结果。若需要完整重放 HELD，必须另外提供与 freeze manifest 完全匹配的 state files。
