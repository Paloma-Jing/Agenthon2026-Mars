# 构建与打包

先完成根目录 README 中的安装与本地验证。镜像脚本必须显式设置自己的命名空间，源码中不包含个人默认账户。

```bash
export NAMESPACE=YOUR_NAMESPACE
./submission/build_and_push.sh --no-push
```

该命令执行 staged smoke、House stub、Linux/amd64 构建与容器接口检查。公开推送由移除 `--no-push` 显式触发；脚本会生成未跟踪的 `submission/image.json`。

打包器读取该镜像配置与 `models.json`，通过官方工具封装提交物。队伍编号通过 `--team-number` 传入，Team Key 通过隐藏提示或 `AGENTHON_TEAM_KEY` 环境变量传入。使用 `--descriptor-only` 只生成待检查描述文件；`--linked-descriptor` 支持已绑定账户的续传流程。完整参数见：

```bash
python tools/prepare_upload.py --help
```

默认 descriptor 的 license 字段为历史配置 `Apache-2.0`，打包时用 `--license` 指定实际获得授权的许可；该字段不构成本源码仓库的许可证。

输出位于 `submission/out/`，包括队伍标识或声明，不能提交 Git。镜像配置、历史 releases 和 ZIP 同样被忽略。上传、截止时间、账户绑定与公开镜像要求以主办方当前文档为准。README 不包含个人队伍编号、镜像地址或实际密钥。
