## 你的任务

完成下面第一个未完成的任务。一定记住你只应该完成一个任务，不要做少也不要做多！使用 TodoWrite 工具记录下面的流程并严格按照其完成，注意使用 TodoWrite 的时候一定要记满下面全部八项：

- 完成任务前，创建一个新分支。
- 根据任务内容完成任务。
- 完整检查代码库、测试、文档，确保没有过时内容，不要让下一个工程师困惑。
- 视情况撰写单元测试。
- 确保所有测试成功且 Pyright 无错误、警告。
- 更新 TASKS.md 文档，标注任务已完成，并且简要概括完成情况，100 字以内。
- 完成任务后，把自己的工作合并至 main 分支。
- 如果还有剩余的任务，输出 {"all_done": false}，否则输出 {"all_done": true}

剩余任务将由其他工程师完成。

## 任务列表

### 0

- 状态：已完成

所有存 /tmp 文件里面的东西改为存在 /tmp/claude/hdr。os.makedirs exist_ok=True 这个目录。同步更新测试和文档等各种东西。因为沙箱只有权限访问 /tmp/claude。

### 1

- 状态：已完成

hdr.tasks.std 的文件怎么能用 LLM verify 检查是否存在……改成直接用 os 函数检查是否存在，并且增加单元测试

已完成：将 File 类的存在检查改为使用 os.path.exists()，移除了 verify/quote 依赖，增加了 5 个单元测试。

### 1.5

- 状态：未完成

迁移项目到 uv，同时配置好 PyRight，保证根目录下运行 pyright 无报错。遵循下面的文件结构：

hdr-skill/                      # 项目根目录（Claude Code skill）
│
├── SKILL.md                    # Claude Code skill 描述文件（保持在根目录）
│
├── pyproject.toml              # uv 项目配置 + 构建系统（替代 setup.py）
├── uv.lock                     # 锁定依赖版本（自动生成，不手动编辑）
├── .python-version             # 指定 Python 版本，例如 "3.12"
├── .gitignore                  # 忽略 __pycache__/, .venv/, dist/, *.egg-info/ 等
│
├── src/                        # 源码根目录（避免导入混乱）
│   └── hdr/                    # 主包（库名）
│       ├── __init__.py         # 仅做包标识 + 导出主要 API（从 core 导入）
│       ├── core.py             # 原 __init__ 中的业务逻辑移到这里
│       └── tasks/              # 子包
│           ├── __init__.py
│           └── std.py          # hdr.tasks.std 实现
│
├── tests/                      # 测试目录（与 src 平级）
│   ├── __init__.py
│   └── test_core.py            # 原 test_hdr.py 内容重命名
│
└── examples/                   # 可选：使用示例（替代原来的 example/）
    └── introduction_writing/
        ├── __init__.py
        ├── task.py
        └── work.py

### 2

- 状态：未完成

examples work.py 里面应该调用空参数 checkout，表示在空目录运行 claude。

### 3

- 状态：未完成

claude 应该在 /tmp 已创建的目录运行，checkout 不再返回目录而是记录在 hdr/__init__ 内部，verify 里面通过一个函数获取这个 /tmp 中已创建的目录，如果已经调用过 checkout 就用这个存储在内部的值，如果没有则创建一个不重名的空目录。

### 4

- 状态：未完成

checkout 额外接受 path 参数，必须是相对路径，相对于调用 python 时的 pwd，表示在这里运行 git archive。

### 5 

- 状态：未完成

不在 git 仓库中要报错并且让 Agent 引导用户在合适位置创建 git 仓库。

### 6

- 状态：未完成

checkout 时在 /tmp 创建目录时应该加上 path 变为绝对路径转义后的结果作为文件目录名前缀。
