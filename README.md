# Harness Done Right

让你的 Agent 把合约形式化定义出来，再**保证完成**。

## 极简示例：文本去 AI 味

### 1. 合约定义

你将需求形式化为一个数据结构：

```python
from hdr.contracts.std import BaseContract
from pydantic import Field

class HumanizeText(BaseContract):
    original: str = Field(description="Original AI-generated text")
    humanized: str = Field(description="Humanized version of the text")

    def __init__(self, **data):
        super().__init__(**data)
        # 基于 LLM 的断言验证
        self.llm_verify("original and humanized convey the same meaning")
        self.llm_verify("humanized reads like natural human-written text")
```

这一步也可以由 Agent 使用 [HDR-Define](skills/hdr-define/SKILL.md) 技能完成。

### 2. 执行

你的 Agent 使用 [HDR-Finish](skills/hdr-finish/SKILL.md) 技能实现合约。

```python
# 实例化时会触发 Pydantic 类型检查以及 LLM 验证
result = HumanizeText(
    original="AI 生成的技术术语...",
    humanized="一个清晰、人性化的解释..."
)
print("合约已验证:", result)
```

## 其他案例

- [代码报告](examples/holomotion_report/README.md)：生成详细代码报告，准确链接仓库源代码内容
- [学术报告](examples/academic_report/README.md)：生成结构化的学术报告，包括验证可访问的链接
- [产品介绍](examples/introduction/README.md)：生成符合 Storytelling 原则的产品介绍，由 LLM 自动验证

## 快速开始

### 环境配置

你只需要记住：Agent 需要知道如何进入存在 `hdr` 库的环境。推荐使用 `uv` 管理虚拟环境。

若已经处在 Python 项目中，你可以在你的 Python 环境安装 `hdr` 库：

```bash
git clone --depth 1 https://github.com/5eqn/harness-done-right.git ~/hdr-skill
uv pip install -e ~/hdr-skill
```

你也可以选择创建一个独立的虚拟环境来安装 `hdr` 库：

```bash
git clone --depth 1 https://github.com/5eqn/harness-done-right.git ~/hdr-skill
uv venv .venv
uv pip install -e ~/hdr-skill
```

在此之后，你需要安装技能：

```bash
# Claude Code
ln -sf ~/hdr-skill/skills/* ~/.claude/skills

# Codex
ln -sf ~/hdr-skill/skills/* ~/.agents/skills

# TODO 其他 Agent 请自行查阅文档
```

### 配置文件

HDR 在第一次调用 `self.llm_verify()` 时会检查 `~/.hdr/config.yaml`：
- 如果文件不存在，会自动创建一个模板文件
- 你需要在其中填写 `anthropic_auth_token`
- `anthropic_model` 和 `anthropic_base_url` 默认已经写好，可按需修改
- `verify_cache_dir` 默认是 `/tmp/claude/hdr_verify_cache`，可按需修改验证缓存目录

## 开发

### 环境配置

```bash
git clone --depth 1 https://github.com/5eqn/harness-done-right.git ~/hdr-skill
cd ~/hdr-skill
uv venv .venv
uv pip install -e ".[dev]"
source .venv/bin/activate
```

接着启动 Claude Code 或你最爱的代码编辑工具，即可正常运行。
