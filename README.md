# Harness Done Right

让你的 Agent 把任务形式化定义出来，再**保证完成**。

## 极简示例：文本去 AI 味

### 1. 任务定义

你的 Agent 将需求形式化为一个数据结构：

```python
from hdr import BaseModel, verify, quote

class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        # 基于 LLM 的断言验证
        verify(f"{quote(self.original)} 和 {quote(self.humanized)} 表达的意思相同")
        verify(f"{quote(self.humanized)} 读起来像自然的人类书写文本")
```

### 2. 执行

你的 Agent 运行脚本来执行并验证任务：

```python
# 实例化时会触发 Pydantic 类型检查以及 LLM 验证
result = HumanizeText(
    original="AI 生成的技术术语...",
    humanized="一个清晰、人性化的解释..."
)
print("任务已验证:", result)
```

## 快速开始

### 环境配置

HDR 需要以下环境变量：
- `ANTHROPIC_API_KEY`：你的 Anthropic API 密钥（必需）
- `ANTHROPIC_MODEL`：模型名称（可选，默认为 claude-4.6-sonnet）
- `ANTHROPIC_BASE_URL`：API 基础 URL（可选）

我们正在为这个仓库开发更多使用场景。

## 开发

### 环境配置

```bash
uv venv .venv
uv pip install -e ".[dev]"
source .venv/bin/activate
```

接着启动 Claude Code 或你最爱的代码编辑工具，即可正常运行。
