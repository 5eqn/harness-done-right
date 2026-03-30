# Harness Done Right

## 整体需求

一个给 Claude Code 的 skills，引导 Claude Code 完成一个任务的时候，先去把自己的任务用 Python 形式化出来，再用 Python 库去根据形式化的任务去分阶段完成。每个任务是一个 Python 类，构造函数的参数就是完成这个任务的要求。任务的实例存储在工作台中，表示这个任务完成的证明，并且可作为参数填充到其他任务中。

## 例子：用户让 Claude 把一段文本去除 AI 味

首先，Claude 在任务定义模式下，把任务形式化为下面的形式：

```humanize_text.py
from hdr import BaseModel, llm_assert, quote

# 继承BaseModel获得自动类型检查功能
class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{quote(self.original)} and {quote(self.humanized)} conveys the same meaning")
        llm_assert(f"{quote(self.humanized)} reads like natural human-written text")
```

Claude 将任务定义和概括含义发给用户检查，用户确认后直接构造任务实例即可：

```python
from hdr import save_config
from humanize_text import HumanizeText

# 运行时自动触发类型检查和LLM断言
result = HumanizeText(original="Text with AI", humanized="Text without AI")
print("Task completed:", result)
```

LLM 调用会自动记录日志，可通过 WebUI 查看 token 消耗和调用历史。

## 嵌套结构

支持任意深度的嵌套类型结构，例如：

```python
from hdr import *

# 所有任务类继承BaseModel获得自动类型检查
class D(BaseModel):
    value: str

class E(BaseModel):
    value: int

class B(BaseModel):
    title: str
    d: D
    e: E

class C(BaseModel):
    data: list[int]

class A(BaseModel):
    title: str
    b: B
    c: C

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{self.title} is the same as {self.b.title}")
```

直接构造对象即可，所有嵌套类型会自动递归校验：

```python
result = A(
    title="Some title",
    b=B(
        title="Some title",
        d=D(value="d-value"),
        e=E(value=42)
    ),
    c=C(data=[1, 2, 3])
)
```

所有 LLM 断言会自动执行，类型错误会立即抛出。

## 快速开始

### 1. 安装配置
```bash
# 克隆项目
git clone <repo-url>
cd hdr/hdr-skill

# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install openai pytest pydantic locache

# 安装hdr库
cd scripts
pip install -e .
```

### 2. 配置API密钥
```python
from hdr import save_config

save_config({
    "openrouter_api_key": "your-openrouter-api-key",
    "openrouter_model": "anthropic/claude-3-opus"  # 可替换为其他OpenRouter支持的模型
})
```

### 3. 定义并运行任务
```python
from hdr import BaseModel, llm_assert, quote

# 定义任务类
class TextSummary(BaseModel):
    original: str
    summary: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{quote(self.summary)} accurately summarizes the main points of {quote(self.original)}")
        llm_assert(f"{quote(self.summary)} is concise and contains no redundant information")

# 执行任务
result = TextSummary(
    original="The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully in the sun, and didn't even notice the fox jumping over him.",
    summary="A fox jumps over a sleeping dog that is sunbathing."
)
print("Task completed successfully!")
```

LLM 操作基于 OpenRouter 平台，支持所有 OpenRouter 接入的大模型。

---

## 项目结构

```
hdr/
├── CLAUDE.md                    # 项目说明文档
└── hdr-skill/                   # HDR技能包
    ├── .venv/                   # Python虚拟环境（自动创建）
    ├── SKILL.md                 # 技能说明文档
    ├── test_hdr.py              # 单元测试
    ├── start-webui.sh           # 一键启动WebUI脚本
    ├── webui/                   # WebUI界面
    │   ├── backend/             # Express后端
    │   └── frontend/            # React前端
    └── scripts/
        ├── hdr.py               # HDR核心库
        └── setup.py             # 安装配置
```

## 开发环境配置

### 虚拟环境配置
```bash
# 创建虚拟环境
cd hdr-skill
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
pip install openai pytest pydantic locache

# 安装 hdr 库
cd scripts
pip install -e .
```

### 配置方式
配置方式：
**配置文件**（可通过WebUI管理）：配置保存在 `~/.hdr/config.json`，包含以下字段：
- `openrouter_api_key`: OpenRouter API Key
- `openrouter_model`: 模型名称

## WebUI 使用
```bash
# 一键启动WebUI（自动安装依赖、构建、启动服务）
cd hdr-skill
./start-webui.sh

# 访问 http://localhost:54789
```

WebUI 功能：
- **配置页面**：可视化管理 OpenRouter API Key 和模型参数
- **看板页面**：查看 token 消耗统计、折线图展示月度消耗、查看历史调用明细、分页浏览请求日志

## 功能特性（已实现）

### 核心功能
- ✅ **类型安全**：基于Pydantic实现完整的运行时类型校验，自动校验任务类的字段类型、嵌套结构
- ✅ **LLM断言**：使用`llm_assert`通过大模型验证自定义条件，仅当评分为5分时通过，失败抛出包含完整思考过程的异常
- ✅ **自动缓存**：相同的LLM断言请求自动缓存，避免重复调用和token浪费
- ✅ **流式输出**：未缓存的LLM调用实时打印流式输出内容，可观察思考过程
- ✅ **自动重试**：网络错误、API调用失败时自动重试10次，提升可靠性
- ✅ **调用日志**：自动记录所有LLM调用的完整日志，包含token用量、请求响应内容、时间戳等
- ✅ **提示词安全**：`quote`函数自动处理任意类型对象，防止提示词注入攻击

### 核心API
| 函数 | 功能 |
|------|------|
| `BaseModel` | 基类，所有任务类继承此类获得自动类型检查能力 |
| `llm_assert(condition: str)` | 用LLM验证条件是否成立，仅当LLM评分为5分时通过，失败则抛出包含LLM思考过程和分数的异常 |
| `quote(obj: Any)` | 安全地将任意对象转换为可嵌入提示词的格式，防止提示词注入，自动处理字符串、数字、Pydantic模型、列表、字典等 |
| `load_config()` | 加载配置文件，返回配置字典 |
| `save_config(config)` | 保存配置到文件 |


## 测试运行
```bash
cd hdr-skill
source .venv/bin/activate
python -m pytest test_hdr.py -v
```

## 后续开发注意事项

### 重要约束
- **Mock模式仅限测试使用**：`"mock"` 模型只能用于 pytest 单元测试中，绝对禁止在真实执行环境或面向用户/Agent的文档中提及或使用Mock模式。Mock模式仅作为内部测试工具存在，不对外暴露。

### 技能开发规范
1. 所有新功能必须包含对应的单元测试，使用Mock LLM模式进行测试
2. 保持API向后兼容性，新增功能不破坏现有调用方式
3. 所有持久化操作保持幂等性，避免重复执行导致数据损坏
4. 错误信息必须清晰明确，包含可操作的修复指引

### 数据持久化
1. 所有配置和日志存储在用户主目录 `~/.hdr/` 目录下
2. `config.json`：保存 OpenRouter API Key、模型等配置
3. `llm_logs.jsonl`：所有 LLM 调用日志，包含完整的请求、响应和 token 用量
4. `cache/`：LLM 调用缓存目录，自动避免重复请求相同内容

### LLM 调用特性
- **流式输出**：未命中缓存的LLM调用会实时打印流式输出内容，前缀为`[LLM Streaming Output]:`，可观察LLM的完整思考过程
- **重试机制**：遇到网络错误、API调用失败等异常时，自动重试最多10次，每次间隔1秒，重试失败后才抛出异常
- **日志记录**：所有未缓存的LLM调用都会自动记录日志，包含：
  - 时间戳、请求类型、prompt 内容、完整响应内容
  - Token 用量明细（输入、输出、总计）
  - 调用是否成功、错误信息（异常情况）
- **缓存机制**：相同的`llm_assert`请求自动缓存，缓存命中时直接返回结果，不产生新的LLM调用和日志

### 类型系统扩展
1. ✅ 已集成Pydantic实现完整的运行时类型校验，所有继承BaseModel的任务类自动获得类型检查能力
2. 支持所有Pydantic类型：str, int, float, bool, list, dict, datetime等，以及嵌套类型和自定义类型
3. 预置类型后续扩展支持：File、Path等特殊类型
4. 自动支持自定义类型的序列化和反序列化（基于Pydantic的model_dump功能）
