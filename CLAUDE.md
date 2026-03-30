# Harness Done Right

## 整体需求

一个给 Claude Code 的 skills，引导 Claude Code 完成一个任务的时候，先去把自己的任务用 Python 形式化出来，再用 Python 库去根据形式化的任务去分阶段完成。每个任务是一个 Python 类，构造函数的参数就是完成这个任务的要求。任务的实例存储在工作台中，表示这个任务完成的证明，并且可作为参数填充到其他任务中。

## 例子：用户让 Claude 把一段文本去除 AI 味

首先，Claude 在任务定义模式下，把任务形式化为下面的形式：

```humanize_text.py
from hdr import BaseModel, llm_assert

# 继承BaseModel获得自动类型检查功能
class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"<a>{self.original}</a> and <b>{self.humanized}</b> conveys the same meaning")
        llm_assert(f"{self.humanized} reads like natural human-written text")
```

Claude 将任务定义和概括含义发给用户检查，用户确认后直接构造任务实例即可：

```python
from hdr import save_config
from humanize_text import HumanizeText

# 启用Mock模式
save_config({"openrouter_model": "mock"})

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

LLM 操作暂时只支持 OpenRouter 就行。

---

## 项目结构

```
hdr/
├── CLAUDE.md                    # 项目说明文档
└── hdr-skill/                   # HDR技能包
    ├── SKILL.md                 # 技能说明文档
    ├── test_hdr.py              # 单元测试
    ├── venv/                    # Python虚拟环境（自动创建）
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
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install openai pytest

# 安装 hdr 库
cd scripts
pip install -e .
```

### 配置方式
配置方式：
**配置文件**（可通过WebUI管理）：配置保存在 `~/.hdr/config.json`，包含以下字段：
- `openrouter_api_key`: OpenRouter API Key
- `openrouter_model`: 模型名称，设置为 `"mock"` 时启用Mock模式（无需API Key，所有断言自动通过）

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

## API 说明（已实现）

### 核心API
| 函数 | 功能 |
|------|------|
| `BaseModel` | 基类，所有任务类继承此类获得自动类型检查能力 |
| `llm_assert(condition: str)` | 用LLM验证条件是否成立，仅当LLM评分为5分时通过，失败则抛出包含LLM思考过程和分数的异常 |
| `load_config()` | 加载配置文件，返回配置字典 |
| `save_config(config)` | 保存配置到文件 |

### Mock模式使用说明
在开发和测试时使用Mock模式可以避免真实API调用：
```python
from hdr import save_config
save_config({"openrouter_model": "mock"})  # 启用Mock模式，所有LLM断言自动通过
```

## 测试运行
```bash
cd hdr-skill
source venv/bin/activate
python -m pytest test_hdr.py -v
```

## 后续开发注意事项

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

### LLM 调用日志
所有 `llm_assert` 调用都会自动记录日志，包含：
- 时间戳、请求类型、prompt 内容、响应内容
- Token 用量明细（输入、输出、总计）
- 缓存状态、调用是否成功、错误信息

### 类型系统扩展
1. ✅ 已集成Pydantic实现完整的运行时类型校验，所有继承BaseModel的任务类自动获得类型检查能力
2. 支持所有Pydantic类型：str, int, float, bool, list, dict, datetime等，以及嵌套类型和自定义类型
3. 预置类型后续扩展支持：File、Path等特殊类型
4. 自动支持自定义类型的序列化和反序列化（基于Pydantic的model_dump功能）
