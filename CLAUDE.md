# Harness Done Right

## 整体需求

一个给 Claude Code 的 skills，引导 Claude Code 完成一个任务的时候，先去把自己的任务用 Python 形式化出来，再用 Python 库去根据形式化的任务去分阶段完成。每个任务是一个 Python 类，构造函数的参数就是完成这个任务的要求。任务的实例存储在工作台中，表示这个任务完成的证明，并且可作为参数填充到其他任务中。

## 例子：用户让 Claude 把一段文本去除 AI 味

首先，Claude 在任务定义模式下，把任务形式化为下面的形式：

```humanize_text.py
from hdr import BaseModel, verify, quote

# 继承BaseModel获得自动类型检查功能
class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.original)} and {quote(self.humanized)} conveys the same meaning")
        verify(f"{quote(self.humanized)} reads like natural human-written text")
```

Claude 将任务定义和概括含义发给用户检查，用户确认后直接构造任务实例即可：

```python
from hdr import save_config
from humanize_text import HumanizeText

# 运行时自动触发类型检查和LLM断言
result = HumanizeText(original="Text with AI", humanized="Text without AI")
print("Task completed:", result)
```

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

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{self.title} is the same as {self.b.title}")
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
pip install pytest pydantic locache

# 安装hdr库
cd scripts
pip install -e .
```

### 2. 配置项目路径
```python
from hdr import save_config

save_config({
    "project_path": "/path/to/your/project"  # Claude Code将在此目录运行
})
```

### 3. 定义并运行任务
```python
from hdr import BaseModel, verify, quote

# 定义任务类
class TextSummary(BaseModel):
    original: str
    summary: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.summary)} accurately summarizes the main points of {quote(self.original)}")
        verify(f"{quote(self.summary)} is concise and contains no redundant information")

# 执行任务
result = TextSummary(
    original="The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully in the sun, and didn't even notice the fox jumping over him.",
    summary="A fox jumps over a sleeping dog that is sunbathing."
)
print("Task completed successfully!")
```

验证使用 Claude Code CLI 调用 Claude 验证条件。

---

## 项目结构

```
hdr/
├── CLAUDE.md                    # 项目说明文档
└── hdr-skill/                   # HDR技能包
    ├── .venv/                   # Python虚拟环境（自动创建）
    ├── SKILL.md                 # 技能说明文档
    ├── test_hdr.py              # 单元测试
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
pip install pytest pydantic locache

# 安装 hdr 库
cd scripts
pip install -e .
```

### 配置方式
**配置文件**：配置保存在 `~/.hdr/config.json`，包含以下字段：
- `project_path`: 项目路径，Claude Code 将在此目录运行

## 功能特性（已实现）

### 核心功能
- ✅ **类型安全**：基于Pydantic实现完整的运行时类型校验，自动校验任务类的字段类型、嵌套结构
- ✅ **LLM断言**：使用`verify`通过Claude Code验证自定义条件，仅当评分为5分时通过，失败抛出包含完整思考过程的异常
- ✅ **自动缓存**：相同的verify请求自动缓存，避免重复调用
- ✅ **提示词安全**：`quote`函数自动处理任意类型对象，防止提示词注入攻击

### 核心API
| 函数 | 功能 |
|------|------|
| `BaseModel` | 基类，所有任务类继承此类获得自动类型检查能力 |
| `verify(condition: str)` | 用Claude Code验证条件是否成立，仅当评分为5分时通过，失败则抛出包含思考过程和分数的异常 |
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

### 技能开发规范
1. 所有新功能必须包含对应的单元测试
2. 保持API向后兼容性，新增功能不破坏现有调用方式
3. 所有持久化操作保持幂等性，避免重复执行导致数据损坏
4. 错误信息必须清晰明确，包含可操作的修复指引

### 数据持久化
1. 所有配置存储在用户主目录 `~/.hdr/` 目录下
2. `config.json`：保存项目路径等配置

### 验证特性
- **缓存机制**：相同的`verify`请求自动缓存，缓存命中时直接返回结果，不产生新的Claude调用
- **错误处理**：验证失败时抛出包含Claude思考过程和分数的AssertionError

### 类型系统扩展
1. ✅ 已集成Pydantic实现完整的运行时类型校验，所有继承BaseModel的任务类自动获得类型检查能力
2. 支持所有Pydantic类型：str, int, float, bool, list, dict, datetime等，以及嵌套类型和自定义类型
3. 预置类型后续扩展支持：File、Path等特殊类型
4. 自动支持自定义类型的序列化和反序列化（基于Pydantic的model_dump功能）
