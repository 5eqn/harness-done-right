# Harness Done Right

## 整体需求

一个给 Claude Code 的 skills，引导 Claude Code 完成一个任务的时候，先去把自己的任务用 Python 形式化出来，再用 Python 库去根据形式化的任务去分阶段完成。每个任务是一个 Python 类，构造函数的参数就是完成这个任务的要求。任务的实例存储在工作台中，表示这个任务完成的证明，并且可作为参数填充到其他任务中。

## 例子：用户让 Claude 把一段文本去除 AI 味

首先，Claude 在任务定义模式下，把任务形式化为下面的形式：

```humanize_text.py
from hdr import BaseModel, llm_assert

def llm_check_humanized(text: str):
  # make llm vote on whether this text is AI generated based on some prebuilt principles
  llm_assert(f"{text} reads like natural human-written text")

# 继承BaseModel获得自动类型检查功能
class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"<a>{self.original}</a> and <b>{self.humanized}</b> conveys the same meaning")
        llm_check_humanized(self.humanized)
```

Claude 将任务定义和概括含义发给用户检查，用户确认后 Claude 直接调用 Python 库去指定任务为目标：

```
from hdr import *
goal(HumanizeText)
```

接着 Claude 会开始试图完成任务。由于任务数据结构简单，Claude 不需要启动子代理，直接进行完成：

```
from hdr import *
create("a", HumanizeText("Text with AI", "Text without AI"))
```

使用 create 试图创建重名对象时会报错，如果失败则会返回报错内容（一般是 LLM 提供的未通过检验原因和改进建议），如果成功则会输出任务状态，并且把对象用 pickle 永久化存储，这样后续可以用 get("a") 读取。若失败，把没通过检验的理由喂给 Claude，Claude 进行重试：

```python
from hdr import *
create("a", HumanizeText(original="Text with AI", humanized="Text without AI"))
```

若成功，进行下一步。最终需要把匹配类型的对象用于完成目标：

```python
from hdr import *
finish(get("a"))
```

如果类型正确，则会引导 Claude 汇报最终结果，Python 库会留下全量日志，Claude 针对这一任务总结永久性经验。

## 递归调用

比如首先定义了这样的任务：

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

Claude 发给用户检查，确认后定义任务：

```
from hdr import *
goal(A)
```

这个时候需要递归构建。首先构建 D 存为 "d"：

```python
from hdr import *
create("d", D(value="d-value"))
```

接着构建 E 存为 "e"，再构建 B 存为 "b"：

```python
from hdr import *
create("e", E(value=42))
create("b", B(title="B Title", d=get("d"), e=get("e")))
```

再构建 C 存为 "c"，这时候要构造 A 则是：

```python
from hdr import *
create("c", C(data=[1, 2, 3]))
create("a", A(title="Some title", b=get("b"), c=get("c")))
```

这套操作也可以用来处理数组等任意东西。核心就是通过 get 来读取动态对象。每个对象只能被使用一次，例如下面的会报错：

```python
from hdr import *
create("b", B(get("d"), get("e")))
create("b", B(get("d"), get("e")))  # 报错："d" 已被消耗
```

如果有类型错误，例如下面的也会报错：

```python
from hdr import *
create("b", B(get("e"), get("d")))
```

由于 B 的第一个参数需要 D 类型，但是提供了 get("e") : E，所以调用 create 的时候会报错。get 只是拿出动态对象。

预置类型不仅要有 str，还要有文件、数字等各种东西。hdr 需要有能力在 create 之后把对象永久化存储起来，并且记录详细日志。

LLM 操作暂时只支持 OpenRouter 就行。

---

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

### 环境变量配置
需要配置以下环境变量才能使用真实 LLM 功能：
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_MODEL="anthropic/claude-3-opus"  # 或其他支持的模型
```

## API 说明（已实现）

### 核心API
| 函数 | 功能 |
|------|------|
| `goal(task_type: Type)` | 设置需要完成的目标任务类型 |
| `create(id: str, instance: Any)` | 创建任务实例并存入工作台 |
| `get(id: str) -> Any` | 从工作台获取任务实例（标记为已消耗，不可重复使用），别名 `with_` |
| `finish(instance: Any)` | 完成目标任务，实例类型必须匹配目标类型 |

### LLM工具API
| 函数 | 功能 |
|------|------|
| `llm_assert(condition: str)` | 用LLM验证条件是否成立，失败则抛出包含解释的异常 |
| `llm_check(predicate: str, value: Any) -> bool` | 用LLM检查谓词是否适用于给定值，返回布尔结果 |

### 测试API
| 函数 | 功能 |
|------|------|
| `mock_llm.enable()` | 启用Mock LLM模式，不需要真实API调用 |
| `mock_llm.disable()` | 禁用Mock LLM模式 |
| `mock_llm.add_response(response: bool)` | 添加Mock响应，按顺序匹配后续LLM调用 |

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
1. 当前使用pickle进行对象序列化，后续考虑替换为更安全的序列化方式（如dill、JSON Schema等）
2. 工作bench数据存储在 `.hdr_workbench.pkl`，包含所有任务实例、消费状态和当前目标
3. 每次调用`goal()`/`create()`/`get()`/`finish()`都会自动保存状态

### 类型系统扩展
1. ✅ 已集成Pydantic实现完整的运行时类型校验，所有继承BaseModel的任务类自动获得类型检查能力
2. 支持所有Pydantic类型：str, int, float, bool, list, dict, datetime等，以及嵌套类型和自定义类型
3. 预置类型后续扩展支持：File、Path等特殊类型
4. 自动支持自定义类型的序列化和反序列化（基于Pydantic的model_dump功能）
