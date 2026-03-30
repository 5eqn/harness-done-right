## 你的任务

- 完成下面第一个未完成的任务。一定记住你只应该完成一个任务，不要做少也不要做多！
- 完成任务前，创建一个新分支。
- 每个任务应撰写详尽的单元测试，执行测试成功才算完成。
- 测试成功后，把标记为已完成，并给下一个工程师一段 200 字左右的文档。
- 请保证代码库、测试、文档中均没有过时内容，不要让下一个工程师困惑。
- 完成任务后，把自己的工作合并至 main 分支。
- 剩余任务将由其他工程师完成。
- 若没有剩余任务，则请额外输出 <promise>COMPLETE</promise>。

## 任务列表

### 任务一：完善 Python 库

- 状态：已完成 ✅

#### 完成说明

已按照要求重构 HDR Python 库为无状态实现：
1. **移除所有状态相关代码**：删除了 workbench、consumed 集合、goal 类型管理、pickle 持久化等有状态逻辑
2. **精简 API**：移除了 `goal()`/`create()`/`get()`/`finish()` 四个 API，现在直接构造对象即可，无需管理状态
3. **增加 LLM 调用缓存**：集成 `locache` 库，为 `llm_assert()` 和 `llm_check()` 增加 `@persist` 装饰器，自动缓存 LLM 调用结果避免重复请求
4. **保留核心功能**：完整保留 `llm_assert()`/`llm_check()`/`mock_llm` 功能，Pydantic 类型检查依然自动生效
5. **简化测试**：删除所有状态相关测试，保留核心功能测试，所有测试已通过

#### 新用法示例
```python
# task.py - 定义任务结构
from hdr import BaseModel, llm_assert

class HumanizeText(BaseModel):
    original: str
    humanized: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"<a>{self.original}</a> and <b>{self.humanized}</b> conveys the same meaning")
        llm_assert(f"{self.humanized} reads like natural human-written text")
```

```python
# work.py - 构造任务（完全无状态，可重复运行）
from hdr import mock_llm
from task import HumanizeText

mock_llm.enable()
result = HumanizeText(original="Text with AI", humanized="Text without AI")
print("Task completed:", result)
```

#### 变更说明
- 无需再管理任务 ID、消费状态、目标类型等，代码更简洁
- LLM 调用自动缓存，相同查询不会重复消耗 Token
- 所有类型检查和断言逻辑保持不变，向下兼容任务定义方式
- 依赖新增 `locache` 包，已添加到 setup.py 依赖列表

### 任务二：WebUI MVP

- 状态：未完成

为 hdr-skill 实现一个匹配的 WebUI，包括前后端，前端 React + TailwindCSS，后端 Express。本地运行，不做权限限制。控制台页面支持配置模型，目前只需支持 OpenRouter，支持输入模型文本以及 API Key。看板页面支持 Token 查询，以曲线形式画出每天的 Token 消耗（在一个月中），并且可以选择月份。曲线只显示总 Token 数，点击当日的数据点可查看当日明细，包括调用次数、总 Cached Input / New Input / Output 以及总 Token 消耗，以及显示当天的所有 LLM 请求，每个请求显示出 CI/NI/O/Total，并且显示输入给 LLM 的全部内容和 LLM 输出的全部内容，Paginate at 100。

所有数据存在用户 Home 的固定目录，~/.hdr，同时修改 Python 库读取固定目录的数据，包括读取 API Key 和日志。

需要一个一键启动脚本，一键启动前端和后端。需要有完备的单元测试，以及测试启动脚本能否正常运行。
