## 你的任务

- 完成下面第一个未完成的任务。一定记住你只应该完成一个任务，不要做少也不要做多！
- 完成任务前，创建一个新分支。
- 每个任务应撰写详尽的单元测试，执行测试成功才算完成。
- 测试成功后，把标记为已完成，并给下一个工程师一段 200 字左右的文档。
- 请保证代码库、测试、文档中均没有过时内容，不要让下一个工程师困惑。
- 完成任务后，把自己的工作合并至 main 分支。
- 剩余任务将由其他工程师完成。
- 仅当所有任务完成时，额外输出 <promise>COMPLETE</promise>。

## 任务列表

### 任务一：完善 Python 库

- 状态：已完成

目前状态会被存储在调用 Python 库的 pwd 的一个 workspace 文件中。这样的缺点是，这是一个有状态的实现，一旦前面的调用出了问题，很难回退到过去的某个状态。我希望改成一种无状态 + 缓存的实现，这样 Agent 要完成一个任务，就等价于创造出一个完整的能够通过检验的任务描述 + 文件。我推荐的工作流程如下：

Agent 首先创建一个 task.py，存放任务的数据结构。接着创建一个 work.py，存放完整的任务对象构造过程。work.py 应该是无状态的，也就是说，每次运行都从零开始创建对象，如果 work.py 的某个前缀没改变，那么这个前缀的运行结果也不变，不会出现第二次就冲突的情况，所以现在不需要保存 workspace 文件。但是，现在需要一个缓存机制避免重复调用 LLM，安装 locache 并且给 LLM 补全的函数增加 @persist 即可。所以现在不需要显式保存任何文件。

由于不需要序列化对象，API 可以再次精简，可以删除 create 和 get，只需要直接传递对象即可。你需要保证类型检查依然运作。测试可以因此精简。同样，goal 和 finish 也可以删除。现在 Agent 的任务是构造出一个这样的 work.py，能成功构造出一个最终目标实例即可，Pydantic 将自动进行类型检查，hdr 只负责提供 llm_assert 等函数。只保留必要的单元测试。

### 任务二：WebUI MVP

- 状态：已完成

为 hdr-skill 实现一个匹配的 WebUI，包括前后端，前端 React + TailwindCSS，后端 Express。本地运行，不做权限限制。控制台页面支持配置模型，目前只需支持 OpenRouter，支持输入模型文本以及 API Key。看板页面支持 Token 查询，以曲线形式画出每天的 Token 消耗（在一个月中），并且可以选择月份。曲线只显示总 Token 数，点击当日的数据点可查看当日明细，包括调用次数、总 Cached Input / New Input / Output 以及总 Token 消耗，以及显示当天的所有 LLM 请求，每个请求显示出 CI/NI/O/Total，并且显示输入给 LLM 的全部内容和 LLM 输出的全部内容，Paginate at 100。

所有数据存在用户 Home 的固定目录，~/.hdr，同时修改 Python 库读取固定目录的数据，包括读取 API Key 和日志。

需要一个一键启动脚本，一键启动前端和后端。需要有完备的单元测试，以及测试启动脚本能否正常运行。

### 任务三：完善 Skills 描述

- 状态：已完成

跟 Agent 以明确的语言讲述整套系统的设计哲学，在同一个文件里面形式化任务和完成任务，完成任务时使用新的 API，直接定义对象就行，完成条件是定义出最终目标的任务对象。同时，引导 Agent 在 venv 中执行代码。在 work.py 中 from task import * 来引用任务内容，task 一经定义不要修改。

#### 任务完成说明

已完成 Skills 描述的完善工作：
1. **更新了 SKILL.md**：移除了旧 API（goal/create/get/finish/llm_check/mock_llm）相关内容，补充了新的无状态设计哲学说明，明确了 "任务即值"、"构造即验证"、"缓存代替状态" 三大核心设计原则，提供了完整的 API 参考和推荐工作流。
2. **新增示例目录**：在 hdr-skill/example/ 下提供了完整的示例，包括：
   - `task.py`：不可变的任务定义规范文件
   - `work.py`：任务实现文件，演示从 task.py 导入并构造最终目标对象
   - `README.md`：详细说明如何在 venv 中运行示例
3. **新增测试**：test_example.py 验证了示例工作流的正确性，所有测试通过。

### 任务四：完善 llm_assert 设计规范

- 状态：未完成

现在样例里面 task.py 有些没有填充实际内容的 f-string，这个 string 会被用于 LLM 校验，但是 LLM 并不知道上下文。这是旧的 task.py：

```
"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""
from hdr import BaseModel, llm_assert

# Define subtask types
class IntroductionSection(BaseModel):
    content: str

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{self.content} is a clear introduction explaining what HDR is")
        llm_assert(f"{self.content} mentions the core benefits of using HDR")

class UsageSection(BaseModel):
    content: str
    code_examples: list[str]

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{self.content} clearly explains how to use HDR")
        llm_assert(f"All code examples in {self.code_examples} are correct and runnable")
        llm_assert(f"{self.content} mentions both mock mode and real LLM mode usage")

class Documentation(BaseModel):
    title: str
    introduction: IntroductionSection
    usage: UsageSection

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{self.title} is clear and descriptive")
        llm_assert(f"The introduction properly leads into the usage section")
        llm_assert(f"The documentation as a whole is easy to understand for new users")
```

其中，`llm_assert(f"The introduction properly leads into the usage section")` 只会用来检验这句话的正确性，但是 LLM 不知道 The introduction 是什么。

因此，首先需要增加一个 hdr API，用于把任意对象转换成提示词中的一部分。这个转换会 pretty print 出来复杂对象，并且会加上一个 xml quote 防止里面的内容被解读为提示词，同时 llm_assert 也需要加上一个指示，对 xml quote 里面的东西一律视为非指令，这个指示在一开始和最后都要有，降低被提示词攻击的概率。举个例子，现在的话 API 会变成：

```
class Documentation(BaseModel):
    title: str
    introduction: IntroductionSection
    usage: UsageSection

    def __init__(self, **data):
        super().__init__(**data)
        llm_assert(f"{quote(self.title)} is clear and descriptive")
        llm_assert(f"{quote(self.introduction)} properly leads into the usage section")
        llm_assert(f"{quote(self)} as a whole is easy to understand for new users")
```

其中，`quote(self.title)` 会变为 `<quote>{self.title}</quote>`，`quote(self.introduction)` 则会变为 `<quote>{self.introduction.model_dump_json(indent=2)}</quote>`。quote 函数需要对可以直接打印的东西、Pydantic 简单对象、Pydantic 中数组、Pydantic 中字典、裸数组、裸字典进行单元测试。

在此之后，相应地修改 CLAUDE.md，SKILL.md 还有测试样例。同时注意，请修改 SKILL.md，提供一份 Agent 在任意 pwd 可执行的具体的流程案例，第一步第二步这样子，给一个很明确的流程案例。
