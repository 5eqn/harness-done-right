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

- 状态：已完成 ✅

#### 完成说明

已按照要求实现完整的 WebUI 系统：
1. **Python 库升级**：
   - 修改 HDR 库支持从 ~/.hdr/config.json 读取配置，同时兼容环境变量
   - 新增 LLM 调用日志功能，所有请求自动记录到 ~/.hdr/llm_logs.jsonl
   - 日志包含时间戳、请求类型、prompt、响应、token 用量、缓存状态等完整信息
   - 保留原有所有功能，所有单元测试通过

2. **后端实现（Express）**：
   - 配置管理接口：支持读取和保存 OpenRouter API Key 和模型配置
   - 日志查询接口：支持按月份聚合查询、按天查询、分页功能（每页100条）
   - 静态文件服务：自动托管前端构建产物
   - 端口：54789，无权限控制，适合本地使用

3. **前端实现（React + TailwindCSS）**：
   - 配置页面：可视化管理 OpenRouter API Key 和模型参数
   - 看板页面：
     - 月度 Token 消耗折线图，支持选择月份
     - 点击数据点可查看当日明细
     - 日志列表显示每次调用的时间、类型、token 用量、状态等
     - 分页导航，支持翻页查看历史记录
   - 响应式设计，界面简洁易用

4. **一键启动脚本**：
   - `start-webui.sh` 脚本自动安装依赖、构建前端、启动服务
   - 首次运行自动安装 node 依赖，后续直接启动

#### 新用法
```bash
# 启动 WebUI
cd hdr-skill
./start-webui.sh
# 打开浏览器访问 http://localhost:54789
```

#### 目录结构
```
hdr-skill/webui/
├── backend/          # Express 后端
├── frontend/         # React 前端
└── start-webui.sh    # 一键启动脚本
```

#### 后续改进建议
- 增加缓存命中统计和展示
- 支持更多模型提供商配置
- 增加日志搜索和过滤功能
- 支持导出统计报表
