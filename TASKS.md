## 你的任务

- 完成下面第一个未完成的任务。一定记住你只应该完成一个任务，不要做少也不要做多！
- 完成任务前，创建一个新分支。
- 如果必要，每个任务应撰写详尽的单元测试，执行测试成功才算完成。
- 测试成功后，把标记为已完成，并给下一个工程师一段 200 字左右的文档。
- 请保证代码库、测试、文档中均没有过时内容，不要让下一个工程师困惑。
- 完成任务后，把自己的工作合并至 main 分支。
- 剩余任务将由其他工程师完成。

## 任务列表

### 任务一：文件使用

删除和 project_path 相关的所有东西，保证零配置，~/.hdr 目录将不再被使用。
增加 checkout API，一开始调用 checkout({commit})，表示 Claude 将在这个 git commi
t 运行。checkout 运行时，会将这个 commit 时整个仓库的状态用 git archive 放在 /tm
p/{commit}，如果已经放置了则不重新运行 git archive，并直接返回这个 /tmp 目录。
后续所有 Claude 都将在 /tmp/{commit} 运行。同时 @persist 的 _verify 需要依赖 c
ommit hash。如果没有 commit hash，给 _verify 传递空字符串，并且在 /tmp 创建一个
空目录在里面运行 Claude Code. 同时和 hdr.py 并列一个 tasks 文件夹，通过 hdr.task
s import，暂时只存放 __init__.py 和一个 std.py（hdr.tasks.std），里面存放一个 File（任务），创建的时候会校验文件（文档中说明偏好相对路径）是否存在，以及 brainstorm 一些可能常用的任务进去。

最终，使用流程就是一开始加一个 checkout({commit})，后面完全没有变化。

### 任务二：跑通写项目介绍的任务

启动 hdr skill，读取 SKILL.md，完善 hdr-skill/example/introduction-writing，跑通 work.py 并增加 README.md

## 碎碎念

检验经常因为自己不知道在检验什么而失败。我需要让 LLM 明确判断是检验器的问题还是实现的问题。

我觉得在设计任务的时候，先设计一个 Context 对象，讲清楚在干什么，后续所有检验都带上 Context，会好很多。

我觉得还是充一个 API 比较合适，这种检验好像不太烧 token。
