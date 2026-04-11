# HDR

It is 11:47 p.m., and an AI agent has 9,000 tokens of chat history, 4 tool
results, and a half-finished migration plan in its context window. The model
says the plan is "done," but the harness never made rollback, backfill timing,
or verification commands into objects the agent had to satisfy. You can ask the
agent to be more careful, but the next run may still end with the same soft
promise and a different-looking gap.

The natural move is to keep the task in prose: "write a complete migration
plan" or "make this explanation good." That feels reasonable because agents
already understand prose, and a general harness can pass that prose through the
model, manage context, call tools, and return the final answer.

The default breaks when the harness must make "done" repeatable instead of
conversational. If the plan must include rollback steps, data ownership,
expected runtime, and verification commands, free-form context gives the agent
no stable task object to construct, no typed dependencies to inspect, and no
single failure point that says which requirement was missed.

HDR is the move of making the harness guide the agent through explicit task
objects that must successfully instantiate. In Harness Done Right, the contract
lives as a Python task class: fields name the required evidence, Pydantic checks
the shape, and `self.verify()` assertions judge semantic conditions against the
constructed object.

An agent system counts as using HDR only when its harness routes the work
through that explicit task contract and completion means the final task instance
constructs successfully. A normal prompt, a checklist in memory, or a post-hoc
review is not HDR by itself, because it leaves "done" as a conversational claim
instead of a harnessed object that can reject bad work.

For an AI agent, HDR changes the working loop. The agent does not merely answer
the request; it creates or satisfies a task definition, builds the needed files
or dependency objects, instantiates the final class, and treats
`ValidationError` or `AssertionError` as feedback to keep working until the task
passes.
