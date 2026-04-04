# Structured Task Execution for AI Coding Agents

When an AI coding agent works on a task, it faces a fundamental problem: **how do you know when the task is truly complete?** Traditional code passes type checks and tests, but qualitative requirements—like "the summary captures the key points" or "the code is readable"—cannot be validated by machines alone.

Many approaches treat task completion as a conversation: you ask the agent to do something, it responds with something, and you judge the quality by reading it. This works for small tasks but breaks down at scale. When an agent produces dozens of files or complex interdependent results, human review becomes a bottleneck, and agents cannot autonomously verify their own work.

The core challenge is that **AI agents need a way to formally define what "done" means for qualitative requirements**, then verify their own output against that definition—without humans in the loop for every decision.

Existing solutions include:
- **Unit tests with LLM-generated assertions**: Write tests that call an LLM to judge output quality. Effective but scattered across individual projects.
- **Conversation-based verification**: Treat verification as a separate chat turn. The agent explains what it did, the LLM judges it. Hard to scale or compose.
- **Rule-based validators**: Regex, schema checks, style linters. Fast but limited to mechanical properties.

What is missing is a **systematic pattern** where:
1. Task requirements are formalized as data structures, not prose instructions
2. Verification logic lives alongside the definition, not scattered across scripts
3. Successful completion is provable by construction, not by trust

This is the problem space this library operates in.
