# HDR Framework Context

HDR (Harness Done Right) is a structured task execution framework for Claude Code.
It allows formalizing tasks as Python classes where successful instantiation
serves as proof of task completion. Key components include:

- Task base class with Pydantic validation
- LLM-powered `verify` assertions
- Standard task library for common operations
