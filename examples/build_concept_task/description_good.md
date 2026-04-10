# HDR Task

An HDR Task is a Python class that inherits from the base `Task` class, defines
typed fields with descriptions, and includes validation logic in its constructor.
It differs from regular Python classes in that successful instantiation
automatically proves the task has been completed to specification, including
both type checks and LLM-powered quality validation.
