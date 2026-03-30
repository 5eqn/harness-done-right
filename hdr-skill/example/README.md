# HDR Example Workflow

This example demonstrates the recommended HDR workflow:

1. **`task.py`**: Immutable task specification - defines what needs to be done
2. **`work.py`**: Implementation - builds the final task instance

## How to Run

### 1. Activate virtual environment
```bash
# From the hdr-skill directory
source .venv/bin/activate
```

### 2. Run the example
```bash
cd example
python work.py
```

## Expected Output
```
Building documentation components...

✅ Task completed successfully!
Generated documentation for: HDR Documentation
Introduction length: 328 characters
Usage section has 2 code examples
```

## Key Points
- `task.py` is the formal specification and should not be modified once agreed upon
- `work.py` imports from `task.py` and implements the solution
- No state is persisted between runs - you can execute `work.py` as many times as you want
- LLM calls are automatically cached, so repeated runs are fast
