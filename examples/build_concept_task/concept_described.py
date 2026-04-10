from pydantic import Field
from hdr.tasks.std import Task
from hdr.tasks.coding import MarkdownFileWritten


class ConceptDescribed(Task):
    """Represents a documented concept within a context."""

    context: MarkdownFileWritten = Field(
        description="File explaining the parent context"
    )
    name: str = Field(description="Name of the concept")
    description: MarkdownFileWritten = Field(
        description="File containing the concept description"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.verify("""The description is written for readers who understand context but do not yet know name; it neither repeats basics from context nor presumes knowledge of sibling/descendant concepts.

Example that PASSES (score 5):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_good.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task is a Python class that inherits from the base `Task` class, defines\\ntyped fields with descriptions, and includes validation logic in its constructor.\\nIt differs from regular Python classes in that successful instantiation\\nautomatically proves the task has been completed to specification, including\\nboth type checks and LLM-powered quality validation.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)

Example that FAILS (score 1):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_bad1.md' # Path to the file
    content = '# HDR Task\\n\\nAs you already know from the HDR context, HDR is a task execution framework.\\nAn HDR Task is a class in that framework. You should already know how to use\\nthem from the documentation.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)""")
        self.verify("""The concept name represents exactly one atomic idea that cannot be meaningfully split into two independent concepts.

Example that PASSES (score 5):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_good.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task is a Python class that inherits from the base `Task` class, defines\\ntyped fields with descriptions, and includes validation logic in its constructor.\\nIt differs from regular Python classes in that successful instantiation\\nautomatically proves the task has been completed to specification, including\\nboth type checks and LLM-powered quality validation.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)

Example that FAILS (score 1):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task and Verification' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_bad2.md' # Path to the file
    content = '# HDR Task and Verification\\n\\nAn HDR Task and Verification is a Python class that includes both task\\ndefinition and verification logic.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)""")
        self.verify("""The description contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date.

Example that PASSES (score 5):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_good.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task is a Python class that inherits from the base `Task` class, defines\\ntyped fields with descriptions, and includes validation logic in its constructor.\\nIt differs from regular Python classes in that successful instantiation\\nautomatically proves the task has been completed to specification, including\\nboth type checks and LLM-powered quality validation.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)

Example that FAILS (score 1):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_bad3.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task is a Python class in the HDR framework. Currently, it uses\\nPydantic v2 for validation, and we recently added support for LLM verification.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)""")
        self.verify("""The description identifies (a) a broader category that name belongs to, and (b) a distinguishing property that separates it from other members of that category.

Example that PASSES (score 5):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_good.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task is a Python class that inherits from the base `Task` class, defines\\ntyped fields with descriptions, and includes validation logic in its constructor.\\nIt differs from regular Python classes in that successful instantiation\\nautomatically proves the task has been completed to specification, including\\nboth type checks and LLM-powered quality validation.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)

Example that FAILS (score 1):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_bad4.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task lets you define tasks that can be validated automatically. You use\\nthem in HDR framework.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)""")
        self.verify("""A reader familiar with context can determine for any concrete instance whether it belongs to name, with at most minor edge-case ambiguity.

Example that PASSES (score 5):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_good.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task is a Python class that inherits from the base `Task` class, defines\\ntyped fields with descriptions, and includes validation logic in its constructor.\\nIt differs from regular Python classes in that successful instantiation\\nautomatically proves the task has been completed to specification, including\\nboth type checks and LLM-powered quality validation.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)

Example that FAILS (score 1):
ConceptDescribed(
  context = MarkdownFileWritten(
    path = 'context.md' # Path to the file
    content = '# HDR Framework Context\\n\\nHDR (Harness Done Right) is a structured task execution framework for Claude Code.\\nIt allows formalizing tasks as Python classes where successful instantiation\\nserves as proof of task completion. Key components include:\\n\\n- Task base class with Pydantic validation\\n- LLM-powered `verify` assertions\\n- Standard task library for common operations\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File explaining the parent context
  name = 'HDR Task' # Name of the concept
  description = MarkdownFileWritten(
    path = 'description_bad5.md' # Path to the file
    content = '# HDR Task\\n\\nAn HDR Task is a piece of code that does something related to tasks in the HDR\\nframework.\\n' # Content of the file, auto-filled from disk (cannot be manually assigned)
  ) # File containing the concept description
)""")
