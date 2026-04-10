# Mind & Humanities Tasks

Task types in `hdr.tasks.mind` for knowledge management, conceptual documentation, and human-centric validation use cases.

## Concept Tasks

### ConceptDescribed

Represents a documented concept within a context, with LLM validation of description quality.

**Fields:**
- `context: MarkdownFileWritten` — File explaining the parent context.
- `name: str` — Name of the concept.
- `description: MarkdownFileWritten` — File containing the concept description.

**Validates (via `self.verify`):**
- The description is written for readers who understand context but do not yet know the concept name.
- The concept name represents exactly one atomic idea.
- The description contains no time-sensitive terms without specifying an exact version or date.
- The description identifies a broader category and a distinguishing property.
- A reader familiar with context can classify concrete instances of the concept.

**Example:**
```python
from hdr.tasks.coding import MarkdownFileWritten
from hdr.tasks.mind import ConceptDescribed

concept = ConceptDescribed(
    context=MarkdownFileWritten(path="context.md"),
    name="Connection Pooling",
    description=MarkdownFileWritten(path="description.md"),
)
```
