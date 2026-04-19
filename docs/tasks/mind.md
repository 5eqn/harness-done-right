# Mind & Humanities Contracts

Contract types in `hdr.contracts.mind` for knowledge management, conceptual documentation, and human-centric validation use cases.

## Concept Contracts

### Concept

Represents a documented concept within a context, with LLM validation of description quality.

**Fields:**
- `context: MarkdownFile` — File explaining the parent context.
- `name: str` — Name of the concept.
- `description: MarkdownFile` — File containing the concept description.

**Validates (via `self.verify`):**
- The description is written for readers who understand context but do not yet know the concept name.
- The concept name represents exactly one atomic idea.
- The description contains no time-sensitive terms without specifying an exact version or date.
- The description identifies a broader category and a distinguishing property.
- A reader familiar with context can classify concrete instances of the concept.

**Example:**
```python
from hdr.contracts.coding import MarkdownFile
from hdr.contracts.mind import Concept

concept = Concept(
    context=MarkdownFile(path="context.md"),
    name="Connection Pooling",
    description=MarkdownFile(path="description.md"),
)
```
