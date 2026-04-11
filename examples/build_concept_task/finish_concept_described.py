from hdr.tasks.coding import MarkdownFileWritten

from concept_described import ConceptDescribed


target_reader = MarkdownFileWritten(path="hdr_target_reader.md")
description = MarkdownFileWritten(path="hdr_description.md")

concept_described = ConceptDescribed(
    target_reader=target_reader,
    name="HDR",
    description=description,
)

print("ConceptDescribed task completed:", concept_described.name)
