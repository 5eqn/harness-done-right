from hdr.contracts.coding import MarkdownFile

from concept import Concept


target_reader = MarkdownFile(path="hdr_target_reader.md")
description = MarkdownFile(path="hdr_description.md")

concept_contract = Concept(
    target_reader=target_reader,
    name="HDR",
    description=description,
)

print("Concept contract completed:", concept_contract.name)
