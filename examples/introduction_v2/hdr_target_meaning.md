# Target Meaning

Introduce HDR as a harness design pattern for agent work.

The reader should understand that ordinary prose instructions can make an agent
sound careful while still leaving completion soft. HDR changes the completion
boundary by representing the intended outcome as an explicit Python task object:
typed fields hold required evidence, programmatic checks reject malformed shape,
and semantic `self.verify()` assertions reject missing meaning.

By the end, the reader should see HDR as the move from conversational promises
to enforceable construction. The agent is not merely asked to do better; it must
keep working until the final task instance constructs successfully.
