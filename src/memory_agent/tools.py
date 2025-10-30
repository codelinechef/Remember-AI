"""Define the agent's tools."""

import uuid
from datetime import datetime
from typing import Annotated

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore


async def upsert_memory(
    content: str,
    context: str,
    importance: int | None = None,
    tags: list[str] | None = None,
    *,
    memory_id: uuid.UUID | None = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    timestamp = datetime.now().isoformat()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={
            "content": content,
            "context": context,
            "importance": importance,
            "tags": tags or [],
            "timestamp": timestamp,
        },
    )
    return f"Stored memory {mem_id}"


async def search_memories(
    query: str | None = None,
    limit: int = 10,
    *,
    # Hidden injected args
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Search for memories for a user by semantic query or list recent.

    Args:
        query: Text to search against memory content/context. If None, returns recent.
        limit: Max number of memories to return.
    """
    # If query is None, use an empty string to get recent items.
    q = query or ""
    results = await store.asearch(("memories", user_id), query=q, limit=limit)
    # Return a concise, tool-friendly summary string
    formatted = [
        f"{r.key}: {getattr(r, 'value', {})} (score={getattr(r, 'score', None)})"
        for r in results
    ]
    return "\n".join(formatted) if formatted else "No memories found"
