"""Prompt queue: run one generation per line of the prompt box.

Off by default. Without it, extra prompt lines keep their usual meaning
(they're added as extra positive prompts to the same image).
"""


def split_prompt_queue(prompt: str | None) -> list[str]:
    """Return the non-empty, stripped lines of ``prompt`` in order."""
    return [line.strip() for line in (prompt or '').splitlines() if line.strip()]


def queue_progress_title(index: int, total: int, prompt: str, max_len: int = 80) -> str:
    """Progress-bar title shown before each queued prompt starts (1-based index)."""
    shown = prompt if len(prompt) <= max_len else prompt[:max_len - 1] + '…'
    return f'Prompt {index}/{total}: {shown}'
