"""Draft mode: fast previews, then a full-quality render of the one you like.

Drafts use a deterministic sampler (DPM++ 2M, karras) at few steps. With the
default SDE sampler, changing the step count changes the noise path and the
picture; with DPM++ 2M the scene layout carries over from a 12-step draft to
the full render at the same seed (benchmarked: ~2.5x faster drafts, small
details such as a head angle can still shift). The full render uses the same
sampler at the performance's normal step count.
"""
from modules.constants import MAX_SEED

DRAFT_STEPS = 12
DRAFT_SAMPLER = 'dpmpp_2m'
DRAFT_SCHEDULER = 'karras'
# These presets force their own fast sampler/steps in the worker, so a draft
# override would be ignored (and they're already fast).
FAST_PERFORMANCES = frozenset({'Extreme Speed', 'Lightning', 'Hyper-SD'})


def apply_draft(values: dict, mode: str | None, seed: int | None = None) -> bool:
    """Adjust named Generate inputs in place for a draft or a full render.

    ``values`` holds at least performance, overwrite_step, sampler_name,
    scheduler_name, image_seed and image_number. ``mode`` is 'draft',
    'full' or None. Returns True when draft settings were applied.
    """
    if mode is None or values.get('performance') in FAST_PERFORMANCES:
        return False
    values['sampler_name'] = DRAFT_SAMPLER
    values['scheduler_name'] = DRAFT_SCHEDULER
    if mode == 'draft':
        values['overwrite_step'] = DRAFT_STEPS
        return True
    if mode == 'full':
        values['overwrite_step'] = -1  # the performance's normal steps
        if seed is not None:
            values['image_seed'] = seed
        values['image_number'] = 1
        return False
    raise ValueError(f'unknown draft mode {mode!r}')


def seed_for_result(task, index) -> int:
    """Seed of the ``index``-th image of ``task``'s gallery. Raises
    ValueError with a user-facing message when it can't be known."""
    if not getattr(task, 'draft_mode', False):
        raise ValueError('Generate with Draft mode on first, then pick one of the drafts.')
    if index is None:
        raise ValueError('Click one of the draft images in the gallery first.')
    if getattr(task, 'prompt_queue', False) and len([l for l in task.prompt.splitlines() if l.strip()]) > 1:
        raise ValueError('Render full quality is not available for prompt-queue batches.')
    if getattr(task, 'should_enhance', False):
        raise ValueError('Render full quality is not available when Enhance is on.')
    count = task.image_number
    if not 0 <= index < count:
        raise ValueError('Pick one of the draft images (not the grid).')
    if task.disable_seed_increment:
        return task.seed % (MAX_SEED + 1)
    return (task.seed + index) % (MAX_SEED + 1)
