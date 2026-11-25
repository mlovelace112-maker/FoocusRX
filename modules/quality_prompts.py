"""Auto quality prompts: model-aware positive/negative additions.

When enabled, generation adds a short quality block suited to the selected
model's family (detected the same way as the per-model best settings in
modules.model_profiles). The user's prompt box is never changed; the added
text appears in the saved full prompt. Tags already in the prompt are skipped.
"""
import os

# Positive / negative additions per family. Pony and anime checkpoints are
# trained on these tag vocabularies (score_* for Pony; "masterpiece, best
# quality" style for anime); photo/general SDXL respond to descriptive quality
# terms; distilled few-step models run at CFG ~1, where a negative prompt has
# little effect, so they only get a short positive. FLUX.1 uses no negative
# prompt and prefers natural-language sentences.
FAMILY_PROMPTS = {
    'general': (
        'highly detailed, sharp focus, intricate details, professional lighting, high quality',
        'lowres, blurry, out of focus, jpeg artifacts, watermark, signature, text, worst quality, low quality, '
        'deformed, disfigured, bad anatomy, bad proportions, extra limbs, extra fingers, mutated hands, '
        'poorly drawn hands, poorly drawn face, cropped',
    ),
    'photo': (
        'photorealistic, highly detailed, sharp focus, realistic skin texture, natural lighting, '
        'professional photography',
        'lowres, blurry, out of focus, jpeg artifacts, watermark, signature, text, worst quality, low quality, '
        'cartoon, illustration, painting, drawing, anime, 3d render, cgi, plastic skin, deformed, disfigured, '
        'bad anatomy, extra limbs, extra fingers, mutated hands, poorly drawn face',
    ),
    'pony': (
        'score_9, score_8_up, score_7_up',
        'score_6, score_5, score_4, worst quality, low quality, lowres, blurry, jpeg artifacts, watermark, '
        'signature, text, bad anatomy, bad hands, extra fingers, deformed',
    ),
    'anime': (
        'masterpiece, best quality, amazing quality, very aesthetic, absurdres, highly detailed',
        'worst quality, low quality, lowres, blurry, jpeg artifacts, watermark, signature, text, bad anatomy, '
        'bad hands, extra fingers, missing fingers, deformed, sketch',
    ),
    'playground': (
        'high quality, highly detailed, sharp focus',
        'lowres, blurry, jpeg artifacts, watermark, signature, text, worst quality, low quality, deformed',
    ),
    'distilled': (
        'highly detailed, sharp focus, high quality',
        '',
    ),
    'flux': (
        'Highly detailed, sharp focus, realistic lighting and textures, high-resolution image.',
        '',
    ),
}
PROFILE_TO_FAMILY = {
    'pony_v6': 'pony',
    'anime': 'anime',
    'realistic': 'photo',
    'playground_v2.5': 'playground',
    'distilled': 'distilled',
}


def family_for_model(model_name: str, model_path: str | None = None, presets: dict | None = None) -> str:
    """Quality family for a base model file name."""
    if str(model_name).lower().endswith('.gguf'):
        return 'flux'
    from modules import model_profiles
    if presets is None:
        presets = model_profiles.load_presets(os.path.abspath('./presets'))
    profile = model_profiles.choose_profile(model_name, model_path, presets)
    return PROFILE_TO_FAMILY.get(profile.name, 'general')


def _missing_terms(addition: str, existing: str, separator: str = ',') -> str:
    """The comma-separated terms of ``addition`` not already in ``existing``."""
    have = existing.lower()
    terms = [t.strip() for t in addition.split(separator) if t.strip()]
    return ', '.join(t for t in terms if t.lower() not in have)


def additions(family: str, prompt: str, negative_prompt: str = '') -> tuple[str, str]:
    """(positive, negative) text to add for ``family``, without terms the
    user already wrote. Either may be ''."""
    positive, negative = FAMILY_PROMPTS.get(family, FAMILY_PROMPTS['general'])
    if family == 'flux':  # a sentence, not tags: add it unless it's already there
        return ('' if positive.lower().rstrip('.') in prompt.lower() else positive), ''
    return _missing_terms(positive, prompt), _missing_terms(negative, negative_prompt)
