"""Automatic negative prompt: negatives chosen from what the prompt asks for.

Rule-based and deterministic (no model): each rule looks for cues in the
positive prompt and adds negatives that pull the image away from the usual
failure modes for that kind of picture, e.g. a photo gets "illustration,
cartoon, 3d render", a single subject gets "duplicate, crowd", "at night"
gets "daylight". A negative is never added when the prompt asks for it
("bokeh" blocks "blurry", "black and white" blocks "monochrome", "a sign"
blocks "text"), terms already present are skipped, and the result is capped.

Negative prompts only work through classifier-free guidance, so callers skip
this at CFG <= 1 (Lightning, Hyper-SD, Extreme Speed, FLUX.1 schnell).
"""
import re

MAX_TERMS = 28

# Always considered (cheap, broadly useful), then filtered like the rest.
BASE = 'worst quality, low quality, lowres, blurry, jpeg artifacts, watermark, text, signature'
EXTRA = 'username, logo, cropped, out of frame'  # used if there is room left


def _cue(*words):
    """Regex matching any of ``words`` as whole words/phrases (case-insensitive)."""
    return re.compile(r'\b(?:' + '|'.join(words) + r')\b', re.I)


# (rule name, cue in the prompt, negatives to add)
RULES = [
    ('photo', _cue(r'photo(?:graph)?s?', r'photo-?realistic', r'realistic', r'raw photo', r'dslr', r'\d+mm',
                   r'film still', r'cinematic still', r'shot on', r'canon', r'nikon', r'sony a\d+', r'kodak',
                   r'portrait photo(?:graphy)?', r'documentary'),
     'illustration, painting, drawing, cartoon, anime, 3d render, cgi, sketch, plastic skin, airbrushed'),
    ('illustration', _cue(r'anime', r'manga', r'illustration', r'cartoon', r'comic', r'drawing', r'sketch',
                          r'painting', r'watercolou?r', r'oil painting', r'digital art', r'concept art',
                          r'vector art', r'cel[- ]shad(?:ed|ing)', r'pixel art', r'line art'),
     'photo, photograph, photorealistic, realistic, 3d render'),
    ('render', _cue(r'3d render', r'octane', r'blender', r'unreal engine', r'cgi', r'isometric', r'low poly'),
     'photo, sketch, flat colors'),
    ('person', _cue(r'person', r'man', r'woman', r'men', r'women', r'girl', r'boy', r'child', r'kid', r'people',
                    r'portrait', r'face', r'character', r'lady', r'guy', r'old (?:man|woman)', r'fisherman',
                    r'warrior', r'knight', r'soldier', r'model', r'dancer', r'chef', r'astronaut'),
     'bad anatomy, deformed face, asymmetrical eyes, extra limbs, extra fingers, fused fingers, '
     'mutated hands, bad hands, long neck'),
    ('animal', _cue(r'cat', r'dog', r'fox', r'wolf', r'bird', r'horse', r'lion', r'tiger', r'bear', r'deer',
                    r'rabbit', r'owl', r'eagle', r'animal', r'puppy', r'kitten', r'cow', r'elephant'),
     'extra legs, missing legs, deformed animal, fused limbs'),
    ('single', _cue(r'(?:a|an|one|single|solo|lone|alone)\s+(?!few|couple|group|pair|bunch|lot)[a-z]+',
                    r'portrait of an?', r'solo'),
     'duplicate, multiple subjects, clones, crowd'),
    ('scene', _cue(r'landscape', r'cityscape', r'city', r'street', r'building', r'architecture', r'interior',
                   r'room', r'skyline', r'castle', r'bridge', r'house', r'lighthouse'),
     'distorted perspective, tilted horizon, warped lines'),
    ('night', _cue(r'night', r'nighttime', r'midnight', r'moonlit', r'starry'),
     'daylight, bright sun, daytime'),
    ('day', _cue(r'daylight', r'daytime', r'sunny', r'noon', r'midday', r'bright sun'),
     'night, dark, underexposed'),
    ('closeup', _cue(r'close-?up', r'headshot', r'macro', r'portrait'),
     'full body, wide shot, distant'),
    ('fullbody', _cue(r'full[- ]body', r'full[- ]length', r'wide shot', r'standing'),
     'cropped, cut off, close-up'),
    ('colour', _cue(r'vibrant', r'colou?rful', r'saturated', r'vivid'),
     'monochrome, grayscale, dull colors, washed out'),
    ('mono', _cue(r'black and white', r'black-and-white', r'b&w', r'monochrome', r'grayscale', r'greyscale',
                  r'noir'),
     'color, colorful, saturated'),
]

# Negative term -> cue in the prompt that blocks it (the prompt wants it).
CONFLICTS = [
    (_cue(r'blurry', r'blur', r'out of focus', r'soft focus'),
     _cue(r'bokeh', r'blurr?(?:y|ed)? background', r'depth of field', r'dof', r'motion blur', r'soft focus',
          r'dreamy', r'out of focus', r'shallow focus', r'long exposure', r'tilt[- ]shift', r'haze', r'hazy', r'fog')),
    (_cue(r'text', r'watermark', r'signature', r'logo', r'username', r'letters'),
     _cue(r'text', r'sign', r'signage', r'logo', r'typography', r'letters?', r'lettering', r'caption', r'words?',
          r'title', r'poster', r'label', r'banner', r'neon sign', r'calligraphy', r'book cover', r'newspaper',
          r'graffiti', r'quote', r'says', r'saying', r'written', r'reads')),
    (_cue(r'monochrome', r'grayscale', r'black and white'),
     _cue(r'black and white', r'black-and-white', r'b&w', r'monochrome', r'grayscale', r'greyscale', r'noir',
          r'sepia')),
    (_cue(r'color', r'colorful', r'saturated'),
     _cue(r'colou?r(?:ful)?', r'vibrant', r'saturated', r'vivid', r'rainbow')),
    (_cue(r'dark', r'night', r'underexposed'),
     _cue(r'dark', r'night', r'moody', r'low[- ]key', r'noir', r'shadow', r'gothic', r'horror', r'moonlit')),
    (_cue(r'daylight', r'bright sun', r'daytime', r'overexposed'),
     _cue(r'daylight', r'daytime', r'sunny', r'bright', r'high[- ]key', r'sunlit', r'noon')),
    (_cue(r'cropped', r'out of frame', r'cut off', r'close-up'),
     _cue(r'close-?up', r'crop(?:ped)?', r'macro', r'headshot', r'extreme close')),
    (_cue(r'full body', r'wide shot', r'distant'),
     _cue(r'full[- ]body', r'wide shot', r'wide angle', r'distant', r'aerial', r'landscape')),
    (_cue(r'duplicate', r'multiple subjects', r'clones', r'crowd'),
     _cue(r'crowd', r'group', r'people', r'twins', r'multiple', r'several', r'many', r'pair', r'couple', r'family',
          r'team', r'army', r'flock', r'herd', r'pack', r'two', r'three', r'four', r'five', r'\d+ (?:\w+ )?\w+s')),
    (_cue(r'jpeg artifacts', r'lowres', r'low quality', r'pixelated'),
     _cue(r'pixel art', r'8-?bit', r'16-?bit', r'retro game', r'glitch', r'lo-?fi')),
    (_cue(r'sketch', r'drawing', r'painting', r'illustration', r'cartoon', r'anime', r'3d render', r'cgi',
          r'photo', r'photograph', r'photorealistic', r'realistic', r'flat colors', r'airbrushed', r'plastic skin'),
     None),  # media terms: blocked by the generic "the prompt contains this term" check below
    (_cue(r'grain', r'noise', r'grainy', r'film grain'),
     _cue(r'grain', r'grainy', r'film grain', r'noise', r'analog', r'film photo')),
    (_cue(r'frame', r'border'), _cue(r'frame', r'framed', r'border', r'polaroid', r'window')),
    (_cue(r'distorted perspective', r'warped lines', r'tilted horizon'),
     _cue(r'fisheye', r'dutch angle', r'tilted', r'surreal', r'distorted', r'warped', r'escher')),
    (_cue(r'extra legs', r'extra limbs', r'extra fingers', r'long neck', r'deformed'),
     _cue(r'spider', r'octopus', r'centipede', r'many[- ]armed', r'multi-?limbed', r'giraffe', r'monster',
          r'creature', r'deformed', r'grotesque', r'mutant', r'body horror')),
]


def _terms(text: str) -> list:
    return [t.strip() for t in (text or '').split(',') if t.strip()]


def conflicts(term: str, prompt: str) -> bool:
    """True when the prompt asks for what ``term`` would suppress."""
    low_prompt = prompt.lower()
    if re.search(r'\b' + re.escape(term.lower()) + r'\b', low_prompt):
        return True  # the prompt literally asks for it
    for term_cue, prompt_cue in CONFLICTS:
        if prompt_cue is not None and term_cue.fullmatch(term.strip()) and prompt_cue.search(prompt):
            return True
    return False


def drop_conflicts(negative: str, prompt: str) -> str:
    """``negative`` (comma-separated) without the terms the prompt contradicts."""
    return ', '.join(t for t in _terms(negative) if not conflicts(t, prompt))


def matched_rules(prompt: str) -> list:
    return [name for name, cue, _ in RULES if cue.search(prompt or '')]


def build(prompt: str, existing_negative: str = '', max_terms: int = MAX_TERMS) -> str:
    """Negatives to add for ``prompt`` (comma-separated, possibly '').

    ``existing_negative`` is everything already in the negative prompt (the
    user's text, style negatives, quality additions); those terms are skipped.
    """
    prompt = prompt or ''
    # Core quality terms first; then the matched rules round-robin (each
    # rule's most important term before anyone's second), so the cap keeps
    # something from every rule that applies.
    groups = [_terms(negatives) for _, cue, negatives in RULES if cue.search(prompt)]
    candidates = _terms(BASE)
    for i in range(max((len(g) for g in groups), default=0)):
        candidates += [g[i] for g in groups if i < len(g)]
    candidates += _terms(EXTRA)
    have = (existing_negative or '').lower()
    out, seen = [], set()
    for term in candidates:
        key = term.lower()
        if key in seen or re.search(r'\b' + re.escape(key) + r'\b', have) or conflicts(term, prompt):
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max_terms:
            break
    return ', '.join(out)


def applies(cfg_scale, family: str | None = None) -> bool:
    """Negatives need classifier-free guidance: off at CFG <= 1 and for FLUX."""
    try:
        return family != 'flux' and float(cfg_scale) > 1.0 + 1e-6
    except (TypeError, ValueError):
        return False
