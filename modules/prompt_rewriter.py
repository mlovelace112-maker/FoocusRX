"""AI prompt rewriter: turn a short idea into a detailed prompt with a small
local language model (Qwen2.5-1.5B-Instruct, Apache-2.0).

The output style follows the selected base model: comma-separated phrases for
SDXL, booru-style tags for Pony/anime checkpoints, natural sentences for
FLUX.1. The model is loaded for each rewrite and freed afterwards (~3 GB), so
it never sits in memory next to the image model. Download it with
scripts/download_prompt_model.py.
"""
import gc
import os
import re
import threading

MODEL_DIR_NAME = 'Qwen2.5-1.5B-Instruct'
REQUIRED_FILES = ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')

STYLES = {
    'sdxl': {
        'target': 'Stable Diffusion XL',
        'format': ('Write it as ONE line of comma-separated descriptive phrases, 25 to 60 words, '
                   'most important subject first.'),
        'example': ('a cat on a windowsill',
                    'a ginger tabby cat curled up on a sunlit wooden windowsill, soft morning light through '
                    'sheer curtains, potted plants, shallow depth of field, cozy atmosphere, 50mm photo, '
                    'warm color palette'),
    },
    'tags': {
        'target': 'an anime-style Stable Diffusion XL model (Pony/Illustrious)',
        'format': ('Write it as ONE line of short comma-separated tags in booru style, 15 to 35 tags, '
                   'subject first. Do not add quality tags such as masterpiece, best quality or score_9.'),
        'example': ('a girl reading in a library',
                    '1girl, solo, reading, open book, long hair, glasses, sweater, sitting, library, '
                    'bookshelves, warm lighting, window, afternoon, peaceful, depth of field'),
    },
    'flux': {
        'target': 'FLUX.1',
        'format': ('Write 2 or 3 natural-language sentences, 40 to 80 words in total, like a vivid, '
                   'specific photo or artwork description.'),
        'example': ('a cat on a windowsill',
                    'A ginger tabby cat lies curled on a sunlit wooden windowsill, its fur glowing in soft '
                    'morning light filtering through sheer white curtains. Potted herbs line the sill, and the '
                    'background falls into a gentle blur, giving the scene a calm, cozy feel.'),
    },
}
SYSTEM = ("You are an expert prompt writer for the text-to-image model {target}. Rewrite the user's idea "
          "into a single image prompt. Keep the user's subject, intent and any details they gave; add concrete "
          "visual details: appearance, setting, lighting, composition, camera or medium, colours and mood. "
          "Do not add text, captions, watermarks or real people's names. {format} "
          "Reply with the prompt only: no quotes, labels or explanations.")

_lock = threading.Lock()


def model_dir(base: str | None = None) -> str:
    if base is None:
        import modules.config as config
        base = config.path_prompt_model
    return os.path.join(base, MODEL_DIR_NAME)


def available(base: str | None = None) -> bool:
    directory = model_dir(base)
    return all(os.path.exists(os.path.join(directory, f)) for f in REQUIRED_FILES)


def style_for_model(model_name: str) -> str:
    """'flux', 'tags' or 'sdxl' for the selected base model."""
    name = str(model_name or '').lower()
    if name.endswith('.gguf'):
        return 'flux'
    if re.search(r'pony|illustrious|noobai|animagine|anime', name):
        return 'tags'
    return 'sdxl'


def build_messages(idea: str, style: str) -> list:
    spec = STYLES.get(style, STYLES['sdxl'])
    example_in, example_out = spec['example']
    return [
        {'role': 'system', 'content': SYSTEM.format(target=spec['target'], format=spec['format'])},
        {'role': 'user', 'content': example_in},
        {'role': 'assistant', 'content': example_out},
        {'role': 'user', 'content': idea.strip()},
    ]


def clean_output(text: str, style: str, max_chars: int = 600) -> str:
    """Strip labels, quotes and stray formatting from the model's reply."""
    text = text.strip()
    text = re.sub(r'^(?:\*\*)?(?:prompt|image prompt|rewritten prompt)\s*:\s*(?:\*\*)?', '', text, flags=re.I)
    text = text.strip().strip('"“”\'`').strip()
    lines = [line.strip(' -*•') for line in text.splitlines() if line.strip()]
    if style == 'flux':
        text = ' '.join(lines)
    else:
        text = ', '.join(part.strip() for line in lines for part in line.split(',') if part.strip())
    text = re.sub(r'\s{2,}', ' ', text)
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut[:cut.rfind(',' if style != 'flux' else '.') + 1 or max_chars].rstrip(', ')
    return text


def rewrite(idea: str, style: str, seed: int | None = None, base: str | None = None,
            device: str | None = None, max_new_tokens: int = 160) -> str:
    """Rewrite ``idea``. Loads the model, generates, then frees it."""
    if not idea or not idea.strip():
        raise ValueError('Type a short idea in the prompt box first.')
    if not available(base):
        raise FileNotFoundError('The prompt rewriter model is not installed. Run: '
                                'venv/bin/python scripts/download_prompt_model.py')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device in ('mps', 'cuda') else torch.float32
    directory = model_dir(base)
    with _lock:
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        # 'eager' attention: with bf16 + the default SDPA path on MPS this model
        # degenerates into repeating the prompt (measured; fp32 or eager are
        # fine). Eager also avoids the mtlflashattn SDPA patch the app installs.
        model = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=dtype, local_files_only=True,
                                                     attn_implementation='eager')
        model = model.to(device).eval()
        try:
            text = tokenizer.apply_chat_template(build_messages(idea, style), tokenize=False,
                                                 add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors='pt').to(device)
            if seed is not None:
                torch.manual_seed(int(seed) % (2 ** 32))
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                                     top_p=0.9, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            del model
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
    result = clean_output(reply, style)
    if not result:
        raise RuntimeError('The rewriter returned an empty prompt; try again.')
    return result
