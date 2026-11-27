# Prompt Studio

Open **✨ Prompt Studio** under the prompt box. Give it an idea in plain
words, a reference image, or both, and press **Build prompt**. It writes:

- a prompt in the style your base model likes (SDXL phrases, anime tags for
  Pony/Illustrious checkpoints, or sentences for FLUX; pick one under
  **Prompt style** to override),
- a negative prompt (not for FLUX),
- up to three Fooocus styles and an aspect ratio.

Everything stays editable. **Send to Generate** copies it into the main
prompt, negative prompt, Styles and Aspect Ratios (Fooocus V2 stays on if it
was on). **Send and Generate** also presses Generate.

## Reference images

The reference is always read and described into the prompt. **Use the
reference for** also decides how it goes into generation when you send:

| Choice | Sent as | Keeps |
| --- | --- | --- |
| Describe only | nothing | only what the prompt says |
| Overall look | Image Prompt | subject and look |
| Pose and outlines | PyraCanny | outlines, pose and layout |
| Depth and structure | CPDS | depth and arrangement |
| Face | FaceSwap | the face |

Sending fills the first Image Prompt slot, switches **Input Image** on and
selects the Image Prompt tab, with that type's default stop and weight.
Image Prompt is SDXL-only, so with other model families only the prompt is
sent. **What the model saw in the reference** shows the description used.

## Offline engine: Ollama (default)

1. Install the Ollama app from ollama.com and open it.
2. Download the default model once (about 3.3 GB). It reads images and
   writes prompts:

   ```bash
   ollama pull gemma3:4b
   ```

3. In **Studio settings**, press **Check Ollama**.

Other models work too, for example `qwen2.5vl:3b` (vision) or `qwen2.5:3b`
(text). Set them in Studio settings, or before launching:

| Variable | Default |
| --- | --- |
| `FOOOCUS_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `FOOOCUS_STUDIO_TEXT_MODEL` | `gemma3:4b` |
| `FOOOCUS_STUDIO_VISION_MODEL` | `gemma3:4b` |
| `FOOOCUS_STUDIO_KEEP_ALIVE` | `2m` (how long Ollama keeps the model loaded) |

When a generation starts, FoocusRX asks Ollama to unload the studio's models
so they don't share memory with the image model.

If Ollama isn't running, the studio falls back to the built-in prompt
rewriter (`scripts/download_prompt_model.py`) and Fooocus' Describe for
images, without a negative prompt or style suggestions.

## Optional: Claude

Pick **Claude (online)** as the engine. Your idea and reference image are sent
to Anthropic. Claude reads the image and writes the prompt in one request.

1. Install the SDK once, from the FoocusRX folder:

   ```bash
   venv/bin/pip install anthropic
   ```

2. Give it your API key, either in the environment before launching
   (`export ANTHROPIC_API_KEY=...`), or in **Studio settings > API key**
   for this session only. The key is never written to disk.

The default model is `claude-opus-5-5` (`FOOOCUS_STUDIO_CLAUDE_MODEL` or
Studio settings to change it). If Claude can't be reached, the studio uses
Ollama instead.

## Where things are saved

Next to your images in the outputs folder:

- `.prompt_studio.json`: engine, Ollama address and model names (no keys),
- `.prompt_studio_history.json`: the last 50 builds (**Recent builds**),
- **Save to Prompts** adds the prompt to the Prompts tab library, tagged
  `studio`.
