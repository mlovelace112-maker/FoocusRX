"""Gradio panel for Prompt Studio (see modules/prompt_studio.py).

build_panel() lays the components out wherever it is called (webui puts it in
an accordion under the prompt box); wire() connects them to the main prompt,
styles, aspect ratio and Image Prompt controls once those exist.
"""
from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from modules import gradio_compat as gc
from modules import prompt_library
from modules import prompt_studio as ps
from modules import ui_family

LIBRARY_TAG = 'studio'


def build_panel(settings: ps.StudioSettings, aspect_labels, style_names) -> SimpleNamespace:
    """Create the Studio components inside the current Gradio layout context."""
    aspect_choices = [(ps.aspect_display(label), label) for label in aspect_labels]
    p = SimpleNamespace()
    gr.Markdown('Describe a picture in plain words, add a reference image, or both. The studio writes the prompt, '
                'negative prompt and suggested styles; **Send to Generate** fills them in (and the reference '
                'as an Image Prompt when you pick how to use it).', elem_classes='rx-notice')
    with gr.Row():
        with gr.Column(scale=4, min_width=280):
            p.idea = gr.Textbox(label='Your idea', lines=3, elem_id='studio_idea',
                                placeholder='e.g. a lighthouse keeper’s cat watching a storm roll in, moody')
            with gr.Row():
                p.image = gr.Image(label='Reference image (optional)', **gc.image_source('upload'),
                                   type='numpy', height=220, elem_id='studio_image')
                with gr.Column(min_width=160):
                    p.reference_mode = gr.Radio(label='Use the reference for', choices=list(ps.REFERENCE_MODES),
                                                value=ps.DEFAULT_REFERENCE_MODE, elem_id='studio_reference_mode')
            with gr.Row():
                p.engine = gr.Radio(label='Engine', choices=ps.ENGINES, value=settings.engine,
                                    elem_id='studio_engine')
                p.target = gr.Dropdown(label='Prompt style', choices=list(ps.TARGETS), value=ps.TARGET_AUTO,
                                       elem_id='studio_target')
            p.build = gr.Button(value='✨ Build prompt', variant='primary', elem_id='studio_build')
        with gr.Column(scale=5, min_width=280):
            p.positive = gr.Textbox(label='Prompt', lines=5, elem_id='studio_positive', interactive=True)
            p.negative = gr.Textbox(label='Negative prompt', lines=2, elem_id='studio_negative', interactive=True)
            with gr.Row():
                p.styles = gr.Dropdown(label='Styles', choices=list(style_names), value=[], multiselect=True,
                                       scale=3, elem_id='studio_styles')
                p.aspect = gr.Dropdown(label='Aspect ratio', choices=aspect_choices, value=None, scale=2,
                                       elem_id='studio_aspect')
            p.status = gr.Markdown(value='', elem_classes='rx-status', elem_id='studio_status')
            with gr.Accordion(label='What the model saw in the reference', open=False):
                p.description = gr.Textbox(show_label=False, lines=4, interactive=False, container=False,
                                           elem_id='studio_description')
            with gr.Row(elem_classes='rx-toolbar'):
                p.send = gr.Button(value='Send to Generate', size='sm', min_width=0, elem_id='studio_send')
                p.send_generate = gr.Button(value='Send and Generate', size='sm', variant='primary', min_width=0,
                                            elem_id='studio_send_generate')
            with gr.Row():
                p.save_name = gr.Textbox(label='Name (optional)', scale=3, elem_id='studio_save_name')
                p.save_tags = gr.Textbox(label='Tags (comma-separated)', scale=3, elem_id='studio_save_tags')
                p.save = gr.Button(value='\U0001f4be Save to Prompts', scale=2, min_width=0, elem_id='studio_save')
    with gr.Accordion(label='Recent builds', open=False) as p.history_accordion:
        with gr.Row():
            p.history = gr.Dropdown(label='Recent builds', choices=[], value=None, scale=5, show_label=False,
                                    container=False, elem_id='studio_history')
            p.history_load = gr.Button(value='Load', size='sm', scale=1, min_width=0, elem_id='studio_history_load')
    with gr.Accordion(label='Studio settings', open=False) as p.settings_accordion:
        gr.Markdown('**Offline (Ollama).** Install the Ollama app, then run `ollama pull '
                    f'{ps.DEFAULT_TEXT_MODEL}` once. It reads images and writes prompts.', elem_classes='rx-notice')
        with gr.Row():
            p.ollama_url = gr.Textbox(label='Ollama address', value=settings.ollama_url, scale=3,
                                      elem_id='studio_ollama_url')
            p.text_model = gr.Textbox(label='Text model', value=settings.text_model, scale=2,
                                      elem_id='studio_text_model')
            p.vision_model = gr.Textbox(label='Vision model', value=settings.vision_model, scale=2,
                                        elem_id='studio_vision_model')
        with gr.Row():
            p.check = gr.Button(value='Check Ollama', size='sm', min_width=0, elem_id='studio_check')
            p.ollama_status = gr.Markdown(value='', elem_id='studio_ollama_status')
        gr.Markdown('**Claude (optional, online).** Uses your own key: set `ANTHROPIC_API_KEY` before launching '
                    'FoocusRX, or paste one below for this session (it is never saved). Needs '
                    '`venv/bin/pip install anthropic` once. Your idea and reference image are sent to Anthropic.',
                    elem_classes='rx-notice')
        with gr.Row():
            p.claude_model = gr.Textbox(label='Claude model', value=settings.claude_model, scale=2,
                                        elem_id='studio_claude_model')
            p.claude_key = gr.Textbox(label='API key (this session only)', type='password', value='', scale=3,
                                      placeholder='Leave empty to use ANTHROPIC_API_KEY', elem_id='studio_claude_key')
    return p


def _settings_from(engine, url, text_model, vision_model, claude_model, claude_key) -> ps.StudioSettings:
    return ps.StudioSettings(engine=engine, ollama_url=url, text_model=text_model, vision_model=vision_model,
                             claude_model=claude_model, claude_api_key=claude_key).clean()


def _fallback_rewrite(idea, style):
    from modules import prompt_rewriter
    if not prompt_rewriter.available():
        return ''
    return prompt_rewriter.rewrite(idea, style)


def _fallback_describe(image):
    # Fooocus' own Describe (Photograph) interrogator; downloads its model on first use.
    from extras.interrogate import default_interrogator
    return default_interrogator(image)


def wire(p: SimpleNamespace, *, root_fn, legal_style_names, aspect_labels, base_model, prompt, negative_prompt,
         style_selections, aspect_ratios_selection, input_image_checkbox, image_input_tabs, current_tab, ip_image,
         ip_type, ip_stop, ip_weight, ip_ad_col, ip_defaults, ip_tab_id='ip_tab', generate_button_id='generate_button'):
    """Connect the panel to the main UI. root_fn() returns the outputs folder."""

    def history_choices(selected=None):
        entries = ps.load_history(root_fn())
        ids = [e['id'] for e in entries]
        return gr.update(choices=[(ps.history_label(e), e['id']) for e in entries],
                         value=selected if selected in ids else None)

    settings_inputs = [p.engine, p.ollama_url, p.text_model, p.vision_model, p.claude_model, p.claude_key]

    def build_clicked(idea, image, reference_mode, target, model_name, *settings_values):
        settings = _settings_from(*settings_values)
        try:
            ps.save_settings(root_fn(), settings)
        except OSError:
            pass
        style = ps.resolve_target(target, model_name)
        try:
            result = ps.build_prompt(idea, image, reference_mode, settings, style=style,
                                     legal_style_names=legal_style_names, aspect_labels=aspect_labels,
                                     fallback_rewrite=_fallback_rewrite, fallback_describe=_fallback_describe)
        except ps.StudioError as exc:
            raise gr.Error(str(exc))
        entry = ps.add_history(root_fn(), {
            'idea': (idea or '').strip(), 'had_image': image is not None, 'reference_mode': reference_mode,
            'engine': result.engine_used, 'positive': result.positive, 'negative': result.negative,
            'styles': result.styles, 'aspect': result.aspect, 'ip_type': result.ip_type,
            'description': result.description})
        status = f'Built with {result.engine_used}.' + (f' {result.notes}' if result.notes else '')
        return [result.positive, result.negative, gr.update(value=result.styles), gr.update(value=result.aspect),
                status, result.description, history_choices(entry['id'])]

    build_outputs = [p.positive, p.negative, p.styles, p.aspect, p.status, p.description, p.history]
    p.build.click(build_clicked,
                  inputs=[p.idea, p.image, p.reference_mode, p.target, base_model] + settings_inputs,
                  outputs=build_outputs, show_progress=True)

    def history_load_clicked(entry_id):
        entry = ps.get_history(root_fn(), entry_id) if entry_id else None
        if entry is None:
            return [gr.update()] * 7 + ['Pick a build first.']
        mode = entry.get('reference_mode') if entry.get('reference_mode') in ps.REFERENCE_MODES else gr.update()
        styles = ps.pick_styles(entry.get('styles'), legal_style_names)
        aspect = entry.get('aspect') if entry.get('aspect') in aspect_labels else None
        note = ('Loaded. Add the reference image again to send it as an Image Prompt.'
                if entry.get('had_image') else 'Loaded.')
        return [entry.get('idea', ''), entry['positive'], entry.get('negative', ''), gr.update(value=styles),
                gr.update(value=aspect), mode, entry.get('description', ''), note]

    p.history_load.click(history_load_clicked, inputs=p.history,
                         outputs=[p.idea, p.positive, p.negative, p.styles, p.aspect, p.reference_mode,
                                  p.description, p.status], queue=False, show_progress=False)
    p.history_accordion.expand(lambda: history_choices(), outputs=p.history, queue=False, show_progress=False)

    def check_clicked(*settings_values):
        settings = _settings_from(*settings_values)
        try:
            ps.save_settings(root_fn(), settings)
        except OSError:
            pass
        return ps.ollama_status(settings)

    p.check.click(check_clicked, inputs=settings_inputs, outputs=p.ollama_status, show_progress=True)

    def save_clicked(positive, negative, name, tags):
        tags = ', '.join([LIBRARY_TAG] + [t for t in (tags or '').split(',') if t.strip()])
        try:
            entry = prompt_library.PromptLibrary(root_fn()).save(positive, negative, name, tags)
        except ValueError:
            return 'Build or type a prompt first.', gr.update(), gr.update()
        return f"Saved “{entry['name']}” to the Prompts tab.", '', ''

    p.save.click(save_clicked, inputs=[p.positive, p.negative, p.save_name, p.save_tags],
                 outputs=[p.status, p.save_name, p.save_tags], queue=False, show_progress=False)

    def send_clicked(positive, negative, styles, aspect, image, reference_mode, current_styles, model_name):
        if not (positive or '').strip():
            raise gr.Error('Build or type a prompt first.')
        new_styles = list(styles or [])
        # Keep the Fooocus V2 prompt expansion if it was on.
        if 'Fooocus V2' in (current_styles or []) and 'Fooocus V2' not in new_styles:
            new_styles.insert(0, 'Fooocus V2')
        kind = ps.REFERENCE_MODES.get(reference_mode) if image is not None else None
        out = [positive.strip(), (negative or '').strip(), gr.update(value=new_styles),
               gr.update(value=aspect) if aspect in aspect_labels else gr.update()]
        if kind is not None and ui_family.family_of(model_name) != ui_family.SDXL:
            return out + [gr.update()] * 8 + ['Sent the prompt. Image Prompt only works with SDXL models, so the '
                                              'reference was not sent.']
        if kind is None:
            return out + [gr.update()] * 8 + ['Sent the prompt to Generate.']
        stop, weight = ip_defaults[kind]
        return out + [True, gr.Tabs(selected=ip_tab_id), 'ip', image, kind, stop, weight, gr.update(visible=True),
                      f'Sent the prompt, and the reference as an Image Prompt ({kind}).']

    send_inputs = [p.positive, p.negative, p.styles, p.aspect, p.image, p.reference_mode, style_selections,
                   base_model]
    send_outputs = [prompt, negative_prompt, style_selections, aspect_ratios_selection, input_image_checkbox,
                    image_input_tabs, current_tab, ip_image, ip_type, ip_stop, ip_weight, ip_ad_col, p.status]
    p.send.click(send_clicked, inputs=send_inputs, outputs=send_outputs, queue=False, show_progress=False)
    p.send_generate.click(send_clicked, inputs=send_inputs, outputs=send_outputs, queue=False,
                          show_progress=False) \
        .success(None, queue=False, show_progress=False, **gc.js_kwarg(
            "() => { setTimeout(() => { const b = document.getElementById('%s');"
            " if (b && b.offsetParent !== null) b.click(); }, 400); return []; }" % generate_button_id))
