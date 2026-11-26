import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
from modules import gradio_compat as gc
import modules.style_sorter as style_sorter
import modules.meta_parser
import modules.prompt_rewriter
import modules.prompt_library
import modules.ui_family
import modules.lora_family
import modules.security
import modules.ui_theme
import modules.draft_mode
import modules.model_profiles
import modules.util
import modules.flux.app
import modules.history
import modules.model_manager
import threading
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
import modules.private_logger
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

history_index = modules.history.HistoryIndex()


def lora_choices(family):
    """'None' plus the LoRA files made for ``family`` (by their tensor names)."""
    return ['None'] + modules.lora_family.loras_for(
        family, modules.config.lora_filenames,
        lambda name: modules.util.get_file_from_folder_list(name, modules.config.paths_loras))


def refiner_choices():
    """Refiners must be SDXL/SD1.5 checkpoints; FLUX.1 can't refine."""
    return ['None'] + [m for m in modules.config.model_filenames if modules.ui_family.is_sdxl(m)]

# Delete / model download are off when the app is reachable without a login.
FILE_ACTIONS_LOCKED = modules.security.file_actions_locked(args_manager.args.listen, args_manager.args.share,
                                                           auth_enabled)
_security_warning = modules.security.startup_warning(args_manager.args.listen, args_manager.args.share,
                                                     auth_enabled)
if _security_warning:
    print(_security_warning)

HISTORY_PAGE = 48  # History tab: images per page ("Load more" adds another page)
HISTORY_STAR_CHOICES = ['Any rating', '\u2605 1+', '\u2605 2+', '\u2605 3+', '\u2605 4+', '\u2605 5']
HISTORY_RATING_CHOICES = ['\u2606', '\u2605', '\u2605\u2605', '\u2605\u2605\u2605', '\u2605\u2605\u2605\u2605', '\u2605\u2605\u2605\u2605\u2605']


def history_ratings():
    return modules.history.RatingsStore(modules.config.path_outputs)


# Positions of the inputs draft mode adjusts, within get_task's args after
# currentTask is removed. Filled in once ctrls is built (see below).
DRAFT_ARG_INDEX = {}


def get_task(*args):
    args = list(args)
    request = args.pop()        # draft_request: {'mode': 'full', 'seed': n} or None
    draft_enabled = args.pop()  # Draft mode checkbox
    args.pop(0)

    if request:
        mode, seed = request['mode'], request.get('seed')
    else:
        mode, seed = ('draft' if draft_enabled else None), None
    named = {name: args[i] for name, i in DRAFT_ARG_INDEX.items()}
    is_draft = modules.draft_mode.apply_draft(named, mode, seed)
    for name, i in DRAFT_ARG_INDEX.items():
        args[i] = named[name]

    task = worker.AsyncTask(args=args)
    task.draft_mode = is_draft
    return task

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    modules.prompt_rewriter.release()  # don't keep ~3 GB next to the image model
    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]


reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title, theme=modules.ui_theme.build_theme()).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    gr.HTML(modules.ui_theme.header_html(title.replace('Fooocus', 'FoocusRX', 1)), elem_id='rx_header')
    with gr.Row(elem_id='rx_main'):
        with gr.Column(scale=2, elem_id='rx_left'):
            with gr.Row():
                progress_window = gr.Image(label='Preview', show_label=True, visible=False, height=640,
                                            elem_classes=['main_view'])
                progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                              height=640, visible=False, elem_classes=['main_view', 'image_gallery'])
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=640,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery')
            with gr.Row():
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        autofocus=True, lines=3)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                    with gr.Row(elem_classes='rx-toolbar'):
                        rewrite_button = gr.Button(value='✨ Rewrite prompt', size='sm', variant='secondary',
                                                   elem_id='rewrite_prompt_button', min_width=0)
                        rewrite_undo = gr.Button(value='↩ Undo rewrite', size='sm', variant='secondary',
                                                 interactive=False, min_width=0)
                        view_larger_button = gr.Button(value='\U0001f50d View larger', size='sm', min_width=0,
                                                       elem_id='view_larger_button')
                        save_prompt_button = gr.Button(value='\U0001f4be Save prompt', size='sm', min_width=0,
                                                       elem_id='save_prompt_button')
                        render_full_button = gr.Button(value='⬆ Render full quality', size='sm',
                                                       variant='secondary', min_width=0,
                                                       elem_id='render_full_button')
                    prompt_before_rewrite = gr.State(None)

                with gr.Column(scale=3, min_width=0):
                    # Gradio 4 removed the (always-ignored) `label` kwarg on
                    # Button — the visible text is `value`. Keeping both used
                    # to be a no-op silently absorbed via **kwargs; Gradio 4's
                    # @component_meta wrapper now raises TypeError.
                    generate_button = gr.Button(value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True, variant='primary')
                    reset_button = gr.Button(value="Reconnect", elem_classes='type_row', elem_id='reset_button', visible=False)
                    load_parameter_button = gr.Button(value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                    stop_button = gr.Button(value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'stop'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    def skip_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'skip'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, **gc.js_kwarg('cancelGenerateForever'))
                    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
            initial_family = modules.ui_family.family_of(modules.config.default_base_model_name)
            with gr.Row(elem_id='rx_model_bar'):
                model_family = gr.Dropdown(label='Model family', choices=modules.ui_family.FAMILIES,
                                           value=initial_family, scale=1, min_width=140,
                                           elem_id='model_family')
                base_model = gr.Dropdown(label='Base model',
                                         choices=modules.ui_family.models_for(initial_family, modules.config.model_filenames),
                                         value=modules.config.default_base_model_name, show_label=True, scale=3,
                                         elem_id='base_model')
            family_memory = gr.State(modules.ui_family.initial_memory(modules.config.default_base_model_name,
                                                                      modules.config.model_filenames))
            flux_notice = gr.Markdown(value=modules.ui_family.NOTICES.get(initial_family, modules.ui_family.FLUX_NOTICE),
                                      elem_classes='rx-notice', visible=initial_family != modules.ui_family.SDXL)
            with gr.Accordion(label='Negative prompt', open=False, elem_id='negative_prompt_accordion',
                              visible=modules.ui_family.uses_negative_prompt(initial_family)) as negative_prompt_accordion:
                negative_prompt = gr.Textbox(show_label=False, placeholder='What you do not want to see',
                                             lines=2, elem_id='negative_prompt', container=False,
                                             value=modules.config.default_prompt_negative)
            with gr.Accordion(label='Compare A/B', open=False, elem_id='compare_accordion'):
                gr.Markdown('Renders the current prompt and settings once with the base model above (A) and '
                            'once with model B, using the **same seed**, side by side.', elem_classes='rx-notice')
                with gr.Row():
                    compare_model_b = gr.Dropdown(label='Model B', choices=modules.config.model_filenames,
                                                  value=next((m for m in modules.config.model_filenames
                                                              if m != modules.config.default_base_model_name),
                                                             modules.config.default_base_model_name),
                                                  scale=3, elem_id='compare_model_b')
                    compare_button = gr.Button(value='Compare', variant='primary', scale=1, min_width=0,
                                               elem_id='compare_button')
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = gr.Image(label='Image', **gc.image_source('upload'), type='numpy', show_label=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')
                    with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image = gr.Image(label='Image', **gc.image_source('upload'), type='numpy', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='Advanced', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = gc.sketch_image(label='Image', color="#FFFFFF", height=500, elem_id='inpaint_canvas', show_label=False)
                                inpaint_advanced_masking_checkbox = gr.Checkbox(label='Enable Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_mask_image = gc.sketch_image(label='Mask Upload', color="#FFFFFF", height=500, elem_id='inpaint_mask_canvas')
                                invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox)
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_dino_prompt_text = gr.Textbox(label='Detection prompt', value='', visible=False, info='Use singular whenever possible', placeholder='Describe what you want to detect.')
                                example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                    samples=modules.config.example_enhance_detection_prompts,
                                    label='Detection Prompt Quick List',
                                    components=[inpaint_mask_dino_prompt_text],
                                    visible=modules.config.default_inpaint_mask_model == 'sam')
                                example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                            inputs=example_inpaint_mask_dino_prompt_text,
                                                                            outputs=inpaint_mask_dino_prompt_text,
                                                                            show_progress=False, queue=False)

                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                    inpaint_mask_sam_max_detections = gr.Slider(label="Maximum number of detections", info="Set to 0 to detect all", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                generate_mask_button = gr.Button(value='Generate mask from image')

                                def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    sam_options = None
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        sam_options = SAMOptions(
                                            dino_prompt=dino_prompt_text,
                                            dino_box_threshold=box_threshold,
                                            dino_text_threshold=text_threshold,
                                            dino_erode_or_dilate=dino_erode_or_dilate,
                                            dino_debug=dino_debug,
                                            max_detections=sam_max_detections,
                                            model_type=sam_model
                                        )

                                    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

                                    return mask


                                inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                    [gr.update(visible=x == 'sam')] * 2 +
                                                                    [gr.update(visible=x == 'sam',
                                                                               samples=modules.config.example_enhance_detection_prompts)],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category,
                                                                   inpaint_mask_dino_prompt_text,
                                                                   inpaint_mask_advanced_options,
                                                                   example_inpaint_mask_dino_prompt_text],
                                                          queue=False, show_progress=False)

                    with gr.Tab(label='Describe', id='describe_tab') as describe_tab:
                        with gr.Row():
                            with gr.Column():
                                describe_input_image = gr.Image(label='Image', **gc.image_source('upload'), type='numpy', show_label=False)
                            with gr.Column():
                                describe_methods = gr.CheckboxGroup(
                                    label='Content Type',
                                    choices=flags.describe_types,
                                    value=modules.config.default_describe_content_type)
                                describe_apply_styles = gr.Checkbox(label='Apply Styles', value=modules.config.default_describe_apply_prompts_checkbox)
                                describe_btn = gr.Button(value='Describe this Image into Prompt')
                                describe_image_size = gr.Textbox(label='Image Size and Recommended Size', elem_id='describe_image_size', visible=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Documentation</a>')

                                def trigger_show_image_properties(image):
                                    value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
                                    return gr.update(value=value, visible=True)

                                describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image,
                                                            outputs=describe_image_size, show_progress=False, queue=False)

                    with gr.Tab(label='Enhance', id='enhance_tab') as enhance_tab:
                        with gr.Row():
                            with gr.Column():
                                enhance_input_image = gr.Image(label='Use with Enhance, skips image generation', **gc.image_source('upload'), type='numpy')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = gr.Image(label='For images created by Fooocus', **gc.image_source('upload'), type='pil')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                with gr.Tabs():
                    with gr.Tab(label='Upscale or Variation'):
                        with gr.Row():
                            with gr.Column():
                                enhance_uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                              value=modules.config.default_enhance_uov_method)
                                enhance_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                        info='Use before to enhance small details and after to enhance large areas.',
                                                                        choices=flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_enhance_uov_processing_order)
                                enhance_uov_prompt_type = gr.Radio(label='Prompt',
                                                                   info='Choose which prompt to use for Upscale or Variation.',
                                                                   choices=flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_enhance_uov_prompt_type,
                                                                   visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)

                                enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                    inputs=enhance_uov_processing_order,
                                                                    outputs=enhance_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                    enhance_ctrls = []
                    enhance_inpaint_mode_ctrls = []
                    enhance_inpaint_engine_ctrls = []
                    enhance_inpaint_update_ctrls = []
                    for index in range(modules.config.default_enhance_tabs):
                        with gr.Tab(label=f'#{index + 1}') as enhance_tab_item:
                            enhance_enabled = gr.Checkbox(label='Enable', value=False, elem_classes='min_check',
                                                          container=False)

                            enhance_mask_dino_prompt_text = gr.Textbox(label='Detection prompt',
                                                                       info='Use singular whenever possible',
                                                                       placeholder='Describe what you want to detect.',
                                                                       interactive=True,
                                                                       visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text = gr.Dataset(
                                samples=modules.config.example_enhance_detection_prompts,
                                label='Detection Prompt Quick List',
                                components=[enhance_mask_dino_prompt_text],
                                visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                        inputs=example_enhance_mask_dino_prompt_text,
                                                                        outputs=enhance_mask_dino_prompt_text,
                                                                        show_progress=False, queue=False)

                            enhance_prompt = gr.Textbox(label="Enhancement positive prompt",
                                                        placeholder="Uses original prompt instead if empty.",
                                                        elem_id='enhance_prompt')
                            enhance_negative_prompt = gr.Textbox(label="Enhancement negative prompt",
                                                                 placeholder="Uses original negative prompt instead if empty.",
                                                                 elem_id='enhance_negative_prompt')

                            with gr.Accordion("Detection", open=False):
                                enhance_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_enhance_inpaint_mask_model)
                                enhance_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                          choices=flags.inpaint_mask_cloth_category,
                                                                          value=modules.config.default_inpaint_mask_cloth_category,
                                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                          interactive=True)

                                with gr.Accordion("SAM Options",
                                                  visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                  open=False) as sam_options:
                                    enhance_mask_sam_model = gr.Dropdown(label='SAM model',
                                                                         choices=flags.inpaint_mask_sam_model,
                                                                         value=modules.config.default_inpaint_mask_sam_model,
                                                                         interactive=True)
                                    enhance_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0,
                                                                           maximum=1.0, value=0.3, step=0.05,
                                                                           interactive=True)
                                    enhance_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0,
                                                                            maximum=1.0, value=0.25, step=0.05,
                                                                            interactive=True)
                                    enhance_mask_sam_max_detections = gr.Slider(label="Maximum number of detections",
                                                                                info="Set to 0 to detect all",
                                                                                minimum=0, maximum=10,
                                                                                value=modules.config.default_sam_max_detections,
                                                                                step=1, interactive=True)

                            with gr.Accordion("Inpaint", visible=True, open=False):
                                enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                   value=modules.config.default_inpaint_method,
                                                                   label='Method', interactive=True)
                                enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                    label='Disable initial latent in inpaint', value=False)
                                enhance_inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                                     value=modules.config.default_inpaint_engine_version,
                                                                     choices=flags.inpaint_engine_versions,
                                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                enhance_inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                                     minimum=0.0, maximum=1.0, step=0.001,
                                                                     value=1.0,
                                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                                          'Only used in inpaint, not used in outpaint. '
                                                                          '(Outpaint always use 1.0)')
                                enhance_inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                             minimum=0.0, maximum=1.0, step=0.001,
                                                                             value=0.618,
                                                                             info='The area to inpaint. '
                                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                                  'Only used in inpaint, not used in outpaint. '
                                                                                  '(Outpaint always use 1.0)')
                                enhance_inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                            minimum=-64, maximum=64, step=1, value=0,
                                                                            info='Positive value will make white area in the mask larger, '
                                                                                 'negative value will make white area smaller. '
                                                                                 '(default is 0, always processed before any mask invert)')
                                enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)

                            gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                        enhance_ctrls += [
                            enhance_enabled,
                            enhance_mask_dino_prompt_text,
                            enhance_prompt,
                            enhance_negative_prompt,
                            enhance_mask_model,
                            enhance_mask_cloth_category,
                            enhance_mask_sam_model,
                            enhance_mask_text_threshold,
                            enhance_mask_box_threshold,
                            enhance_mask_sam_max_detections,
                            enhance_inpaint_disable_initial_latent,
                            enhance_inpaint_engine,
                            enhance_inpaint_strength,
                            enhance_inpaint_respective_field,
                            enhance_inpaint_erode_or_dilate,
                            enhance_mask_invert
                        ]

                        enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                        enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

                        enhance_inpaint_update_ctrls += [[
                            enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ]]

                        enhance_inpaint_mode.change(inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state], outputs=[
                            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                            enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ], show_progress=False, queue=False)

                        enhance_mask_model.change(
                            lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                      [gr.update(visible=x == 'sam')] * 2 +
                                      [gr.update(visible=x == 'sam',
                                                 samples=modules.config.example_enhance_detection_prompts)],
                            inputs=enhance_mask_model,
                            outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                     example_enhance_mask_dino_prompt_text],
                            queue=False, show_progress=False)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, **gc.js_kwarg(switch_js))
            ip_advanced.change(lambda: None, queue=False, show_progress=False, **gc.js_kwarg(down_js))

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, **gc.js_kwarg(down_js), show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, **gc.js_kwarg(down_js), show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, **gc.js_kwarg(down_js), show_progress=False)
            describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, **gc.js_kwarg(down_js), show_progress=False)
            enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, **gc.js_kwarg(down_js), show_progress=False)
            metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, **gc.js_kwarg(down_js), show_progress=False)
            enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                                        outputs=enhance_input_panel, queue=False, show_progress=False, **gc.js_kwarg(switch_js))

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox, elem_id='rx_panel') as advanced_column:
            with gr.Tab(label='Settings'):
                if not args_manager.args.disable_preset_selection:
                    preset_selection = gr.Dropdown(label='Preset',
                                                   choices=modules.config.available_presets,
                                                   value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                   interactive=True)

                performance_selection = gr.Radio(label='Performance',
                                                 choices=flags.Performance.values(),
                                                 value=modules.config.default_performance,
                                                 elem_classes=['performance_selection'])

                with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                    aspect_ratios_selection = gr.Radio(label='Aspect Ratios', show_label=False,
                                                       choices=modules.config.available_aspect_ratios_labels,
                                                       value=modules.config.default_aspect_ratio,
                                                       info='width × height',
                                                       elem_classes='aspect_ratios')

                    aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, **gc.js_kwarg('(x)=>{refresh_aspect_ratios_label(x);}'))
                    shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, **gc.js_kwarg('(x)=>{refresh_aspect_ratios_label(x);}'))

                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
                with gr.Group(elem_classes='rx-options'):
                    auto_quality_prompts = gr.Checkbox(label='Auto quality prompts', value=True,
                                                       info='Adds quality terms suited to the model. '
                                                            'Your prompt box is not changed.')
                    auto_negative_prompt = gr.Checkbox(label='Auto negative prompt', value=True,
                                                       info='Adds negatives that fit your prompt (e.g. '
                                                            '“illustration” for a photo); never ones it asks for. '
                                                            'Not used at CFG 1 or with FLUX.',
                                                       elem_id='auto_negative_prompt')
                    draft_mode = gr.Checkbox(label='Draft mode (fast previews)', value=False,
                                             info=f'{modules.draft_mode.DRAFT_STEPS}-step drafts, ~2.5x faster. '
                                                  'Pick one, then “Render full quality”.',
                                             visible=initial_family == modules.ui_family.SDXL)
                    prompt_queue = gr.Checkbox(label='Prompt queue', value=False,
                                               info='Each prompt line becomes its own generation.')
                draft_request = gr.State(None)       # {'mode': 'full', 'seed': int} for one run
                gallery_selected_index = gr.State(None)

                output_format = gr.Radio(label='Output Format',
                                         choices=flags.OutputFormat.list(),
                                         value=modules.config.default_output_format)

                seed_random = gr.Checkbox(label='Random', value=True)
                image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(value=f'<a href="{modules.ui_gradio_extensions.file_url(get_current_html_path(output_format))}" target="_blank">\U0001F4DA History Log</a>')

                history_link = gr.HTML()
                shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)

            with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=modules.config.default_styles)

                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="\U0001F50E Type here to search styles ...",
                                              value="",
                                              label='Search Styles')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                        outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, **gc.js_kwarg('()=>{refresh_style_localization();}'))

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, **gc.js_kwarg('()=>{refresh_style_localization();}'))

            with gr.Tab(label='Prompts', elem_id='prompts_tab') as prompts_tab:
                with gr.Row():
                    prompts_query = gr.Textbox(label='Search', placeholder='Name, prompt or tag', scale=3,
                                               container=False)
                    prompts_tag = gr.Dropdown(label='Tag', choices=['All tags'], value='All tags', scale=2,
                                              container=False, min_width=110)
                prompts_list = gr.Dropdown(label='Saved prompts', choices=[], value=None, interactive=True,
                                           elem_id='prompts_list')
                prompts_preview = gr.Textbox(label='Prompt', lines=4, interactive=False, elem_id='prompts_preview')
                with gr.Row(elem_classes='rx-toolbar'):
                    prompts_use = gr.Button(value='Use this prompt', size='sm', variant='primary', min_width=0,
                                            interactive=False, elem_id='prompts_use')
                    prompts_delete = gr.Button(value='\U0001f5d1 Delete', size='sm', variant='stop', min_width=0,
                                               interactive=False, elem_id='prompts_delete')
                gr.Markdown('**Save the current prompt**')
                with gr.Row():
                    prompts_name = gr.Textbox(label='Name (optional)', scale=3, elem_id='prompts_name')
                    prompts_tags = gr.Textbox(label='Tags (comma-separated)', scale=3, elem_id='prompts_tags')
                prompts_save = gr.Button(value='\U0001f4be Save current prompt', elem_id='prompts_save')
                prompts_status = gr.Markdown(value='', elem_classes='rx-status', elem_id='prompts_status')

            with gr.Tab(label='Models'):
                with gr.Group():
                    auto_model_settings = gr.Checkbox(label='Auto-apply best settings for the selected model',
                                                      value=True, container=False)
                    auto_model_settings_status = gr.Markdown(value='', elem_classes='rx-status')

                with gr.Column(visible=initial_family == modules.ui_family.SDXL) as refiner_col:
                    refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=refiner_choices(), value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Accordion(label='LoRAs', open=any(e and f != 'None' for e, f, _ in modules.config.default_loras),
                                  visible=True) as lora_accordion:
                    lora_ctrls = []
                    lora_models = []
                    initial_lora_choices = lora_choices(initial_family)

                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'], scale=1)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=initial_lora_choices,
                                                     value=filename if filename in initial_lora_choices else 'None',
                                                     elem_classes='lora_model', scale=5)
                            lora_models.append(lora_model)
                            lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                                    elem_classes='lora_weight', scale=5)
                            lora_ctrls += [lora_enabled, lora_model, lora_weight]

                with gr.Row():
                    refresh_files = gr.Button(value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')

                with gr.Accordion(label='Model manager', open=False):
                    # A callable value is re-evaluated on every page load.
                    model_manager_table = gr.Dataframe(value=lambda: modules.model_manager.list_installed(
                                                           modules.model_manager.default_dirs()),
                                                       headers=['Type', 'File', 'Size', 'Modified'],
                                                       datatype=['str', 'str', 'str', 'str'],
                                                       interactive=False, wrap=True)
                    model_manager_list_refresh = gr.Button(value='\U0001f504 Refresh list', variant='secondary',
                                                           elem_classes='refresh_button')
                    gr.Markdown('**Download a model** (.safetensors only)')
                    model_download_url = gr.Textbox(label='URL', placeholder='https://…/model.safetensors')
                    with gr.Row():
                        model_download_kind = gr.Dropdown(label='Type', choices=list(modules.model_manager.KINDS),
                                                          value='LoRA', scale=1)
                        model_download_name = gr.Textbox(label='Save as (optional)', placeholder='name.safetensors',
                                                         scale=2)
                    model_download_sha = gr.Textbox(label='SHA-256 (optional, verifies the download)')
                    model_download_button = gr.Button(value='Download', variant='primary',
                                                      interactive=not FILE_ACTIONS_LOCKED)
                    if FILE_ACTIONS_LOCKED:
                        gr.Markdown(modules.security.LOCKED_MESSAGE, elem_classes='rx-notice')
                    model_download_status = gr.Textbox(label='Status', interactive=False, lines=2)
            with gr.Tab(label='Advanced'):
                with gr.Column(visible=initial_family == modules.ui_family.SDXL) as sdxl_guidance_col:
                    guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                               value=modules.config.default_cfg_scale,
                                               info='Higher value means style is cleaner, vivider, and more artistic.')
                    sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                          value=modules.config.default_sample_sharpness,
                                          info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Documentation</a>')
                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=modules.config.default_developer_debug_mode_checkbox, container=False)

                with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                 value=modules.config.default_clip_skip,
                                                 info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).')
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)
                        vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                     value=modules.config.default_vae, show_label=True)

                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                               minimum=-1, maximum=1.0, step=0.001,
                                                               value=modules.config.default_overwrite_upscale,
                                                               info='Set as negative number to disable. For developer debugging.')

                        disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw,
                                                      info='Disable preview during generation.')
                        disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                      value=flags.Performance.has_restricted_features(modules.config.default_performance),
                                                      info='Disable intermediate results during generation, only show final gallery.')

                        disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                             info='Disable automatic seed increment when image number is > 1.',
                                                             value=False)
                        read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False)

                        black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='Use black image if NSFW is detected.')

                        black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                              inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                              show_progress=False)

                        if not args_manager.args.disable_image_log:
                            save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image',
                                                                         value=modules.config.default_save_only_final_enhanced_image)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme],
                                                           queue=False, show_progress=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                        debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False,
                                                                       info='Show enhance masks in preview and final results')
                        debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False,
                                                     info='Use GroundingDINO boxes instead of more detailed SAM masks')
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller. '
                                                                 '(default is 0, always processed before any mask invert)')
                        dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                                         minimum=-64, maximum=64, step=1, value=0,
                                                         info='Positive value will make white area in the mask larger, '
                                                              'negative value will make white area smaller. '
                                                              '(default is 0, processed before SAM)')

                        inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                 inputs=inpaint_advanced_masking_checkbox,
                                                                 outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                 queue=False, show_progress=False)

                        inpaint_mask_color.change(gc.update_sketch_brush_color, inputs=inpaint_mask_color,
                                                  outputs=inpaint_input_image,
                                                  queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                def dev_mode_checked(r):
                    return gr.update(visible=r)

                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def refresh_files_clicked(family):
                    modules.config.update_files()
                    results = [gr.update(choices=modules.ui_family.models_for(family, modules.config.model_filenames))]
                    results += [gr.update(choices=refiner_choices())]
                    results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(modules.config.default_max_lora_number):
                        results += [gr.update(interactive=True), gr.update(choices=lora_choices(family)), gr.update()]
                    return results

                refresh_files_output = [base_model, refiner_model, vae_name]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [model_family], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

                def model_manager_list():
                    return modules.model_manager.list_installed(modules.model_manager.default_dirs())

                def model_download_clicked(url, kind, name, sha):
                    if FILE_ACTIONS_LOCKED:  # enforced here too, not only by the disabled button
                        yield modules.security.LOCKED_MESSAGE
                        return
                    try:
                        request = modules.model_manager.prepare_download(
                            url, kind, name, sha, modules.model_manager.default_dirs())
                    except ValueError as exc:
                        yield f'Not started: {exc}'
                        return
                    state = {'done': 0, 'total': None, 'path': None, 'error': None}

                    def on_progress(done, total):
                        state['done'], state['total'] = done, total

                    def run():
                        try:
                            state['path'] = modules.model_manager.download(request, on_progress)
                        except Exception as exc:  # shown to the user below
                            state['error'] = str(exc)

                    thread = threading.Thread(target=run, daemon=True)
                    thread.start()
                    yield f'Starting download of {request.file_name} …'
                    while thread.is_alive():
                        thread.join(timeout=0.5)
                        yield modules.model_manager.progress_text(request.file_name, state['done'], state['total'])
                    if state['error']:
                        yield f'Failed: {state["error"]}'
                    else:
                        yield f'Installed {request.file_name} ({request.kind}) in {request.model_dir}'

                model_manager_list_refresh.click(model_manager_list, outputs=model_manager_table,
                                                 queue=False, show_progress=False)
                model_download_button.click(
                    model_download_clicked,
                    inputs=[model_download_url, model_download_kind, model_download_name, model_download_sha],
                    outputs=model_download_status) \
                    .then(refresh_files_clicked, [model_family], refresh_files_output + lora_ctrls, queue=False, show_progress=False) \
                    .then(model_manager_list, outputs=model_manager_table, queue=False, show_progress=False)

            with gr.Tab(label='History') as history_tab:
                with gr.Row():
                    history_query = gr.Textbox(label='Search', placeholder='Prompt, seed, model or style',
                                               scale=4, container=False)
                    history_refresh = gr.Button(value='\U0001f504 Refresh', variant='secondary',
                                                elem_classes='refresh_button', scale=1)
                with gr.Row(elem_id='history_filters'):
                    history_filter_model = gr.Dropdown(label='Model', choices=[modules.history.ALL_MODELS],
                                                       value=modules.history.ALL_MODELS, scale=3, min_width=120)
                    history_filter_day = gr.Dropdown(label='Day', choices=[modules.history.ALL_DAYS],
                                                     value=modules.history.ALL_DAYS, scale=2, min_width=110)
                    history_filter_stars = gr.Dropdown(label='Rating', choices=HISTORY_STAR_CHOICES,
                                                       value=HISTORY_STAR_CHOICES[0], scale=2, min_width=90)
                history_filter_fav = gr.Checkbox(label='\u2665 Favourites only', value=False, container=False)
                # Sized in css/style.css (#history_gallery .grid-wrap) so the grid
                # scrolls; Gradio's height= clipped it instead.
                history_gallery = gr.Gallery(show_label=False, columns=4, object_fit='contain',
                                             allow_preview=False, elem_id='history_gallery')
                with gr.Row(elem_classes='rx-toolbar'):
                    history_count = gr.Markdown(value='', elem_classes='rx-status', elem_id='history_count')
                    history_more = gr.Button(value='Load more', size='sm', min_width=0, visible=False,
                                             elem_id='history_more')
                history_select_mode = gr.Checkbox(label='Select multiple', value=False, container=False,
                                                  info='Click images to add them to the selection.')
                with gr.Row(elem_classes='rx-toolbar'):
                    history_select_all = gr.Button(value='Select all shown', size='sm', min_width=0)
                    history_clear = gr.Button(value='Clear selection', size='sm', min_width=0)
                history_marked_gallery = gr.Gallery(label='Selected (click one to unselect)', columns=8,
                                                    object_fit='cover', allow_preview=False, visible=False,
                                                    elem_id='history_marked_gallery')
                with gr.Row(elem_classes='rx-toolbar'):
                    history_delete = gr.Button(value='\U0001f5d1 Delete', size='sm', variant='stop',
                                               interactive=False, min_width=0, elem_id='history_delete')
                    history_undo = gr.Button(value='\u21a9 Undo delete', size='sm', interactive=False,
                                             min_width=0, elem_id='history_undo')
                history_status = gr.Markdown(value='', elem_classes='rx-status', elem_id='history_status')
                with gr.Row(elem_classes='rx-toolbar', elem_id='history_rating_row'):
                    history_view = gr.Button(value='\U0001f50d View larger', size='sm', min_width=0,
                                             elem_id='history_view')
                    history_fav = gr.Button(value='\u2661 Favourite', size='sm', min_width=0, interactive=False,
                                            elem_id='history_fav')
                    history_stars = gr.Radio(label='Rating', choices=HISTORY_RATING_CHOICES,
                                             value=HISTORY_RATING_CHOICES[0], interactive=False,
                                             elem_id='history_stars', container=False)
                history_details = gr.Textbox(label='Selected image', lines=6, interactive=False)
                history_load = gr.Button(value='Load these settings', interactive=False)
                history_items = gr.State([])
                history_selected = gr.State(None)
                history_marked = gr.State([])       # paths picked in "Select multiple" mode
                history_last_moves = gr.State([])   # [(original, trashed, log block)] for Undo
                history_limit = gr.State(HISTORY_PAGE)

                def history_selection_view(marked, item):
                    if FILE_ACTIONS_LOCKED:
                        delete = gr.update(value='\U0001f5d1 Delete (disabled: no login)', interactive=False)
                    elif marked:
                        delete = gr.update(value=f'\U0001f5d1 Delete {len(marked)} selected', interactive=True)
                    elif item is not None:
                        delete = gr.update(value='\U0001f5d1 Delete this image', interactive=True)
                    else:
                        delete = gr.update(value='\U0001f5d1 Delete', interactive=False)
                    marked_view = gr.update(value=[(path, os.path.basename(path)) for path in marked] or None,
                                            visible=bool(marked))
                    return [marked, marked_view, delete]

                def history_rating_view(item):
                    if item is None:
                        return [gr.update(value='\u2661 Favourite', interactive=False),
                                gr.update(value=HISTORY_RATING_CHOICES[0], interactive=False)]
                    rating = history_ratings().get(item.path)
                    return [gr.update(value='\u2665 Favourite' if rating['favourite'] else '\u2661 Favourite',
                                      interactive=True),
                            gr.update(value=HISTORY_RATING_CHOICES[rating['stars']], interactive=True)]

                def history_refresh_clicked(query, marked, model, day, stars, favourites_only, limit):
                    store = history_ratings()
                    everything = history_index.scan(modules.config.path_outputs)
                    model_choices, day_choices = modules.history.filter_choices(everything)
                    model = model if model in model_choices else modules.history.ALL_MODELS
                    day = day if day in day_choices else modules.history.ALL_DAYS
                    matches = modules.history.filter_items(
                        modules.history.search(everything, query), model, day, favourites_only,
                        HISTORY_STAR_CHOICES.index(stars) if stars in HISTORY_STAR_CHOICES else 0, store.load())
                    ratings = store.load()
                    items = matches[:limit]
                    gallery = [(modules.history.ensure_thumbnail(item.path, modules.config.path_outputs),
                                modules.history.caption(item, ratings.get(modules.history.RatingsStore.key(item.path))))
                               for item in items]
                    marked = [path for path in (marked or []) if os.path.isfile(path)]
                    count = f'Showing {len(items)} of {len(matches)}' if matches else 'No images match.'
                    return [gallery, items, None, '', gr.update(interactive=False)] + \
                        history_selection_view(marked, None) + history_rating_view(None) + \
                        [gr.update(choices=model_choices, value=model), gr.update(choices=day_choices, value=day),
                         count, gr.update(visible=len(matches) > len(items))]

                def history_select(items, select_mode, marked, evt: gr.SelectData):
                    if not items or evt.index is None or evt.index >= len(items):
                        return [None, '', gr.update(interactive=False)] + history_selection_view(marked, None) + \
                            history_rating_view(None)
                    item = items[evt.index]
                    if select_mode:
                        marked = [p for p in marked if p != item.path] if item.path in marked else marked + [item.path]
                    return [item, modules.history.describe(item), gr.update(interactive=item.source != 'none')] + \
                        history_selection_view(marked, item) + history_rating_view(item)

                def history_unmark(marked, item, evt: gr.SelectData):
                    if evt.index is not None and 0 <= evt.index < len(marked):
                        marked = marked[:evt.index] + marked[evt.index + 1:]
                    return history_selection_view(marked, item)

                def history_select_all_clicked(items, marked, item):
                    marked = list(dict.fromkeys(list(marked) + [i.path for i in items]))
                    return history_selection_view(marked, item)

                def history_delete_clicked(marked, item, *refresh_args):
                    if FILE_ACTIONS_LOCKED:
                        query, _marked, *filters = refresh_args
                        return history_refresh_clicked(query, marked, *filters) + \
                            [modules.security.LOCKED_MESSAGE, gr.update(interactive=False), []]
                    paths = list(marked) if marked else ([item.path] if item is not None else [])
                    moves, skipped = modules.history.delete_items(
                        paths, modules.config.path_outputs,
                        log_remover=modules.private_logger.remove_log_entries)
                    where = modules.history.default_trash_dir(modules.config.path_outputs)
                    where = 'the Trash' if where == os.path.expanduser('~/.Trash') else where
                    status = f'Moved {len(moves)} image(s) to {where}.' if moves else 'Nothing was deleted.'
                    if skipped:
                        status += f' {len(skipped)} could not be moved.'
                    query, _marked, *filters = refresh_args
                    return history_refresh_clicked(query, [], *filters) + \
                        [status, gr.update(interactive=bool(moves)), moves]

                def history_undo_clicked(moves, *refresh_args):
                    restored = modules.history.restore_items(
                        moves, log_restorer=modules.private_logger.restore_log_entries)
                    return history_refresh_clicked(*refresh_args) + \
                        [f'Restored {restored} image(s).', gr.update(interactive=False), []]

                def history_fav_clicked(item, *refresh_args):
                    if item is not None:
                        current = history_ratings().get(item.path)['favourite']
                        history_ratings().update(item.path, favourite=not current)
                    return history_rating_view(item)

                def history_stars_changed(item, stars):
                    if item is not None and stars in HISTORY_RATING_CHOICES:
                        history_ratings().update(item.path, stars=HISTORY_RATING_CHOICES.index(stars))
                    return history_rating_view(item)

                history_filter_inputs = [history_filter_model, history_filter_day, history_filter_stars,
                                         history_filter_fav, history_limit]
                history_refresh_inputs = [history_query, history_marked] + history_filter_inputs
                history_selection_outputs = [history_marked, history_marked_gallery, history_delete]
                history_rating_outputs = [history_fav, history_stars]
                history_refresh_outputs = [history_gallery, history_items, history_selected, history_details,
                                           history_load] + history_selection_outputs + history_rating_outputs + \
                                          [history_filter_model, history_filter_day, history_count, history_more]
                history_undo_outputs = [history_status, history_undo, history_last_moves]

                # Emptying the gallery first makes Gradio forget its selected
                # thumbnail; otherwise clicking that position again after a
                # refresh fires no select event (Gallery.svelte keeps the index).
                def history_reset_page():
                    return HISTORY_PAGE, None

                for trigger in (history_refresh.click, history_query.submit, history_tab.select,
                                history_filter_model.input, history_filter_day.input, history_filter_stars.input,
                                history_filter_fav.input):
                    trigger(history_reset_page, outputs=[history_limit, history_gallery], queue=False,
                            show_progress=False) \
                        .then(history_refresh_clicked, inputs=history_refresh_inputs,
                              outputs=history_refresh_outputs, queue=False, show_progress=False)
                history_more.click(lambda limit: limit + HISTORY_PAGE, inputs=history_limit, outputs=history_limit,
                                   queue=False, show_progress=False) \
                    .then(history_refresh_clicked, inputs=history_refresh_inputs, outputs=history_refresh_outputs,
                          queue=False, show_progress=False)
                history_gallery.select(history_select, inputs=[history_items, history_select_mode, history_marked],
                                       outputs=[history_selected, history_details, history_load] +
                                               history_selection_outputs + history_rating_outputs,
                                       queue=False, show_progress=False)
                history_marked_gallery.select(history_unmark, inputs=[history_marked, history_selected],
                                              outputs=history_selection_outputs, queue=False, show_progress=False)
                history_select_all.click(history_select_all_clicked,
                                         inputs=[history_items, history_marked, history_selected],
                                         outputs=history_selection_outputs, queue=False, show_progress=False)
                history_clear.click(lambda item: history_selection_view([], item), inputs=history_selected,
                                    outputs=history_selection_outputs, queue=False, show_progress=False)
                history_delete.click(lambda: None, outputs=history_gallery, queue=False, show_progress=False) \
                    .then(history_delete_clicked,
                                     inputs=[history_marked, history_selected] + history_refresh_inputs,
                                     outputs=history_refresh_outputs + history_undo_outputs,
                                     queue=False, show_progress=False)
                history_undo.click(lambda: None, outputs=history_gallery, queue=False, show_progress=False) \
                    .then(history_undo_clicked, inputs=[history_last_moves] + history_refresh_inputs,
                                   outputs=history_refresh_outputs + history_undo_outputs,
                                   queue=False, show_progress=False)
                history_view_url = gr.Textbox(visible=False, elem_id='history_view_url')

                def history_view_clicked(item):
                    if item is None or not os.path.isfile(item.path):
                        return ''
                    return modules.ui_gradio_extensions.file_url(item.path)  # the original, not the thumbnail

                history_view.click(history_view_clicked, inputs=history_selected, outputs=history_view_url,
                                   queue=False, show_progress=False) \
                    .then(None, inputs=history_view_url, queue=False, show_progress=False,
                          **gc.js_kwarg("(url) => { if (url) { rxOpenViewer([url], 0); } return []; }"))
                history_fav.click(history_fav_clicked, inputs=history_selected, outputs=history_rating_outputs,
                                  queue=False, show_progress=False)
                history_stars.input(history_stars_changed, inputs=[history_selected, history_stars],
                                    outputs=history_rating_outputs, queue=False, show_progress=False)

        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                             base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                             seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                             inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                             load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def preset_selection_change(preset, is_generating, inpaint_mode):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                default_model = preset_prepared.get('base_model')
                previous_default_models = preset_prepared.get('previous_default_models', [])
                checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                lora_downloads = preset_prepared.get('lora_downloads', {})
                vae_downloads = preset_prepared.get('vae_downloads', {})

                preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = launch.download_models(
                    default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
                    vae_downloads)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating, inpaint_mode)


            def inpaint_engine_state_change(inpaint_engine_version, *args):
                if inpaint_engine_version == 'empty':
                    inpaint_engine_version = modules.config.default_inpaint_engine_version

                result = []
                for inpaint_mode in args:
                    if inpaint_mode != modules.flags.inpaint_option_detail:
                        result.append(gr.update(value=inpaint_engine_version))
                    else:
                        result.append(gr.update())

                return result

            preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, **gc.js_kwarg('()=>{refresh_style_localization();}')) \
                .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)

        performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                               [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                               [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)

        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, **gc.js_kwarg('refresh_grid_delayed'), queue=False, show_progress=False)

        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        # load configured default_inpaint_method
        default_inpaint_ctrls = [inpaint_mode, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
        for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
            shared.gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
                inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
                engine, strength, respective_field
            ], show_progress=False, queue=False)

        generate_mask_button.click(fn=generate_mask,
                                   inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                           inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                           inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                           inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                                   outputs=inpaint_mask_image, show_progress=True, queue=True)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
            read_wildcards_in_order, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
        ctrls += [sampler_name, scheduler_name, vae_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        if not args_manager.args.disable_image_log:
            ctrls += [save_final_enhanced_image_only]

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls
        ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                  enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
                  enhance_uov_prompt_type]
        ctrls += enhance_ctrls
        ctrls += [prompt_queue]  # AsyncTask reads these after the enhance controls, in this order
        ctrls += [auto_quality_prompts]
        ctrls += [auto_negative_prompt]

        # get_task drops currentTask (ctrls[0]), hence the -1.
        DRAFT_ARG_INDEX.update({
            'performance': ctrls.index(performance_selection) - 1,
            'overwrite_step': ctrls.index(overwrite_step) - 1,
            'sampler_name': ctrls.index(sampler_name) - 1,
            'scheduler_name': ctrls.index(scheduler_name) - 1,
            'image_seed': ctrls.index(image_seed) - 1,
            'image_number': ctrls.index(image_number) - 1,
        })
        task_inputs = ctrls + [draft_mode, draft_request]  # get_task strips the last two

        # --- Prompt library (Prompts tab + toolbar "Save prompt") ---------------
        def prompt_library():
            return modules.prompt_library.PromptLibrary(modules.config.path_outputs)

        def prompts_refresh(query, tag, selected=None):
            lib = prompt_library()
            entries = lib.search(query, '' if tag in (None, 'All tags') else tag)
            choices = [(modules.prompt_library.label(e), e['id']) for e in entries]
            ids = [e['id'] for e in entries]
            selected = selected if selected in ids else None
            return [gr.update(choices=choices, value=selected),
                    gr.update(choices=['All tags'] + lib.all_tags(), value=tag or 'All tags')] + \
                prompts_pick(selected)

        def prompts_pick(entry_id):
            entry = prompt_library().get(entry_id) if entry_id else None
            if entry is None:
                return ['', gr.update(interactive=False), gr.update(interactive=False)]
            text = entry['prompt'] + (f"\n\nNegative: {entry['negative']}" if entry.get('negative') else '')
            return [text, gr.update(interactive=True), gr.update(interactive=True)]

        def prompts_save_clicked(current_prompt, current_negative, name, tags, query, tag):
            try:
                entry = prompt_library().save(current_prompt, current_negative, name, tags)
            except ValueError as exc:
                return [gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), str(exc), gr.update(),
                        gr.update()]
            return prompts_refresh(query, tag, entry['id']) + [f"Saved \u201c{entry['name']}\u201d.", '', '']

        def prompts_use_clicked(entry_id):
            entry = prompt_library().get(entry_id) if entry_id else None
            if entry is None:
                return gr.update(), gr.update()
            return entry['prompt'], entry.get('negative', '')

        def prompts_delete_clicked(entry_id, query, tag):
            deleted = prompt_library().delete(entry_id) if entry_id else False
            return prompts_refresh(query, tag) + ['Deleted.' if deleted else 'Nothing was deleted.']

        prompts_outputs = [prompts_list, prompts_tag, prompts_preview, prompts_use, prompts_delete]
        for trigger in (prompts_tab.select, prompts_query.submit, prompts_tag.input):
            trigger(prompts_refresh, inputs=[prompts_query, prompts_tag], outputs=prompts_outputs,
                    queue=False, show_progress=False)
        prompts_list.input(prompts_pick, inputs=prompts_list, outputs=prompts_outputs[2:], queue=False,
                           show_progress=False)
        prompts_save.click(prompts_save_clicked,
                           inputs=[prompt, negative_prompt, prompts_name, prompts_tags, prompts_query, prompts_tag],
                           outputs=prompts_outputs + [prompts_status, prompts_name, prompts_tags],
                           queue=False, show_progress=False)
        prompts_use.click(prompts_use_clicked, inputs=prompts_list, outputs=[prompt, negative_prompt],
                          queue=False, show_progress=False)
        prompts_delete.click(prompts_delete_clicked, inputs=[prompts_list, prompts_query, prompts_tag],
                             outputs=prompts_outputs + [prompts_status], queue=False, show_progress=False)

        def save_prompt_toolbar(current_prompt, current_negative):
            try:
                entry = prompt_library().save(current_prompt, current_negative)
            except ValueError as exc:
                raise gr.Error(str(exc))
            gr.Info(f"Saved to Prompts: \u201c{entry['name']}\u201d")

        view_larger_button.click(None, queue=False, show_progress=False,
                                 **gc.js_kwarg("() => { rxViewGallery('#final_gallery'); return []; }"))
        save_prompt_button.click(save_prompt_toolbar, inputs=[prompt, negative_prompt], queue=False,
                                 show_progress=False)

        # --- Compare A/B: same prompt, settings and seed on two base models ------
        def compare_clicked(model_b, seed_is_random, *args):
            import ldm_patched.modules.model_management as model_management
            args = list(args)
            with model_management.interrupt_processing_mutex:
                model_management.interrupt_processing = False
            modules.prompt_rewriter.release()
            args[-1], args[-2] = None, False  # no draft request, draft mode off
            seed = refresh_seed(seed_is_random, args[ctrls.index(image_seed)])
            args[ctrls.index(image_seed)] = seed
            args[ctrls.index(image_number)] = 1
            model_a = args[ctrls.index(base_model)]
            args_b = list(args)
            args_b[ctrls.index(base_model)] = model_b
            hidden = [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
            yield [gr.update(visible=True, value=modules.html.make_progress_html(1, 'Compare: starting A ...')),
                   gr.update(visible=True, value=None), gr.update(visible=False), gr.update(visible=False),
                   gr.update(interactive=False), seed]
            results = []
            for name, model, task_args in (('A', model_a, args), ('B', model_b, args_b)):
                task = get_task(*task_args)
                worker.async_tasks.append(task)
                finished = False
                while not finished:
                    time.sleep(0.02)
                    while task.yields:
                        flag, product = task.yields.pop(0)
                        if flag == 'preview':
                            percentage, title, image = product
                            yield [gr.update(visible=True, value=modules.html.make_progress_html(
                                       percentage, f'Compare {name} ({model}): {title}')),
                                   gr.update(visible=True, value=image) if image is not None else gr.update(),
                                   gr.update(), gr.update(visible=False), gr.update(), gr.update()]
                        elif flag == 'finish':
                            images = [p for p in (product or []) if isinstance(p, str)]
                            results.append((images[-1], f'{name} \u00b7 {model}') if images else None)
                            finished = True
            pairs = [r for r in results if r]
            yield hidden + [gr.update(visible=True, value=pairs or None), gr.update(interactive=True), gr.update()]

        compare_button.click(compare_clicked, inputs=[compare_model_b, seed_random] + task_inputs,
                             outputs=[progress_html, progress_window, progress_gallery, gallery, compare_button,
                                      image_seed])
        base_model.change(lambda: gr.update(choices=modules.config.model_filenames), outputs=compare_model_b,
                          queue=False, show_progress=False)

        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            if is_json(raw_prompt_txt):
                loaded_json = json.loads(raw_prompt_txt)

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(file, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                print('Could not find metadata in the image!')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.to_json(parameters)

            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating, inpaint_mode)

        def rewrite_prompt_clicked(current_prompt, model_name):
            try:
                rewritten = modules.prompt_rewriter.rewrite(
                    current_prompt, modules.prompt_rewriter.style_for_model(model_name))
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                raise gr.Error(str(exc))
            return rewritten, current_prompt, gr.update(interactive=True)

        rewrite_button.click(rewrite_prompt_clicked, inputs=[prompt, base_model],
                             outputs=[prompt, prompt_before_rewrite, rewrite_undo], show_progress=True)
        rewrite_undo.click(lambda previous: (previous, None, gr.update(interactive=False)),
                           inputs=prompt_before_rewrite, outputs=[prompt, prompt_before_rewrite, rewrite_undo],
                           queue=False, show_progress=False)

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        # Best settings for the selected checkpoint. Wired to .input (user picks
        # only), so loading a preset or an image's settings isn't overridden.
        profile_components = {
            'performance': performance_selection, 'guidance_scale': guidance_scale, 'sharpness': sharpness,
            'sampler_name': sampler_name, 'scheduler_name': scheduler_name, 'overwrite_step': overwrite_step,
            'aspect_ratio': aspect_ratios_selection, 'vae_name': vae_name, 'clip_skip': clip_skip,
            'adaptive_cfg': adaptive_cfg,
        }
        profile_keys = list(profile_components)

        def base_model_selected(model, enabled, *current_values):
            unchanged = [gr.update()] * len(profile_keys)
            if not enabled or not model:
                return unchanged + [gr.update()]
            path = modules.util.get_file_from_folder_list(model, modules.config.paths_checkpoints)
            profile = modules.model_profiles.choose_profile(
                model, path, modules.model_profiles.load_presets(os.path.abspath('./presets')))
            if not profile.supported:
                return unchanged + [f'{model}: {profile.reason}.']
            new = modules.model_profiles.resolve_for_ui(
                profile.settings, vae_filenames=modules.config.vae_filenames,
                aspect_ratio_labels=modules.config.available_aspect_ratios_labels,
                add_ratio=modules.config.add_ratio, default_vae=flags.default_vae)
            changes = modules.model_profiles.describe_changes(dict(zip(profile_keys, current_values)), new)
            status = f'Best settings for {model} ({profile.reason}). '
            status += ('Changed: ' + '; '.join(changes)) if changes else 'Your settings already match.'
            return [new.get(key, gr.update()) for key in profile_keys] + [status]

        base_model.input(base_model_selected,
                         inputs=[base_model, auto_model_settings] + [profile_components[k] for k in profile_keys],
                         outputs=[profile_components[k] for k in profile_keys] + [auto_model_settings_status],
                         queue=False, show_progress=False)

        # Model family switch. The family dropdown reacts to user picks only
        # (.input); base_model.change follows every source (user, preset,
        # History, loaded parameters) and keeps the family, the model list and
        # the SDXL-only controls in step, without re-triggering itself.
        sdxl_only = [negative_prompt_accordion, performance_selection, draft_mode, render_full_button, auto_negative_prompt,
                     refiner_col, sdxl_guidance_col]

        def family_selected(family, memory):
            family, choices, model, warning = modules.ui_family.select_family(
                family, memory, modules.config.model_filenames)
            if warning:
                gr.Warning(warning)
            if model is None:
                return gr.update(value=family), gr.update()
            return gr.update(value=family), gr.update(choices=choices, value=model)

        model_family.input(family_selected, inputs=[model_family, family_memory],
                           outputs=[model_family, base_model], queue=False, show_progress=False)

        def base_model_changed(model, shown_family, memory, *current_loras):
            family, memory, choices = modules.ui_family.model_changed(
                model, shown_family, memory, modules.config.model_filenames)
            # LoRAs: each dropdown lists only this family's LoRAs; each family
            # remembers its own picks (so the SDXL offset LoRA comes back).
            if family != shown_family:
                memory[f'loras:{shown_family}'] = list(current_loras)
                options = lora_choices(family)
                remembered = memory.get(f'loras:{family}') or []
                lora_updates = []
                for i, current in enumerate(current_loras):
                    if current != 'None' and current in options:
                        value = current  # e.g. History > Load set this family's LoRA with the model
                    else:
                        value = remembered[i] if i < len(remembered) else 'None'
                    lora_updates.append(gr.update(choices=options, value=value if value in options else 'None'))
            else:
                lora_updates = [gr.update() for _ in current_loras]
            sdxl = family == modules.ui_family.SDXL
            # FLUX keeps Input Image (Upscale or Variation works) but hides the
            # Image Prompt / Inpaint tabs; Krea 2 / MOSS-Image are text-to-image
            # only, so Input Image goes too. Hidden toggles are unticked so
            # nothing stays on out of sight.
            if sdxl:
                image_toggles = [gr.update(visible=True), gr.update(visible=True)]
            elif family == modules.ui_family.FLUX:
                image_toggles = [gr.update(visible=True), gr.update(visible=False, value=False)]
            else:
                image_toggles = [gr.update(visible=False, value=False), gr.update(visible=False, value=False)]
            image_toggles += [gr.update(visible=sdxl), gr.update(visible=sdxl)]  # ip_tab, inpaint_tab
            notice = gr.update(visible=False) if sdxl else \
                gr.update(visible=True, value=modules.ui_family.NOTICES.get(family, ''))
            # Negative prompt also applies to MOSS-Image (CFG 3); the rest is SDXL-only.
            family_toggles = [gr.update(visible=modules.ui_family.uses_negative_prompt(family))] + \
                [gr.update(visible=sdxl)] * (len(sdxl_only) - 1)
            return ([gr.update(value=family), memory,
                     gr.update() if choices is None else gr.update(choices=choices),
                     notice] + image_toggles + family_toggles + lora_updates)

        base_model.change(base_model_changed, inputs=[base_model, model_family, family_memory] + lora_models,
                          outputs=[model_family, family_memory, base_model, flux_notice,
                                   input_image_checkbox, enhance_checkbox, ip_tab, inpaint_tab] + sdxl_only + lora_models,
                          queue=False, show_progress=False)

        # Gradio 5 ignores visibility updates on tabs, so also hide the Image
        # Prompt / Inpaint tab buttons in FLUX mode with a page class + CSS
        # (keyed by tab id, so translated labels don't matter; harmless on 4.x).
        _flux_tab_ids = [str(ip_tab.id), str(inpaint_tab.id), str(ip_tab._id), str(inpaint_tab._id)]
        _flux_tab_css = ', '.join(f'body.rx-flux button[data-tab-id="{i}"]' for i in _flux_tab_ids)
        flux_class_js = ("(family) => { let s = document.getElementById('rx-flux-style'); if (!s) { "
                         "s = document.createElement('style'); s.id = 'rx-flux-style'; "
                         f"s.textContent = '{_flux_tab_css} {{ display: none !important; }}'; "
                         "document.head.appendChild(s); } "
                         "document.body.classList.toggle('rx-flux', !!family && family !== 'SDXL'); return []; }")
        model_family.change(None, inputs=model_family, queue=False, show_progress=False, **gc.js_kwarg(flux_class_js))
        shared.gradio_root.load(None, inputs=model_family, queue=False, show_progress=False,
                                **gc.js_kwarg(flux_class_js))

        def history_load_clicked(item, is_generating):
            if item is None:
                return [gr.update()] * len(load_data_outputs)
            params = modules.history.to_load_parameters(
                item.metadata, modules.config.model_filenames, modules.config.lora_filenames,
                modules.config.vae_filenames)
            return modules.meta_parser.load_parameter_button_click(params, is_generating, inpaint_mode)

        history_load.click(history_load_clicked, inputs=[history_selected, state_is_generating],
                           outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        def start_generation_ui():
            return (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True),
                    gr.update(visible=False, interactive=False), [], True, None)

        start_generation_outputs = [stop_button, skip_button, generate_button, gallery, state_is_generating,
                                    gallery_selected_index]

        def run_generation(event, refresh_seed_first):
            """The shared Generate pipeline, appended to ``event``."""
            if refresh_seed_first:
                event = event.then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed)
            return event \
                .then(fn=get_task, inputs=task_inputs, outputs=currentTask) \
                .then(fn=lambda: None, outputs=draft_request) \
                .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
                .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                      outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
                .then(fn=update_history_link, outputs=history_link) \
                .then(fn=lambda: None, **gc.js_kwarg('playNotification')).then(fn=lambda: None, **gc.js_kwarg('refresh_grid_delayed'))

        run_generation(generate_button.click(start_generation_ui, outputs=start_generation_outputs),
                       refresh_seed_first=True)

        def prepare_full_render(task, index):
            try:
                seed = modules.draft_mode.seed_for_result(task, index)
            except ValueError as exc:
                raise gr.Error(str(exc))
            return {'mode': 'full', 'seed': seed}

        # .success() stops here when prepare_full_render raises (message shown as a toast).
        run_generation(render_full_button.click(prepare_full_render, inputs=[currentTask, gallery_selected_index],
                                                outputs=draft_request, queue=False, show_progress=False)
                       .success(start_generation_ui, outputs=start_generation_outputs),
                       refresh_seed_first=False)

        def gallery_selected(evt: gr.SelectData):
            return evt.index

        gallery.select(gallery_selected, outputs=gallery_selected_index, queue=False, show_progress=False)

        reset_button.click(lambda: [worker.AsyncTask(args=[]), False, gr.update(visible=True, interactive=True)] +
                                   [gr.update(visible=False)] * 6 +
                                   [gr.update(visible=True, value=[])],
                           outputs=[currentTask, state_is_generating, generate_button,
                                    reset_button, stop_button, skip_button,
                                    progress_html, progress_window, progress_gallery, gallery],
                           queue=False)

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(modes, img, apply_styles):
            describe_prompts = []
            styles = set()

            if flags.describe_type_photo in modes:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                describe_prompts.append(default_interrogator_photo(img))
                styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

            if flags.describe_type_anime in modes:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                describe_prompts.append(default_interrogator_anime(img))
                styles.update(["Fooocus V2", "Fooocus Masterpiece"])

            if len(styles) == 0 or not apply_styles:
                styles = gr.update()
            else:
                styles = list(styles)

            if len(describe_prompts) == 0:
                describe_prompt = gr.update()
            else:
                describe_prompt = ', '.join(describe_prompts)

            return describe_prompt, styles

        describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                           outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, **gc.js_kwarg('()=>{refresh_style_localization();}'))

        if args_manager.args.enable_auto_describe_image:
            def trigger_auto_describe(mode, img, prompt, apply_styles):
                # keep prompt if not empty
                if prompt == '':
                    return trigger_describe(mode, img, apply_styles)
                return gr.update(), gr.update()

            uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                                   outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, **gc.js_kwarg('()=>{refresh_style_localization();}'))

            enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                      outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, **gc.js_kwarg('()=>{refresh_style_localization();}'))

def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

# Gradio 4 rejects any `file=` request outside `allowed_paths` with a 403,
# even for files this app itself injects via <script src=...>/<link href=...>
# in ui_gradio_extensions.reload_javascript(). Upstream's allowed_paths only
# covered the outputs folder, so every custom JS file (script.js,
# localization.js, zoom.js, edit-attention.js, viewer.js, imageviewer.js,
# contextMenus.js) and style.css silently 404/403'd after the Gradio 4
# migration — surfacing as scattered "X is not defined" JS console errors
# (e.g. refresh_aspect_ratios_label) rather than an obvious asset-loading
# failure. Allow just the two directories those assets live in.
_webui_root = os.path.dirname(os.path.abspath(__file__))

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[
        modules.config.path_outputs,
        os.path.join(_webui_root, 'javascript'),
        os.path.join(_webui_root, 'css'),
    ],
    blocked_paths=[constants.AUTH_FILENAME]
)
