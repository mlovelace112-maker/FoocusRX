"""FoocusRX look: a Gradio theme plus the page header.

System fonts only (no Google Fonts request), so the UI works offline. Colours
come from Gradio's palette, which defines light and dark values for each one;
the theme follows --theme / ?__theme= like before.
"""
import html

import gradio as gr

SYSTEM_FONTS = ['-apple-system', 'BlinkMacSystemFont', 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue',
                'Arial', 'sans-serif']
MONO_FONTS = ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace']


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
        font=SYSTEM_FONTS,
        font_mono=MONO_FONTS,
    ).set(
        body_background_fill='*neutral_50',
        body_background_fill_dark='*neutral_950',
        background_fill_primary='white',
        background_fill_primary_dark='*neutral_900',
        background_fill_secondary='*neutral_50',
        background_fill_secondary_dark='*neutral_900',
        block_background_fill='white',
        block_background_fill_dark='*neutral_900',
        block_border_color='*neutral_200',
        block_border_color_dark='*neutral_800',
        block_label_background_fill='*neutral_100',
        block_label_background_fill_dark='*neutral_800',
        block_title_text_weight='600',
        block_shadow='0 1px 2px rgba(15, 23, 42, 0.06)',
        block_shadow_dark='none',
        input_background_fill='*neutral_50',
        input_background_fill_dark='*neutral_950',
        input_border_color='*neutral_200',
        input_border_color_dark='*neutral_700',
        input_border_color_focus='*primary_400',
        input_border_color_focus_dark='*primary_500',
        button_primary_background_fill='linear-gradient(135deg, *primary_500, *secondary_600)',
        button_primary_background_fill_dark='linear-gradient(135deg, *primary_500, *secondary_600)',
        button_primary_background_fill_hover='linear-gradient(135deg, *primary_600, *secondary_700)',
        button_primary_background_fill_hover_dark='linear-gradient(135deg, *primary_400, *secondary_500)',
        button_primary_text_color='white',
        button_primary_text_color_dark='white',
        button_primary_border_color='transparent',
        button_primary_border_color_dark='transparent',
        button_secondary_background_fill='*neutral_100',
        button_secondary_background_fill_dark='*neutral_800',
        button_secondary_background_fill_hover='*neutral_200',
        button_secondary_background_fill_hover_dark='*neutral_700',
        checkbox_label_background_fill_selected='*primary_50',
        checkbox_label_background_fill_selected_dark='*primary_950',
        slider_color='*primary_500',
        slider_color_dark='*primary_400',
    )


def header_html(title: str, subtitle: str = 'SDXL and FLUX.1 image generation on Apple Silicon') -> str:
    return (f'<div class="rx-header"><div class="rx-logo" aria-hidden="true">RX</div>'
            f'<div><div class="rx-title">{html.escape(title)}</div>'
            f'<div class="rx-subtitle">{html.escape(subtitle)}</div></div></div>')
