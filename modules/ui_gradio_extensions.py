# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager

from modules.localization import localization_js


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


# Gradio 5 serves files at /gradio_api/file=<absolute path>; Gradio 4 at /file=<path>.
FILE_ROUTE = 'gradio_api/file=' if int(gr.__version__.split('.')[0]) >= 5 else 'file='


def file_url(fn):
    """URL (relative to the app root) for a local file Gradio may serve."""
    if FILE_ROUTE == 'file=' and fn.startswith(script_path):
        return FILE_ROUTE + os.path.relpath(fn, script_path).replace('\\', '/')
    return FILE_ROUTE + os.path.abspath(fn)


def webpath(fn):
    return f'{file_url(fn)}?{os.path.getmtime(fn)}'


JS_FILES = ('script.js', 'contextMenus.js', 'localization.js', 'zoom.js', 'edit-attention.js', 'viewer.js',
            'imageviewer.js', 'rx_viewer.js')


def _inline_script(name):
    # Inlined like style.css: Gradio serves file= as application/octet-stream,
    # and Gradio 5 adds X-Content-Type-Options: nosniff, so browsers refuse to
    # run <script src="file=..."> at all. Our own files: only </script needs escaping.
    with open(os.path.join(script_path, 'javascript', name), encoding='utf-8') as f:
        code = f.read().replace('</script', '<\\/script')
    return f'<script type="text/javascript" data-file="{name}">\n{code}\n</script>\n'


def javascript_html():
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    head += ''.join(_inline_script(name) for name in JS_FILES)
    head += f'<meta name="samples-path" content="{samples_path}">\n'

    if args_manager.args.theme:
        head += f'<script type="text/javascript">set_theme(\"{args_manager.args.theme}\");</script>\n'

    return head


def css_html():
    # Inlined: Gradio 4's file= route serves .css as application/octet-stream,
    # and browsers refuse a <link>ed stylesheet with that type, so none of
    # style.css applied. Our own file, so no escaping beyond </style>.
    with open(os.path.join(script_path, 'css', 'style.css'), encoding='utf-8') as f:
        css = f.read().replace('</style', '<\\/style')
    return f'<style id="foocusrx-style">\n{css}\n</style>'


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
