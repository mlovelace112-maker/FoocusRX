import html
import os
import re
import threading
import args_manager
import modules.config
import json
import urllib.parse

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from modules.flags import OutputFormat
from modules.meta_parser import MetadataParser, get_exif
from modules.util import generate_temp_filename

log_cache = {}
# Serialises log.html rewrites: generation (log) and History deletes
# (remove_log_entries / restore_log_entries) both rewrite the whole file.
log_lock = threading.RLock()
LOG_SPLIT = '<!--fooocus-log-split-->'
_ITEM_START_RE = re.compile(r'(?=<div id="[^"]+" class="image-container">)')
_ITEM_ID_RE = re.compile(r'<div id="([^"]+)" class="image-container">')


def build_log_item(only_name: str, metadata, task=None) -> str:
    """Render one image's entry in log.html.

    Every user-influenced string (prompts, styles, LoRA names, metadata
    loaded from a dragged-in image) is HTML-escaped. log.html is served
    from the same origin as the Gradio app, so an unescaped prompt like
    ``<script>...</script>`` would execute there.
    """
    def esc(value) -> str:
        return html.escape(str(value), quote=True)

    div_name = esc(only_name.replace('.', '_'))
    name = esc(only_name)
    item = f"<div id=\"{div_name}\" class=\"image-container\"><hr><table><tr>\n"
    item += f"<td><a href=\"{name}\" target=\"_blank\"><img src='{name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'/></a><div>{name}</div></td>"
    item += "<td><table class='metadata'>"
    for label, key, value in metadata:
        value_txt = esc(value).replace('\n', ' </br> ')
        item += f"<tr><td class='label'>{esc(label)}</td><td class='value'>{value_txt}</td></tr>\n"

    if task is not None and 'positive' in task and 'negative' in task:
        positive = ', '.join(esc(p) for p in task['positive'])
        negative = ', '.join(esc(n) for n in task['negative'])
        full_prompt_details = f"""<details><summary>Positive</summary>{positive}</details>
        <details><summary>Negative</summary>{negative}</details>"""
        item += f"<tr><td class='label'>Full raw prompt</td><td class='value'>{full_prompt_details}</td></tr>\n"

    item += "</table>"

    # quote(..., safe='') percent-encodes quotes and angle brackets, so this
    # is safe inside the single-quoted JS string in the onclick attribute.
    js_txt = urllib.parse.quote(json.dumps({k: v for _, k, v, in metadata}, indent=0), safe='')
    item += f"</br><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>"

    item += "</td>"
    item += "</tr></table></div>\n\n"
    return item


def get_current_html_path(output_format=None):
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs,
                                                                         extension=output_format)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name


def log(img, metadata, metadata_parser: MetadataParser | None = None, output_format=None, task=None, persist_image=True) -> str:
    path_outputs = modules.config.temp_path if args_manager.args.disable_image_log or not persist_image else modules.config.path_outputs
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=path_outputs, extension=output_format)
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    parsed_parameters = metadata_parser.to_string(metadata.copy()) if metadata_parser is not None else ''
    image = Image.fromarray(img)

    if output_format == OutputFormat.PNG.value:
        if parsed_parameters != '':
            pnginfo = PngInfo()
            pnginfo.add_text('parameters', parsed_parameters)
            pnginfo.add_text('fooocus_scheme', metadata_parser.get_scheme().value)
        else:
            pnginfo = None
        image.save(local_temp_filename, pnginfo=pnginfo)
    elif output_format == OutputFormat.JPEG.value:
        image.save(local_temp_filename, quality=95, optimize=True, progressive=True, exif=get_exif(parsed_parameters, metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
    elif output_format == OutputFormat.WEBP.value:
        image.save(local_temp_filename, quality=95, lossless=False, exif=get_exif(parsed_parameters, metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
    else:
        image.save(local_temp_filename)

    if args_manager.args.disable_image_log:
        return local_temp_filename

    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .label { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        "hr { border-color: gray; } "
        "button { background-color: black; color: white; border: 1px solid grey; border-radius: 5px; padding: 5px 10px; text-align: center; display: inline-block; font-size: 16px; cursor: pointer; }"
        "button:hover {background-color: grey; color: black;}"
        "</style>"
    )

    js = (
        """<script>
        function to_clipboard(txt) { 
        txt = decodeURIComponent(txt);
        if (navigator.clipboard && navigator.permissions) {
            navigator.clipboard.writeText(txt)
        } else {
            const textArea = document.createElement('textArea')
            textArea.value = txt
            textArea.style.width = 0
            textArea.style.position = 'fixed'
            textArea.style.left = '-999px'
            textArea.style.top = '10px'
            textArea.setAttribute('readonly', 'readonly')
            document.body.appendChild(textArea)

            textArea.select()
            document.execCommand('copy')
            document.body.removeChild(textArea)
        }
        alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\nCurrent clipboard content is:\\n\\n' + txt);
        }
        </script>"""
    )

    begin_part = f"<!DOCTYPE html><html><head><title>Fooocus Log {date_string}</title>{css_styles}</head><body>{js}<p>Fooocus Log {date_string} (private)</p>\n<p>Metadata is embedded if enabled in the config or developer debug mode. You can find the information for each image in line Metadata Scheme.</p><!--fooocus-log-split-->\n\n"
    end_part = f'\n<!--fooocus-log-split--></body></html>'

    with log_lock:
        middle_part = log_cache.get(html_name, "")

        if middle_part == "":
            if os.path.exists(html_name):
                existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--fooocus-log-split-->')
                if len(existing_split) == 3:
                    middle_part = existing_split[1]
                else:
                    middle_part = existing_split[0]

        middle_part = build_log_item(only_name, metadata, task) + middle_part

        with open(html_name, 'w', encoding='utf-8') as f:
            f.write(begin_part + middle_part + end_part)

        log_cache[html_name] = middle_part

    print(f'Image generated with private log at: {html_name}')

    return local_temp_filename


def _split_items(middle: str) -> list:
    return [chunk for chunk in _ITEM_START_RE.split(middle) if chunk.strip()]


def _item_id(block: str) -> str:
    match = _ITEM_ID_RE.match(block)
    return html.unescape(match.group(1)) if match else ''


def _rewrite_middle(html_name: str, edit) -> object:
    """Apply ``edit(items) -> (new_items, result)`` to a log.html's entries,
    atomically, keeping the page header/footer. Returns ``result`` (None when
    the file is missing or not in the expected format)."""
    with log_lock:
        try:
            with open(html_name, encoding='utf-8') as f:
                parts = f.read().split(LOG_SPLIT)
        except OSError:
            return None
        if len(parts) != 3:
            return None
        middle = log_cache.get(html_name) or parts[1]
        items, result = edit(_split_items(middle))
        new_middle = '\n\n' + ''.join(items) if items else '\n\n'
        tmp = html_name + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(parts[0] + LOG_SPLIT + new_middle + LOG_SPLIT + parts[2])
        os.replace(tmp, html_name)
        if html_name in log_cache:
            log_cache[html_name] = new_middle
        return result


def remove_log_entries(html_name: str, filenames) -> dict:
    """Drop the entries for ``filenames`` from a day's log.html.
    Returns {filename: removed_html_block} for restore_log_entries."""
    wanted = {name.replace('.', '_'): name for name in filenames}

    def edit(items):
        kept, removed = [], {}
        for block in items:
            name = wanted.get(_item_id(block))
            if name is not None and name not in removed:
                removed[name] = block
            else:
                kept.append(block)
        return kept, removed

    return _rewrite_middle(html_name, edit) or {}


def restore_log_entries(html_name: str, blocks) -> int:
    """Put back blocks from remove_log_entries, newest first by filename
    (names start with the generation timestamp)."""
    blocks = [b for b in blocks or [] if b]

    def edit(items):
        present = {_item_id(b) for b in items}
        added = 0
        for block in blocks:
            if _item_id(block) in present:
                continue
            index = next((i for i, other in enumerate(items) if _item_id(other) < _item_id(block)), len(items))
            items.insert(index, block)
            present.add(_item_id(block))
            added += 1
        return items, added

    return _rewrite_middle(html_name, edit) or 0
