import sys

# args_manager parses sys.argv at import time; drop the test runner's own arguments.
sys.argv = sys.argv[:1]

import json
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import numpy as np

from modules import prompt_studio as ps

LABELS = ['704×1408 <span style="color: grey;"> ∣ 1:2</span>',
          '896×1152 <span style="color: grey;"> ∣ 7:9</span>',
          '1024×1024 <span style="color: grey;"> ∣ 1:1</span>',
          '1152×896 <span style="color: grey;"> ∣ 9:7</span>',
          '1344×768 <span style="color: grey;"> ∣ 7:4</span>']
STYLES = ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp', 'SAI Cinematic', 'SAI Anime', 'Watercolor 2']


class FakeOllama:
    """A tiny Ollama stand-in: /api/tags, /api/chat, /api/generate."""

    def __init__(self, models=('gemma3:4b',), reply=None):
        self.models = list(models)
        self.reply = reply or {'positive': 'a ginger cat on a stormy lighthouse balcony, rain, dramatic light',
                               'negative': 'cartoon, illustration, cartoon',
                               'styles': ['sai cinematic', 'Not A Style', 'Fooocus Enhance'],
                               'aspect': 'landscape', 'notes': 'Went for a moody, cinematic frame.'}
        self.requests = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {'models': [{'name': m} for m in fake.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fake.requests.append((self.path, body))
                if self.path == '/api/generate':
                    return self._send(200, {'done': True})
                if body['model'] not in fake.models:
                    return self._send(404, {'error': f"model '{body['model']}' not found"})
                if body['messages'][-1].get('images'):
                    return self._send(200, {'message': {'content': 'A cat sitting on a railing at dusk.'}})
                return self._send(200, {'message': {'content': json.dumps(fake.reply)}})

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def dead_url():
    server = HTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    return f'http://127.0.0.1:{port}'


class TestHelpers(unittest.TestCase):
    def test_pick_aspect(self):
        self.assertEqual(ps.pick_aspect('square', LABELS), LABELS[2])
        self.assertEqual(ps.pick_aspect('Landscape', LABELS), LABELS[3])
        self.assertEqual(ps.pick_aspect('wide', LABELS), LABELS[4])
        self.assertEqual(ps.pick_aspect('tall', LABELS), LABELS[0])
        self.assertIsNone(ps.pick_aspect('sideways', LABELS))
        self.assertEqual(ps.aspect_display(LABELS[3]), '1152×896 (9:7)')

    def test_pick_styles(self):
        self.assertEqual(ps.pick_styles(['sai cinematic', 'nope', 'SAI Cinematic', 'Watercolor 2', 'SAI Anime',
                                         'Fooocus Enhance'], STYLES), ['SAI Cinematic', 'Watercolor 2', 'SAI Anime'])
        self.assertEqual(ps.available_styles(STYLES), ['Fooocus Enhance', 'Fooocus Sharp', 'SAI Cinematic',
                                                       'SAI Anime', 'Watercolor 2'])

    def test_parse_json_reply(self):
        self.assertEqual(ps.parse_json_reply('```json\n{"positive": "x"}\n```')['positive'], 'x')
        self.assertEqual(ps.parse_json_reply('Sure! {"positive": "y"} hope it helps')['positive'], 'y')
        self.assertEqual(ps.parse_json_reply('just a prompt, no json')['positive'], 'just a prompt, no json')

    def test_clean_negative(self):
        self.assertEqual(ps.clean_negative('blurry, Blurry,\n text. , watermark'), 'blurry, text, watermark')

    def test_resolve_target(self):
        self.assertEqual(ps.resolve_target(ps.TARGET_AUTO, 'flux1-dev-Q4.gguf'), 'flux')
        self.assertEqual(ps.resolve_target(ps.TARGET_AUTO, 'ponyDiffusionV6XL.safetensors'), 'tags')
        self.assertEqual(ps.resolve_target('SDXL phrases', 'flux1-dev-Q4.gguf'), 'sdxl')

    def test_prompts(self):
        system = ps.build_system_prompt('sdxl', ['SAI Anime'])
        self.assertIn('SAI Anime', system)
        self.assertIn('Stable Diffusion XL', system)
        self.assertIn('empty string', ps.build_system_prompt('flux', []))
        user = ps.build_user_prompt('a fox', 'PyraCanny', 'a dog on grass', False)
        self.assertIn('a dog on grass', user)
        self.assertIn('outlines', user)
        self.assertNotIn('Reference', ps.build_user_prompt('a fox', None, None, False))

    def test_image_to_png_b64_downscales(self):
        import base64
        import io
        from PIL import Image
        data = ps.image_to_png_b64(np.zeros((2000, 1000, 3), dtype=np.uint8), max_side=512)
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(data))).size, (256, 512))


class TestSettingsAndHistory(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_settings_roundtrip_never_saves_key(self):
        s = ps.StudioSettings(engine=ps.ENGINE_CLAUDE, text_model='qwen2.5:3b', claude_api_key='sk-secret')
        ps.save_settings(self.root, s)
        with open(f'{self.root}/{ps.SETTINGS_FILE}', encoding='utf-8') as fh:
            self.assertNotIn('sk-secret', fh.read())
        loaded = ps.load_settings(self.root)
        self.assertEqual((loaded.engine, loaded.text_model, loaded.claude_api_key),
                         (ps.ENGINE_CLAUDE, 'qwen2.5:3b', ''))

    def test_bad_settings_file(self):
        with open(f'{self.root}/{ps.SETTINGS_FILE}', 'w', encoding='utf-8') as fh:
            fh.write('{"engine": "Telepathy", "text_model": 5, "junk": 1}')
        loaded = ps.load_settings(self.root)
        self.assertEqual((loaded.engine, loaded.text_model), (ps.ENGINE_OLLAMA, ps.DEFAULT_TEXT_MODEL))

    def test_history_newest_first_and_capped(self):
        for i in range(ps.HISTORY_LIMIT + 3):
            last = ps.add_history(self.root, {'idea': f'idea {i}', 'positive': f'p{i}'})
        entries = ps.load_history(self.root)
        self.assertEqual(len(entries), ps.HISTORY_LIMIT)
        self.assertEqual(entries[0]['id'], last['id'])
        self.assertEqual(ps.get_history(self.root, last['id'])['positive'], f'p{ps.HISTORY_LIMIT + 2}')
        self.assertIn('idea', ps.history_label(last))


class TestOllamaEngine(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)
        self.settings = ps.StudioSettings(ollama_url=self.fake.url)

    def test_idea_only(self):
        result = ps.build_prompt('a cat at a lighthouse', None, ps.DEFAULT_REFERENCE_MODE, self.settings,
                                 legal_style_names=STYLES, aspect_labels=LABELS)
        self.assertTrue(result.positive.startswith('a ginger cat'))
        self.assertEqual(result.negative, 'cartoon, illustration')
        self.assertEqual(result.styles, ['SAI Cinematic', 'Fooocus Enhance'])
        self.assertEqual(result.aspect, LABELS[3])
        self.assertIsNone(result.ip_type)  # no image, nothing to send as Image Prompt
        self.assertEqual(len(self.fake.requests), 1)
        path, body = self.fake.requests[0]
        self.assertEqual(path, '/api/chat')
        self.assertEqual(body['format'], ps.OUTPUT_SCHEMA)
        self.assertEqual(body['keep_alive'], ps.DEFAULT_KEEP_ALIVE)

    def test_reference_image_is_described_then_used(self):
        image = np.full((64, 48, 3), 128, dtype=np.uint8)
        result = ps.build_prompt('', image, 'Pose and outlines (PyraCanny)', self.settings,
                                 legal_style_names=STYLES, aspect_labels=LABELS)
        self.assertEqual(result.ip_type, 'PyraCanny')
        self.assertIn('railing', result.description)
        (_, describe), (_, compose) = self.fake.requests
        self.assertEqual(len(describe['messages'][0]['images']), 1)
        self.assertIn('railing', compose['messages'][1]['content'])
        self.assertIn('outlines', compose['messages'][1]['content'])

    def test_flux_has_no_negative(self):
        result = ps.build_prompt('a cat', None, ps.REF_DESCRIBE_ONLY, self.settings, style='flux')
        self.assertEqual(result.negative, '')

    def test_missing_model_says_how_to_pull(self):
        settings = ps.StudioSettings(ollama_url=self.fake.url, text_model='llama9:70b')
        with self.assertRaises(ps.StudioError) as ctx:
            ps.build_prompt('a cat', None, ps.REF_DESCRIBE_ONLY, settings)
        self.assertIn('ollama pull llama9:70b', str(ctx.exception))

    def test_status(self):
        self.assertIn('running', ps.ollama_status(self.settings))
        self.assertIn('ollama pull qwen2.5vl:3b',
                      ps.ollama_status(ps.StudioSettings(ollama_url=self.fake.url, vision_model='qwen2.5vl:3b')))

    def test_release_unloads_used_models(self):
        ps.build_prompt('a cat', None, ps.REF_DESCRIBE_ONLY, self.settings)
        self.assertTrue(ps.release(wait=True))
        self.assertEqual(self.fake.requests[-1], ('/api/generate', {'model': 'gemma3:4b', 'keep_alive': 0}))
        self.assertFalse(ps.release(wait=True))  # nothing left to unload

    def test_background_release_can_be_waited_for(self):
        ps.build_prompt('a cat', None, ps.REF_DESCRIBE_ONLY, self.settings)
        self.assertTrue(ps.release())  # background, as when Generate is clicked
        self.assertTrue(ps.wait_for_release(timeout=10))
        self.assertEqual(self.fake.requests[-1], ('/api/generate', {'model': 'gemma3:4b', 'keep_alive': 0}))

    def test_wait_for_release_with_nothing_pending(self):
        self.assertTrue(ps.wait_for_release(timeout=0.1))

    def test_needs_idea_or_image(self):
        with self.assertRaises(ps.StudioError):
            ps.build_prompt('   ', None, ps.REF_DESCRIBE_ONLY, self.settings)


class TestDegradedMode(unittest.TestCase):
    def setUp(self):
        self.settings = ps.StudioSettings(ollama_url=dead_url())

    def test_ollama_down_without_fallbacks(self):
        with self.assertRaises(ps.OllamaUnavailable) as ctx:
            ps.build_prompt('a cat', None, ps.REF_DESCRIBE_ONLY, self.settings)
        self.assertIn("isn't running", str(ctx.exception))
        self.assertIn("isn't running", ps.ollama_status(self.settings))

    def test_ollama_down_uses_builtin_tools(self):
        seen = {}

        def rewrite(text, style):
            seen['text'], seen['style'] = text, style
            return 'a rewritten prompt'

        result = ps.build_prompt('a cat', np.zeros((8, 8, 3), dtype=np.uint8), 'Face (FaceSwap)', self.settings,
                                 style='tags', fallback_rewrite=rewrite, fallback_describe=lambda img: 'a person')
        self.assertEqual(result.positive, 'a rewritten prompt')
        self.assertEqual(seen, {'text': 'a cat, a person', 'style': 'tags'})
        self.assertEqual((result.engine_used, result.ip_type), ('Built-in', 'FaceSwap'))
        self.assertIn("isn't running", result.notes)


class FakeClaude:
    """Stands in for anthropic.Anthropic(): records the request, returns JSON text."""

    def __init__(self, text=None, stop_reason='end_turn'):
        self.calls = []
        reply = text or json.dumps({'positive': 'portrait of a woman in a red coat, snowy street, bokeh',
                                    'negative': 'cartoon', 'styles': ['SAI Cinematic'], 'aspect': 'portrait',
                                    'notes': 'Kept the coat from the reference.'})
        response = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type='text', text=reply)])

        def create(**kwargs):
            self.calls.append(kwargs)
            return response

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


class TestClaudeEngine(unittest.TestCase):
    def setUp(self):
        self.settings = ps.StudioSettings(engine=ps.ENGINE_CLAUDE, ollama_url=dead_url())

    def test_one_request_with_image_and_schema(self):
        client = FakeClaude()
        result = ps.build_prompt('make it winter', np.zeros((32, 32, 3), dtype=np.uint8),
                                 'Overall look (Image Prompt)', self.settings, legal_style_names=STYLES,
                                 aspect_labels=LABELS, claude_client=client)
        self.assertEqual((result.engine_used, result.ip_type), (ps.ENGINE_CLAUDE, 'ImagePrompt'))
        self.assertEqual(result.aspect, LABELS[1])
        self.assertEqual(result.styles, ['SAI Cinematic'])
        (call,) = client.calls
        self.assertEqual(call['model'], 'claude-opus-5-5')
        self.assertEqual(call['output_config']['format']['schema'], ps.OUTPUT_SCHEMA)
        self.assertEqual(call['fallbacks'], 'default')
        content = call['messages'][0]['content']
        self.assertEqual([c['type'] for c in content], ['image', 'text'])
        self.assertIn('attached', content[1]['text'])

    def test_other_models_skip_fallbacks(self):
        client = FakeClaude()
        settings = ps.StudioSettings(engine=ps.ENGINE_CLAUDE, claude_model='claude-haiku-5-5')
        ps.build_prompt('a cat', None, ps.REF_DESCRIBE_ONLY, settings, claude_client=client)
        self.assertNotIn('fallbacks', client.calls[0])

    def test_refusal(self):
        with self.assertRaises(ps.StudioError):
            ps.build_prompt('a cat', None, ps.REF_DESCRIBE_ONLY, self.settings,
                            claude_client=FakeClaude(stop_reason='refusal'))


try:
    import anthropic
    try:
        import httpx2 as httpx  # anthropic 1.x+ is built on httpx2
    except ImportError:
        import httpx
except ImportError:
    anthropic = None


@unittest.skipIf(anthropic is None, 'anthropic SDK not installed (Claude mode is optional)')
class TestClaudeSDK(unittest.TestCase):
    """The real SDK against a mocked HTTP transport: request shape and error mapping."""

    def client(self, handler):
        return anthropic.Anthropic(api_key='test-key', max_retries=0,
                                   http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_request_and_reply(self):
        seen = {}

        def handler(request):
            seen['url'] = str(request.url)
            seen['body'] = json.loads(request.content)
            seen['beta'] = request.headers.get('anthropic-beta', '')
            return httpx.Response(200, json={
                'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'claude-opus-5-5',
                'content': [{'type': 'text', 'text': json.dumps({'positive': 'a fox in snow', 'negative': '',
                                                                 'styles': [], 'aspect': 'square',
                                                                 'notes': ''})}],
                'stop_reason': 'end_turn', 'stop_sequence': None,
                'usage': {'input_tokens': 10, 'output_tokens': 10}})

        settings = ps.StudioSettings(engine=ps.ENGINE_CLAUDE)
        result = ps.build_prompt('a fox', None, ps.REF_DESCRIBE_ONLY, settings, aspect_labels=LABELS,
                                 claude_client=self.client(handler))
        self.assertEqual(result.positive, 'a fox in snow')
        self.assertIn('/v1/messages', seen['url'])
        self.assertIn('server-side-fallback-2026-07-01', seen['beta'])
        self.assertEqual(seen['body']['fallbacks'], 'default')
        self.assertEqual(seen['body']['output_config']['effort'], 'low')

    def test_bad_key_message(self):
        def handler(request):
            return httpx.Response(401, json={'type': 'error', 'error': {'type': 'authentication_error',
                                                                         'message': 'invalid x-api-key'}})

        with self.assertRaises(ps.StudioError) as ctx:
            ps.build_prompt('a fox', None, ps.REF_DESCRIBE_ONLY, ps.StudioSettings(engine=ps.ENGINE_CLAUDE),
                            claude_client=self.client(handler))
        self.assertIn('API key', str(ctx.exception))

    def test_offline_falls_back_to_ollama(self):
        def handler(request):
            raise httpx.ConnectError('no network')

        fake = FakeOllama()
        self.addCleanup(fake.close)
        result = ps.build_prompt('a fox', None, ps.REF_DESCRIBE_ONLY,
                                 ps.StudioSettings(engine=ps.ENGINE_CLAUDE, ollama_url=fake.url),
                                 claude_client=self.client(handler))
        self.assertEqual(result.engine_used, ps.ENGINE_OLLAMA)
        self.assertIn("Couldn't reach Claude", result.notes)


if __name__ == '__main__':
    unittest.main()
