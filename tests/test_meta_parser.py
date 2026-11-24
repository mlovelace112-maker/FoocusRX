import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import io
import json
import unittest
from unittest import mock

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import modules.config
from modules.flags import MetadataScheme, SAMPLERS
from modules.meta_parser import (
    A1111MetadataParser,
    FooocusMetadataParser,
    get_exif,
    get_metadata_parser,
    load_parameter_button_click,
    read_info_from_image,
)

UPDATE = {'__type__': 'update'}  # gr.update(): "leave this UI field unchanged"

MODEL = 'juggernautXL_v8Rundiffusion.safetensors'
LORA = 'sd_xl_offset_example-lora_1.0.safetensors'
LIGHTNING_LORA = 'sdxl_lightning_4step_lora.safetensors'

# Positions in load_parameter_button_click's result list. The order mirrors
# the call sequence in that function and webui.py's load_data_outputs.
I_PROMPT, I_NEGATIVE, I_STYLES, I_PERFORMANCE, I_STEPS = 2, 3, 4, 5, 6
I_ASPECT, I_WIDTH, I_HEIGHT = 8, 9, 10
I_GUIDANCE, I_SHARPNESS = 11, 12
I_ADM = 13  # three values
I_BASE_MODEL, I_SAMPLER, I_SCHEDULER = 19, 22, 23
I_SEED_RANDOM, I_SEED = 25, 26
I_INPAINT_METHOD = 29
I_FREEU = I_INPAINT_METHOD + 1 + modules.config.default_enhance_tabs + 2  # enabled + 4 values
I_LORAS = I_FREEU + 5  # (enabled, name, weight) per slot

MALICIOUS = "__import__('os').system('echo pwned')"


def fooocus_metadata(**overrides):
    """A fresh dict shaped like FooocusMetadataParser.to_string output."""
    data = {
        'prompt': 'a cat in a hat',
        'negative_prompt': 'blurry',
        'styles': "['Fooocus V2', 'Fooocus Enhance']",
        'performance': 'Speed',
        'steps': 30,
        'resolution': '(1024, 1024)',
        'guidance_scale': 4.0,
        'sharpness': 2.0,
        'adm_guidance': '(1.5, 0.8, 0.3)',
        'base_model': 'juggernautXL_v8Rundiffusion',
        'refiner_model': 'None',
        'sampler': 'dpmpp_2m_sde_gpu',
        'scheduler': 'karras',
        'seed': '12345',
        'lora_combined_1': 'sd_xl_offset_example-lora_1.0 : 0.1',
        'version': 'Fooocus v2.5.5',
    }
    data.update(overrides)
    return data


def reopen(image: Image.Image, fmt: str, **save_kwargs) -> Image.Image:
    """Save to bytes and reopen, as if the user dragged the file in."""
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    return Image.open(buf)


def png_with(parameters: str | None = None, scheme: str | None = None) -> Image.Image:
    info = PngInfo()
    if parameters is not None:
        info.add_text('parameters', parameters)
    if scheme is not None:
        info.add_text('fooocus_scheme', scheme)
    return reopen(Image.new('RGB', (8, 8)), 'PNG', pnginfo=info)


class ConfigPatched(unittest.TestCase):
    """Pin the installed-model lists so results don't depend on disk contents."""

    def setUp(self):
        for name, value in [
            ('model_filenames', [MODEL]),
            ('lora_filenames', [LORA, LIGHTNING_LORA]),
            ('vae_filenames', []),
        ]:
            patcher = mock.patch.object(modules.config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class TestReadInfoFromImage(unittest.TestCase):
    def test_image_without_metadata(self):
        self.assertEqual(read_info_from_image(png_with()), (None, None))

    def test_png_fooocus_scheme(self):
        params, scheme = read_info_from_image(
            png_with(json.dumps(fooocus_metadata()), MetadataScheme.FOOOCUS.value))
        self.assertEqual(scheme, MetadataScheme.FOOOCUS)
        self.assertEqual(params['prompt'], 'a cat in a hat')

    def test_png_a1111_scheme(self):
        text = 'a cat\nSteps: 30, Sampler: Euler, Seed: 1'
        params, scheme = read_info_from_image(png_with(text, MetadataScheme.A1111.value))
        self.assertEqual(scheme, MetadataScheme.A1111)
        self.assertEqual(params, text)

    def test_unknown_scheme_falls_back_by_payload_type(self):
        _, scheme = read_info_from_image(png_with(json.dumps({'prompt': 'x'}), 'bogus'))
        self.assertEqual(scheme, MetadataScheme.FOOOCUS)
        _, scheme = read_info_from_image(png_with('a cat\nSteps: 30, Seed: 1, CFG scale: 4', None))
        self.assertEqual(scheme, MetadataScheme.A1111)

    def _exif_round_trip(self, fmt, prompt):
        payload = json.dumps(fooocus_metadata(prompt=prompt))
        image = reopen(Image.new('RGB', (8, 8)), fmt,
                       exif=get_exif(payload, MetadataScheme.FOOOCUS.value))
        params, scheme = read_info_from_image(image)
        self.assertEqual(scheme, MetadataScheme.FOOOCUS)
        self.assertEqual(params['prompt'], prompt)

    def test_jpeg_exif_round_trip(self):
        self._exif_round_trip('JPEG', 'a cat in a hat')

    def test_webp_exif_round_trip(self):
        self._exif_round_trip('WEBP', 'a cat in a hat')

    def test_jpeg_exif_round_trip_non_ascii(self):
        self._exif_round_trip('JPEG', 'café 🌸')

    def test_webp_exif_round_trip_non_ascii(self):
        self._exif_round_trip('WEBP', 'café 🌸')


class TestFooocusToJson(ConfigPatched):
    def test_model_and_lora_names_get_installed_filenames(self):
        data = FooocusMetadataParser().to_json(fooocus_metadata())
        self.assertEqual(data['base_model'], MODEL)
        self.assertEqual(data['lora_combined_1'], f'{LORA} : 0.1')

    def test_missing_model_becomes_none(self):
        data = FooocusMetadataParser().to_json(fooocus_metadata(base_model='not_installed'))
        self.assertIsNone(data['base_model'])


class TestA1111ToJson(ConfigPatched):
    def parse(self, text):
        return A1111MetadataParser().to_json(text)

    def test_fooocus_generated_a1111_text(self):
        text = (
            'a cat in a hat, highly detailed\n'
            'Negative prompt: blurry\n'
            'Steps: 30, Sampler: DPM++ 2M SDE Karras, Seed: 12345, Size: 1024x1024, CFG scale: 4, '
            'Model: juggernautXL_v8Rundiffusion, Performance: Speed, '
            'Raw prompt: a cat in a hat, Raw negative prompt: blurry, '
            'Lora weights: "sd_xl_offset_example-lora_1.0: 0.1", Version: Fooocus v2.5.5'
        )
        data = self.parse(text)
        self.assertEqual(data['prompt'], 'a cat in a hat')
        self.assertEqual(data['negative_prompt'], 'blurry')
        self.assertEqual(data['seed'], '12345')
        self.assertEqual(data['resolution'], str(('1024', '1024')))
        self.assertEqual(data['base_model'], MODEL)
        self.assertEqual(data['lora_combined_1'], f'{LORA} : 0.1')
        self.assertEqual(SAMPLERS[data['sampler']], 'DPM++ 2M SDE')  # " Karras" stripped, mapped to key

    def test_multiline_negative_prompt(self):
        data = self.parse('a cat\nNegative prompt: blurry\nugly\nSteps: 30, Seed: 1, CFG scale: 4, Raw prompt: a cat')
        self.assertEqual(data['negative_prompt'], 'blurry\nugly')

    def test_unknown_keys_are_ignored(self):
        data = self.parse('a cat\nSteps: 30, Seed: 1, Denoising strength: 0.5, Raw prompt: a cat')
        self.assertEqual(data['seed'], '1')
        self.assertNotIn('Denoising strength', data)

    def test_plain_a1111_image_keeps_its_prompt(self):
        # Regression: with no style match and no "Raw prompt" field, the
        # extracted prompt used to be '', clearing the prompt box.
        data = self.parse('a cat, masterpiece\nNegative prompt: blurry\nSteps: 30, Seed: 1, CFG scale: 4')
        self.assertEqual(data['prompt'], 'a cat, masterpiece')
        self.assertEqual(data['negative_prompt'], 'blurry')

    def test_prompt_expansion_repeat_still_detected(self):
        # The pre-existing path: Fooocus V2 expansion repeats the first word.
        data = self.parse('a cat, a cat sitting, sunlit\nSteps: 30, Seed: 1, CFG scale: 4')
        self.assertEqual(data['prompt'], 'a cat')

    def test_performance_inferred_from_preset_steps(self):
        # Regression: `'performance' in data is None` was always False.
        data = self.parse('a cat\nSteps: 30, Seed: 1, CFG scale: 4, Raw prompt: a cat')
        self.assertEqual(data.get('performance'), 'Speed')
        data = self.parse('a cat\nSteps: 4, Seed: 1, CFG scale: 4, Raw prompt: a cat')
        self.assertEqual(data.get('performance'), 'Lightning')

    def test_explicit_performance_is_not_overridden(self):
        data = self.parse('a cat\nSteps: 30, Seed: 1, CFG scale: 4, Performance: Quality, Raw prompt: a cat')
        self.assertEqual(data['performance'], 'Quality')

    def test_non_preset_steps_do_not_crash_or_set_performance(self):
        data = self.parse('a cat\nSteps: 37, Seed: 1, CFG scale: 4, Raw prompt: a cat')
        self.assertEqual(data['steps'], '37')
        self.assertNotIn('performance', data)

    def test_a1111_string_round_trip(self):
        parser = A1111MetadataParser()
        parser.raw_prompt, parser.full_prompt = 'a cat', ['a cat', 'highly detailed']
        parser.raw_negative_prompt, parser.full_negative_prompt = 'blurry', ['blurry']
        parser.steps, parser.base_model_hash = 30, 'abc123'
        metadata = [(k, k, v) for k, v in fooocus_metadata(vae='Default (model)').items()]
        data = A1111MetadataParser().to_json(parser.to_string(metadata))
        self.assertEqual(data['prompt'], 'a cat')
        self.assertEqual(data['negative_prompt'], 'blurry')
        self.assertEqual(data['seed'], '12345')
        self.assertEqual(data['base_model'], MODEL)


class TestLoadParameterButtonClick(ConfigPatched):
    def load(self, data):
        return load_parameter_button_click(data, False, modules.flags.inpaint_option_default)

    def test_result_length_is_constant(self):
        # Gradio needs one value per output component; a length change breaks the UI.
        garbage = {k: None for k in fooocus_metadata()} | {'resolution': 7, 'styles': 3}
        lengths = {len(self.load(d)) for d in ({}, fooocus_metadata(), garbage)}
        self.assertEqual(len(lengths), 1, lengths)

    def test_accepts_json_string(self):
        self.assertEqual(self.load(json.dumps(fooocus_metadata())), self.load(fooocus_metadata()))

    def test_loads_core_fields(self):
        r = self.load(fooocus_metadata())
        self.assertTrue(r[0])
        self.assertEqual(r[I_PROMPT], 'a cat in a hat')
        self.assertEqual(r[I_NEGATIVE], 'blurry')
        self.assertEqual(r[I_STYLES], ['Fooocus V2', 'Fooocus Enhance'])
        self.assertEqual(r[I_PERFORMANCE], 'Speed')
        self.assertEqual(r[I_ASPECT], modules.config.add_ratio('1024*1024'))
        self.assertEqual(r[I_GUIDANCE], 4.0)
        self.assertEqual(r[I_ADM:I_ADM + 3], [1.5, 0.8, 0.3])
        self.assertEqual(r[I_SAMPLER], 'dpmpp_2m_sde_gpu')
        self.assertEqual((r[I_SEED_RANDOM], r[I_SEED]), (False, 12345))

    def test_custom_resolution_goes_to_width_and_height(self):
        r = self.load(fooocus_metadata(resolution='(1000, 1234)'))
        self.assertEqual(r[I_ASPECT:I_HEIGHT + 1], [UPDATE, 1000, 1234])

    def test_steps_matching_performance_use_preset(self):
        self.assertEqual(self.load(fooocus_metadata(steps=30))[I_STEPS], -1)
        self.assertEqual(self.load(fooocus_metadata(steps=25))[I_STEPS], 25)

    def test_image_number_is_clamped(self):
        r = self.load(fooocus_metadata(image_number=10_000))
        self.assertEqual(r[1], modules.config.default_max_image_number)

    def test_freeu(self):
        r = self.load(fooocus_metadata(freeu='(1.01, 1.02, 0.99, 0.95)'))
        self.assertEqual(r[I_FREEU:I_FREEU + 5], [True, 1.01, 1.02, 0.99, 0.95])
        self.assertFalse(self.load(fooocus_metadata())[I_FREEU])

    def test_code_in_metadata_is_never_evaluated(self):
        data = fooocus_metadata(styles=MALICIOUS, resolution=MALICIOUS,
                                adm_guidance=MALICIOUS, freeu=MALICIOUS)
        with mock.patch('os.system') as system:
            r = self.load(data)
        system.assert_not_called()
        self.assertEqual(r[I_STYLES], UPDATE)
        self.assertEqual(r[I_ASPECT:I_HEIGHT + 1], [UPDATE] * 3)
        self.assertEqual(r[I_ADM:I_ADM + 3], [UPDATE] * 3)
        self.assertEqual(r[I_FREEU:I_FREEU + 5], [False] + [UPDATE] * 4)

    def test_performance_lora_is_not_loaded_twice(self):
        r = self.load(fooocus_metadata(
            performance='Lightning',
            lora_combined_1=f'{LIGHTNING_LORA} : 1.0',
            lora_combined_2=f'{LORA} : 0.5',
        ))
        self.assertEqual(r[I_LORAS:I_LORAS + 3], [True, 'None', 1])
        self.assertEqual(r[I_LORAS + 3:I_LORAS + 6], [True, LORA, 0.5])


class TestImageToUiEndToEnd(ConfigPatched):
    """Image file -> read_info_from_image -> parser.to_json -> UI values."""

    def run_pipeline(self, image):
        params, scheme = read_info_from_image(image)
        data = get_metadata_parser(scheme).to_json(params)
        return load_parameter_button_click(data, False, modules.flags.inpaint_option_default)

    def assert_loaded(self, r):
        self.assertEqual(r[I_PROMPT], 'a cat in a hat')
        self.assertEqual(r[I_SEED], 12345)
        self.assertEqual(r[I_ASPECT], modules.config.add_ratio('1024*1024'))
        self.assertEqual(r[I_BASE_MODEL], MODEL)

    def test_png_fooocus(self):
        self.assert_loaded(self.run_pipeline(
            png_with(json.dumps(fooocus_metadata()), MetadataScheme.FOOOCUS.value)))

    def test_jpeg_fooocus(self):
        payload = json.dumps(fooocus_metadata())
        self.assert_loaded(self.run_pipeline(reopen(
            Image.new('RGB', (8, 8)), 'JPEG', exif=get_exif(payload, MetadataScheme.FOOOCUS.value))))

    def test_png_a1111(self):
        text = ('a cat in a hat\nNegative prompt: blurry\n'
                'Steps: 30, Sampler: Euler, Seed: 12345, Size: 1024x1024, CFG scale: 4, '
                'Model: juggernautXL_v8Rundiffusion, Raw prompt: a cat in a hat')
        self.assert_loaded(self.run_pipeline(png_with(text, MetadataScheme.A1111.value)))


if __name__ == '__main__':
    unittest.main()
