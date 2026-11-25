import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import json
import os
import shutil
import struct
import tempfile
import unittest

from modules import model_profiles as mp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = mp.load_presets(os.path.join(REPO, 'presets'))
SDXL_TENSORS = ['conditioner.embedders.1.model.ln_final.weight', 'model.diffusion_model.out.0.weight']


def add_ratio(x):
    w, h = x.replace('*', ' ').split()[:2]
    return f'{w}×{h} <span>r</span>'


class TempModels(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def model(self, name, meta=None, tensors=SDXL_TENSORS):
        """A tiny .safetensors with only a header (no weights needed)."""
        header = {t: {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]} for t in tensors}
        if meta is not None:
            header['__metadata__'] = meta
        raw = json.dumps(header).encode()
        path = os.path.join(self.dir, name)
        with open(path, 'wb') as fh:
            fh.write(struct.pack('<Q', len(raw)) + raw + b'\0\0')
        return path

    def profile(self, name, meta=None, tensors=SDXL_TENSORS):
        return mp.choose_profile(name, self.model(name, meta, tensors), PRESETS)


class TestChooseProfile(TempModels):
    def test_exact_preset_model(self):
        p = self.profile('ponyDiffusionV6XL.safetensors')
        self.assertEqual(p.name, 'pony_v6')
        self.assertEqual(p.settings['guidance_scale'], 7.0)
        self.assertEqual(p.settings['vae_name'], 'ponyDiffusionV6XL_vae.safetensors')

    def test_juggernaut_uses_default_not_lcm_speed_preset(self):
        self.assertEqual(self.profile('juggernautXL_v8Rundiffusion.safetensors').name, 'default')

    def test_family_from_file_name(self):
        cases = {
            'autismmixSDXL_ponyFaetality.safetensors': 'pony_v6',
            'illustriousXL_v01.safetensors': 'anime',
            'RealVisXL_V4.0.safetensors': 'realistic',
            'playground-v2.5-other.safetensors': 'playground_v2.5',
            'dreamshaperXL_lightningDPMSDE.safetensors': 'distilled',
            'myMix_Hyper-SD_8step.safetensors': 'distilled',
        }
        for name, expected in cases.items():
            self.assertEqual(self.profile(name).name, expected, name)

    def test_no_false_positives_for_similar_words(self):
        for name in ('hyperRealismXL.safetensors', 'animalsXL.safetensors', 'photonXL.safetensors'):
            self.assertEqual(self.profile(name).name, 'default', name)

    def test_family_from_embedded_title_and_merge_sources(self):
        p = self.profile('merge_0001.safetensors', {'modelspec.architecture': mp.SDXL_ARCHITECTURE,
                                                    'modelspec.title': 'My Pony Merge'})
        self.assertEqual(p.name, 'pony_v6')
        workflow = json.dumps({'4': {'inputs': {'ckpt_name': 'noobaiXL_v1.safetensors'}}})
        p = self.profile('merge_0002.safetensors', {'modelspec.architecture': mp.SDXL_ARCHITECTURE,
                                                    'prompt': workflow})
        self.assertEqual(p.name, 'anime')

    def test_comfy_ui_layout_is_ignored_only_executed_graph_counts(self):
        sdxl = {'modelspec.architecture': mp.SDXL_ARCHITECTURE}
        prompt = json.dumps({'4': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'mixA.safetensors'}}})
        layout = json.dumps({'nodes': [{'widgets_values': ['dmd2_sdxl_4step_lora.safetensors']}]})
        p = self.profile('merge_a.safetensors', {**sdxl, 'prompt': prompt, 'workflow': layout})
        self.assertEqual(p.name, 'default')
        # A distillation LoRA actually merged in does count...
        merged = json.dumps({'5': {'inputs': {'lora_name': 'dmd2_sdxl_4step_lora.safetensors', 'strength_model': 1.0}}})
        self.assertEqual(self.profile('merge_b.safetensors', {**sdxl, 'prompt': merged}).name, 'distilled')
        # ...but not one loaded at strength 0.
        off = json.dumps({'5': {'inputs': {'lora_name': 'dmd2_sdxl_4step_lora.safetensors', 'strength_model': 0}}})
        self.assertEqual(self.profile('merge_c.safetensors', {**sdxl, 'prompt': off}).name, 'default')

    def test_distilled_settings_match_app_lightning(self):
        s = self.profile('x_lightning.safetensors').settings
        self.assertEqual((s['sampler_name'], s['scheduler_name'], s['overwrite_step'], s['guidance_scale']),
                         ('euler', 'sgm_uniform', 8, 1.0))
        self.assertEqual(s['performance'], 'Speed')  # not Lightning: that would add the LoRA again

    def test_non_sdxl_is_left_alone(self):
        p = self.profile('sd15.safetensors', {'modelspec.architecture': 'stable-diffusion-v1'})
        self.assertFalse(p.supported)
        self.assertEqual(p.settings, {})
        p = self.profile('sd15b.safetensors', tensors=['model.diffusion_model.input_blocks.0.0.weight'])
        self.assertFalse(p.supported)

    def test_unknown_or_unreadable_falls_back_to_default(self):
        self.assertEqual(self.profile('mystery.safetensors').name, 'default')
        self.assertEqual(mp.choose_profile('old.ckpt', '/nope/old.ckpt', PRESETS).name, 'default')


class TestReadHeader(TempModels):
    def test_reads_metadata_and_names_only(self):
        meta, tensors = mp.read_header(self.model('a.safetensors', {'modelspec.title': 't'}))
        self.assertEqual(meta, {'modelspec.title': 't'})
        self.assertEqual(sorted(tensors), sorted(SDXL_TENSORS))

    def test_garbage(self):
        path = os.path.join(self.dir, 'bad.safetensors')
        with open(path, 'wb') as fh:
            fh.write(b'\xff' * 16)
        self.assertEqual(mp.read_header(path), ({}, []))


class TestResolveAndDescribe(unittest.TestCase):
    labels = [add_ratio('896*1152'), add_ratio('1152*896')]

    def resolve(self, settings, vaes=()):
        return mp.resolve_for_ui(settings, vae_filenames=list(vaes), aspect_ratio_labels=self.labels,
                                 add_ratio=add_ratio, default_vae='Default (model)')

    def test_aspect_ratio_and_vae(self):
        out = self.resolve({'aspect_ratio': '896*1152', 'vae_name': 'pony_vae.safetensors'})
        self.assertEqual(out['aspect_ratio'], add_ratio('896*1152'))
        self.assertEqual(out['vae_name'], 'Default (model)')  # not installed
        out = self.resolve({'vae_name': 'pony_vae.safetensors'}, vaes=['pony_vae.safetensors'])
        self.assertEqual(out['vae_name'], 'pony_vae.safetensors')
        self.assertNotIn('aspect_ratio', self.resolve({'aspect_ratio': '640*1536'}))

    def test_describe_changes(self):
        lines = mp.describe_changes(
            {'guidance_scale': 4.0, 'sampler_name': 'dpmpp_2m_sde_gpu', 'sharpness': 2,
             'aspect_ratio': add_ratio('1152*896')},
            {'guidance_scale': 7.0, 'sampler_name': 'dpmpp_2m_sde_gpu', 'sharpness': 2.0,
             'aspect_ratio': add_ratio('896*1152')})
        self.assertEqual(lines, ['Guidance scale: 4 → 7', 'Aspect ratio: 1152×896 → 896×1152'])


if __name__ == '__main__':
    unittest.main()
