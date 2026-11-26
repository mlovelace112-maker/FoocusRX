import sys

sys.argv = sys.argv[:1]  # args_manager parses sys.argv at import time

import json
import os
import struct
import tempfile
import unittest

from modules import lora_family as lf
from modules.sdcpp import app as sdcpp_app

KREA = ['transformer.final_layer.linear.lora_A.weight',
        'transformer.text_fusion.layerwise_blocks.0.attn.to_q.lora_B.weight']
KREA_BLOCKS_ONLY = ['transformer.transformer_blocks.3.attn.to_gate.lora_A.weight',
                    'transformer.transformer_blocks.3.ff.up.lora_B.weight']
FLUX_DIFFUSERS_DOUBLE = ['transformer.transformer_blocks.0.attn.add_k_proj.lora_A.weight',
                         'transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight']
MOSS = ['diffusion_model.layers.3.proj.q.lora_A.weight', 'diffusion_model.layers.3.proj.ffn_down.lora_B.weight']
MOSS_BARE = ['layers.0.proj.k.lora_A', 'layers.0.proj.o.lora_B']
FLUX_KOHYA = ['lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight', 'lora_unet_double_blocks_0_img_attn_qkv.alpha']
FLUX_DIFFUSERS = ['transformer.single_transformer_blocks.3.attn.to_q.lora_A.weight']
FLUX_BFL = ['diffusion_model.double_blocks.1.img_mlp.0.lora_B.weight']
SDXL = ['lora_unet_input_blocks_4_1_proj_in.lora_down.weight', 'lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha']
SDXL_DIFFUSERS = ['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight']


def write_safetensors(path, names):
    header = {n: {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]} for n in names}
    data = json.dumps(header).encode()
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(data)) + data + b'\0\0')


class TestFamilyFromKeys(unittest.TestCase):
    def test_families(self):
        self.assertEqual(lf.family_from_keys(KREA), 'Krea 2')
        self.assertEqual(lf.family_from_keys(KREA_BLOCKS_ONLY), 'Krea 2')
        self.assertEqual(lf.family_from_keys(FLUX_DIFFUSERS_DOUBLE), 'FLUX.1')
        self.assertEqual(lf.family_from_keys(MOSS), 'MOSS-Image')
        self.assertEqual(lf.family_from_keys(MOSS_BARE), 'MOSS-Image')
        for keys in (FLUX_KOHYA, FLUX_DIFFUSERS, FLUX_BFL):
            self.assertEqual(lf.family_from_keys(keys), 'FLUX.1', keys)
        for keys in (SDXL, SDXL_DIFFUSERS, [], ['something.unknown']):
            self.assertEqual(lf.family_from_keys(keys), 'SDXL', keys)


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        for name, keys in (('krea_style.safetensors', KREA), ('moss_anime.safetensors', MOSS),
                           ('offset.safetensors', SDXL), ('flux_style.safetensors', FLUX_KOHYA)):
            write_safetensors(os.path.join(self.dir, name), keys)
        open(os.path.join(self.dir, 'old.ckpt'), 'wb').close()

    def tearDown(self):
        self.tmp.cleanup()

    def resolve(self, name):
        return os.path.join(self.dir, name)

    def test_loras_for_family(self):
        names = ['krea_style.safetensors', 'moss_anime.safetensors', 'offset.safetensors',
                 'flux_style.safetensors', 'old.ckpt']
        self.assertEqual(lf.loras_for('Krea 2', names, self.resolve), ['krea_style.safetensors'])
        self.assertEqual(lf.loras_for('FLUX.1', names, self.resolve), ['flux_style.safetensors'])
        # Unreadable / non-safetensors files stay listed for SDXL.
        self.assertEqual(lf.loras_for('SDXL', names, self.resolve), ['offset.safetensors', 'old.ckpt'])

    def test_carry_over_is_filtered(self):
        """The default preset's SDXL offset LoRA must not reach Krea 2."""
        usable, skipped = lf.split_by_family([('offset.safetensors', 0.1), ('krea_style.safetensors', 0.8)],
                                             'Krea 2', self.resolve)
        self.assertEqual(usable, [('krea_style.safetensors', 0.8)])
        self.assertEqual(skipped, [('offset.safetensors', 0.1)])

    def test_stage_loras_uses_safe_names(self):
        with tempfile.TemporaryDirectory() as out:
            lora_dir, tags = sdcpp_app.stage_loras([('krea_style.safetensors', 0.8), ('moss_anime.safetensors', 1.0)],
                                                   os.path.join(out, 'loras'), self.resolve)
            self.assertEqual(tags, '<lora:l0:0.8><lora:l1:1>')
            self.assertEqual(sorted(os.listdir(lora_dir)), ['l0.safetensors', 'l1.safetensors'])
            self.assertEqual(os.path.realpath(os.path.join(lora_dir, 'l0.safetensors')),
                             os.path.realpath(self.resolve('krea_style.safetensors')))
        self.assertEqual(sdcpp_app.stage_loras([], '/nonexistent', self.resolve), (None, ''))


class TestPromptLoras(unittest.TestCase):
    def test_extract_keeps_line_breaks_and_json(self):
        prompt = 'A sign that reads:\n"OPEN\nlate" <lora:krea_style:0.7>'
        text, found = sdcpp_app.extract_prompt_loras(prompt, ['sub/krea_style.safetensors'])
        self.assertEqual(text, 'A sign that reads:\n"OPEN\nlate"')
        self.assertEqual(found, [('sub/krea_style.safetensors', 0.7)])
        layout = '{"summary": {"caption": "a, b"}, "items": []}'
        text, found = sdcpp_app.extract_prompt_loras(layout + '<lora:missing:1>', [])
        self.assertEqual((text, found), (layout, []))
        self.assertTrue(sdcpp_app.is_json_prompt(text))

    def test_merge(self):
        merged = sdcpp_app.merge_loras([('a.safetensors', 0.5), ('None', 1.0), ('z.safetensors', 0)],
                                       [('a.safetensors', 0.9), ('b.safetensors', 1.0)], limit=5)
        self.assertEqual(merged, [('a.safetensors', 0.5), ('b.safetensors', 1.0)])
        self.assertEqual(len(sdcpp_app.merge_loras([(f'{i}', 1) for i in range(9)], [], limit=5)), 5)


if __name__ == '__main__':
    unittest.main()
