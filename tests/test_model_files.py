import sys

sys.argv = sys.argv[:1]  # args_manager parses sys.argv at import time

import json
import os
import stat
import struct
import tempfile
import unittest
from unittest import mock

from modules import lora_family as lf
from modules import model_files as mf
from modules import ui_family
from modules.sdcpp import app, registry, runner
from modules.sdcpp.registry import KREA2, MOSS_IMAGE

# Tensor names as real files spell them.
KREA_LORA_COMFY = ['diffusion_model.blocks.0.attn.wq.lora_A.weight', 'diffusion_model.blocks.0.attn.wq.lora_B.weight',
                   'diffusion_model.blocks.0.mlp.gate.lora_A.weight']
KREA_LORA_KOHYA = ['lora_unet_blocks_0_attn_wq.lora_down.weight', 'lora_unet_blocks_0_attn_wq.alpha',
                   'lora_unet_txtfusion_layerwise_blocks_0_attn_wk.lora_up.weight']
KREA_LORA_DIFFUSERS = ['transformer.transformer_blocks.3.attn.to_gate.lora_A.weight']
MOSS_LORA_KOHYA = ['lora_unet_layers_0_proj_q.lora_down.weight', 'lora_unet_layers_0_proj_ffn_down.lora_up.weight']
MOSS_LORA_PEFT = ['base_model.model.layers.3.proj.ffn_up_gate.lora_A.weight']
SDXL_LORA = ['lora_unet_input_blocks_4_1_proj_in.lora_down.weight', 'lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha']
QWEN_IMAGE_LORA = ['transformer_blocks.0.attn.add_q_proj.lora_A.weight', 'transformer_blocks.0.img_mod.1.lora_A.weight']
ZIMAGE_LORA = ['lora_unet_layers_0_attention_qkv.lora_down.weight']

KREA_MODEL = ['model.diffusion_model.first.weight', 'model.diffusion_model.blocks.0.attn.wq.weight',
              'model.diffusion_model.txtfusion.projector.weight', 'model.diffusion_model.last.linear.weight']
KREA_MODEL_GGUF = ['first.weight', 'blocks.0.attn.wq.weight', 'blocks.0.mlp.gate.weight', 'txtfusion.projector.weight']
MOSS_MODEL = ['layers.0.proj.q.weight', 'layers.0.proj.k.weight', 'layers.0.proj.ffn_down.weight']
SDXL_CHECKPOINT = ['model.diffusion_model.input_blocks.0.0.weight',
                   'conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight',
                   'first_stage_model.decoder.conv_in.weight']
VAE = ['decoder.conv_in.weight', 'encoder.conv_in.weight', 'post_quant_conv.weight']
TEXT_ENCODER = ['model.layers.0.self_attn.q_proj.weight', 'model.embed_tokens.weight']


def write_safetensors(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = {n: {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]} for n in names}
    data = json.dumps(header).encode()
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(data)) + data + b'\0\0')


def _gguf_string(text):
    raw = text.encode()
    return struct.pack('<Q', len(raw)) + raw


def write_gguf(path, names, architecture=None, tokens=0):
    """A minimal GGUF v3 file: metadata (architecture, an optional string-array
    tokenizer, a scalar array, a float) and tensor infos, no data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kvs = []
    if architecture:
        kvs.append(_gguf_string('general.architecture') + struct.pack('<I', 8) + _gguf_string(architecture))
    kvs.append(_gguf_string('general.alignment') + struct.pack('<I', 4) + struct.pack('<I', 32))
    kvs.append(_gguf_string('some.scale') + struct.pack('<I', 6) + struct.pack('<f', 1.5))
    kvs.append(_gguf_string('some.ints') + struct.pack('<IIQ', 9, 5, 3) + struct.pack('<3i', 1, 2, 3))
    if tokens:
        kvs.append(_gguf_string('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, tokens)
                   + b''.join(_gguf_string(f'tok{i}') for i in range(tokens)))
    tensors = b''.join(_gguf_string(n) + struct.pack('<I', 2) + struct.pack('<QQ', 4, 4)
                       + struct.pack('<I', 0) + struct.pack('<Q', 0) for n in names)
    with open(path, 'wb') as fh:
        fh.write(b'GGUF' + struct.pack('<I', 3) + struct.pack('<QQ', len(names), len(kvs)) + b''.join(kvs) + tensors)


class TestClassify(unittest.TestCase):
    def test_loras_in_every_spelling(self):
        for names, family in ((KREA_LORA_COMFY, mf.KREA2), (KREA_LORA_KOHYA, mf.KREA2), (KREA_LORA_DIFFUSERS, mf.KREA2),
                              (MOSS_LORA_KOHYA, mf.MOSS), (MOSS_LORA_PEFT, mf.MOSS), (SDXL_LORA, mf.SDXL),
                              (QWEN_IMAGE_LORA, mf.OTHER), (ZIMAGE_LORA, mf.OTHER)):
            self.assertEqual(mf.classify_names(names), mf.FileInfo(mf.LORA, family), names)

    def test_models_and_support_files(self):
        self.assertEqual(mf.classify_names(KREA_MODEL), mf.FileInfo(mf.MODEL, mf.KREA2))
        self.assertEqual(mf.classify_names(KREA_MODEL_GGUF), mf.FileInfo(mf.MODEL, mf.KREA2))
        self.assertEqual(mf.classify_names(MOSS_MODEL), mf.FileInfo(mf.MODEL, mf.MOSS))
        self.assertEqual(mf.classify_names(SDXL_CHECKPOINT), mf.FileInfo(mf.MODEL, mf.SDXL))
        self.assertEqual(mf.classify_names(VAE).kind, mf.VAE)
        self.assertEqual(mf.classify_names(TEXT_ENCODER).kind, mf.TEXT_ENCODER)
        self.assertEqual(mf.classify_names([], 'qwen3vl').kind, mf.TEXT_ENCODER)
        self.assertEqual(mf.classify_names(['x.weight'], 'krea2'), mf.FileInfo(mf.MODEL, mf.KREA2))
        self.assertEqual(mf.classify_names([]).kind, mf.UNKNOWN)

    def test_lora_family_keeps_old_defaults(self):
        self.assertEqual(lf.family_from_keys([]), lf.SDXL)
        self.assertEqual(lf.family_from_keys(['something.unknown']), lf.SDXL)
        self.assertEqual(lf.family_from_keys(ZIMAGE_LORA), lf.OTHER)


class TestHeaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_gguf_diffusion_model_and_lora(self):
        model = os.path.join(self.dir, 'realism_engine_Q8_0.gguf')
        write_gguf(model, KREA_MODEL_GGUF, architecture='krea2')
        self.assertEqual(mf.read_gguf_header(model), ('krea2', KREA_MODEL_GGUF))
        self.assertEqual(mf.inspect(model), mf.FileInfo(mf.MODEL, mf.KREA2))
        lora = os.path.join(self.dir, 'MOSS-Image-Anime-LoRA-F16.gguf')
        write_gguf(lora, ['layers.0.proj.q.lora_down.weight', 'layers.0.proj.q.lora_up.weight'])
        self.assertEqual(mf.inspect(lora), mf.FileInfo(mf.LORA, mf.MOSS))

    def test_gguf_text_encoder_stops_at_architecture(self):
        llm = os.path.join(self.dir, 'Qwen3VL-4B.gguf')
        write_gguf(llm, ['token_embd.weight', 'blk.0.attn_q.weight'], architecture='qwen3vl', tokens=500)
        self.assertEqual(mf.read_gguf_header(llm), ('qwen3vl', []))
        self.assertEqual(mf.inspect(llm).kind, mf.TEXT_ENCODER)
        no_arch = os.path.join(self.dir, 'te.gguf')
        write_gguf(no_arch, ['token_embd.weight', 'blk.0.attn_q.weight'], tokens=50)
        self.assertEqual(mf.inspect(no_arch).kind, mf.TEXT_ENCODER)

    def test_unreadable_files(self):
        for name, data in (('empty.gguf', b''), ('bad.gguf', b'GGUF\x03\x00\x00\x00\xff'),
                           ('bad.safetensors', b'\xff' * 8), ('old.ckpt', b'PK')):
            path = os.path.join(self.dir, name)
            with open(path, 'wb') as fh:
                fh.write(data)
            self.assertEqual(mf.inspect(path).kind, mf.UNKNOWN, name)
        self.assertEqual(mf.inspect(os.path.join(self.dir, 'missing.gguf')).kind, mf.UNKNOWN)

    def test_scan_folder_recurses_and_skips_partials(self):
        for rel in ('a.gguf', 'sub/b.safetensors', 'c.gguf.partial', '.hidden.gguf', 'notes.txt', '.cache/d.gguf'):
            path = os.path.join(self.dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').close()
        self.assertEqual([n for n, _ in mf.scan_folder(self.dir)], ['a.gguf', os.path.join('sub', 'b.safetensors')])
        self.assertEqual(mf.scan_folder(os.path.join(self.dir, 'nope')), [])


class TestSdcppDiscovery(unittest.TestCase):
    """The Krea 2 / MOSS-Image model lists: every checkpoint of the family,
    whatever its name, in its folder, a subfolder or the checkpoints folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.engine, self.krea, self.moss, self.ckpt = (os.path.join(root, d) for d in
                                                        ('engine', 'krea', 'moss', 'checkpoints'))
        for d in (self.engine, self.krea, self.moss, self.ckpt):
            os.makedirs(d)
        cli = os.path.join(self.engine, 'sd-cli')
        open(cli, 'w').close()
        os.chmod(cli, os.stat(cli).st_mode | stat.S_IXUSR)
        for spec, folder in ((KREA2, self.krea), (MOSS_IMAGE, self.moss)):
            for f in spec.support_files():
                open(os.path.join(folder, f.name), 'w').close()
        self.dirs = {'krea2': self.krea, 'moss_image': self.moss}

    def tearDown(self):
        registry.register_installed({})
        self.tmp.cleanup()

    def test_lists_every_model_of_the_family(self):
        open(os.path.join(self.krea, KREA2.file('diffusion').name), 'w').close()  # the downloaded one (no header)
        write_gguf(os.path.join(self.krea, 'realismEngine_v2_Q8_0.gguf'), KREA_MODEL_GGUF)  # any name
        write_gguf(os.path.join(self.krea, 'base', 'Krea-2-Base-Q8_0.gguf'), KREA_MODEL_GGUF)  # subfolder
        write_safetensors(os.path.join(self.krea, 'cyberrealistic_fp8.safetensors'), KREA_MODEL)  # safetensors
        write_safetensors(os.path.join(self.krea, 'Krea2-realism-V2.safetensors'), KREA_LORA_COMFY)  # a LoRA
        write_gguf(os.path.join(self.krea, 'qwen3vl_4b-iq4_xs.gguf'), [], architecture='qwen3vl')  # text encoder
        write_gguf(os.path.join(self.moss, 'moss_image_sft_turbo-Q8_0.gguf'), MOSS_MODEL)
        write_gguf(os.path.join(self.moss, 'MOSS-Image-NSFW-LoRA-F16.gguf'),
                   ['layers.0.proj.q.lora_down.weight'])  # a LoRA whose name looks like a model
        write_safetensors(os.path.join(self.ckpt, 'krea2_raw_bf16.safetensors'), KREA_MODEL)  # in checkpoints
        write_safetensors(os.path.join(self.ckpt, 'juggernautXL.safetensors'), SDXL_CHECKPOINT)

        names = app.model_choices(self.engine, self.dirs, extra_dirs=[self.ckpt])
        self.assertEqual(names, [os.path.join('base', 'Krea-2-Base-Q8_0.gguf'), 'cyberrealistic_fp8.safetensors',
                                 'Krea-2-Turbo-Q5_K_S.gguf', 'krea2_raw_bf16.safetensors',
                                 'realismEngine_v2_Q8_0.gguf', 'moss_image_sft_turbo-Q8_0.gguf'])
        self.assertIs(registry.spec_for('realismEngine_v2_Q8_0.gguf'), KREA2)
        self.assertIs(registry.spec_for('krea2_raw_bf16.safetensors'), KREA2)
        self.assertEqual(app.diffusion_path('krea2_raw_bf16.safetensors'),
                         os.path.join(self.ckpt, 'krea2_raw_bf16.safetensors'))
        self.assertEqual(ui_family.models_for(ui_family.KREA2, names),
                         [n for n in names if registry.spec_for(n) is KREA2])

    def test_argv_takes_the_found_path(self):
        write_safetensors(os.path.join(self.ckpt, 'krea2_raw_bf16.safetensors'), KREA_MODEL)
        app.model_choices(self.engine, self.dirs, extra_dirs=[self.ckpt])
        settings = KREA2.settings_for('krea2_raw_bf16.safetensors')
        argv = runner.build_argv('sd-cli', KREA2, self.krea, app.diffusion_path('krea2_raw_bf16.safetensors'),
                                 prompt='p', negative_prompt='blurry', width=1024, height=1024, seed=1,
                                 steps=settings['steps'], cfg=settings['cfg'], output='o.png',
                                 uses_negative=settings['uses_negative'], scheduler=settings['scheduler'])
        self.assertEqual(argv[argv.index('--diffusion-model') + 1], os.path.join(self.ckpt, 'krea2_raw_bf16.safetensors'))
        self.assertEqual(argv[argv.index('--llm') + 1], os.path.join(self.krea, KREA2.file('llm').name))
        self.assertEqual(argv[argv.index('-n') + 1], 'blurry')
        self.assertEqual(argv[argv.index('--steps') + 1], '50')
        self.assertEqual(argv[argv.index('--cfg-scale') + 1], '3')


class TestVariants(unittest.TestCase):
    def test_settings(self):
        turbo = KREA2.settings_for('Krea-2-Turbo-Q5_K_S.gguf')
        self.assertEqual((turbo['steps'], turbo['cfg'], turbo['uses_negative']), (8, 1.0, False))
        for name in ('Krea-2-Base-Q8_0.gguf', 'krea2_raw-Q4_K_M.gguf', 'krea2_raw_bf16-Q8_0.gguf'):
            s = KREA2.settings_for(name)
            self.assertEqual((s['steps'], s['cfg'], s['uses_negative']), (50, 3.0, True), name)
        self.assertEqual(KREA2.settings_for('Krea-2-Base-Turbo-fp8.safetensors')['steps'], 8)
        self.assertEqual(MOSS_IMAGE.settings_for('MOSS-Image-SFT-Q8_0.gguf')['steps'], 50)
        self.assertEqual(MOSS_IMAGE.settings_for('moss_image_sft_turbo-Q8_0.gguf')['cfg'], 1.0)
        self.assertEqual(MOSS_IMAGE.settings_for('moss_image_sft_4step_v2.0.gguf')['steps'], 4)

    def test_negative_prompt_box(self):
        self.assertTrue(ui_family.uses_negative_prompt(ui_family.SDXL, 'x.safetensors'))
        self.assertFalse(ui_family.uses_negative_prompt(ui_family.KREA2, 'Krea-2-Turbo-Q5_K_S.gguf'))
        self.assertTrue(ui_family.uses_negative_prompt(ui_family.KREA2, 'Krea-2-Base-Q8_0.gguf'))
        self.assertTrue(ui_family.uses_negative_prompt(ui_family.MOSS))
        self.assertFalse(ui_family.uses_negative_prompt(ui_family.MOSS, 'moss_image_sft_turbo-Q8_0.gguf'))


class TestLoraPanel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        for name, keys in (('krea_style.safetensors', KREA_LORA_KOHYA), ('moss_anime.safetensors', MOSS_LORA_KOHYA),
                           ('offset.safetensors', SDXL_LORA), ('qwen_style.safetensors', QWEN_IMAGE_LORA)):
            write_safetensors(os.path.join(self.dir, name), keys)
        self.names = ['krea_style.safetensors', 'moss_anime.safetensors', 'offset.safetensors',
                      'qwen_style.safetensors']

    def tearDown(self):
        self.tmp.cleanup()

    def resolve(self, name):
        return os.path.join(self.dir, name)

    def test_lists_and_note(self):
        self.assertEqual(lf.loras_for(lf.KREA2, self.names, self.resolve), ['krea_style.safetensors'])
        self.assertEqual(lf.loras_for(lf.MOSS, self.names, self.resolve), ['moss_anime.safetensors'])
        self.assertEqual(lf.loras_for(lf.SDXL, self.names, self.resolve), ['offset.safetensors'])
        note = lf.panel_note(lf.KREA2, self.names, self.resolve)
        self.assertTrue(note.startswith('1 LoRA fit Krea 2.'), note)
        for part in ('1 for SDXL (offset)', '1 for MOSS-Image (moss_anime)', "can't run (qwen_style)"):
            self.assertIn(part, note)
        self.assertIn('No LoRAs found', lf.panel_note(lf.SDXL, [], self.resolve))


class TestConfigScan(unittest.TestCase):
    """LoRAs saved next to the models are listed; non-SDXL files in the
    checkpoints folder stay out of the SDXL list."""

    def test_checkpoints_and_loras_in_model_folders(self):
        import modules.config as config
        with tempfile.TemporaryDirectory() as root:
            ckpt, loras, krea = (os.path.join(root, d) for d in ('checkpoints', 'loras', 'krea2'))
            os.makedirs(loras)
            write_safetensors(os.path.join(ckpt, 'juggernautXL.safetensors'), SDXL_CHECKPOINT)
            write_safetensors(os.path.join(ckpt, 'detail_slider.safetensors'), SDXL_LORA)
            write_safetensors(os.path.join(ckpt, 'krea2_raw_bf16.safetensors'), KREA_MODEL)
            write_safetensors(os.path.join(ckpt, 'qwen_image_vae.safetensors'), VAE)
            open(os.path.join(ckpt, 'old_sd15.ckpt'), 'w').close()
            write_safetensors(os.path.join(krea, 'Krea2-realism-V2.safetensors'), KREA_LORA_COMFY)
            write_safetensors(os.path.join(loras, 'offset.safetensors'), SDXL_LORA)
            with mock.patch.object(config, 'paths_checkpoints', [ckpt]), \
                    mock.patch.object(config, 'paths_loras', [loras]), \
                    mock.patch.object(config, 'path_krea2', krea), \
                    mock.patch.object(config, 'path_moss_image', os.path.join(root, 'none')), \
                    mock.patch.object(config, 'path_flux', os.path.join(root, 'none2')):
                names = config.get_model_filenames([ckpt])
                self.assertEqual(sorted(config._sdxl_checkpoints(names)), ['juggernautXL.safetensors', 'old_sd15.ckpt'])
                extra, folders = config._loras_in_model_folders(['offset.safetensors'])
                self.assertEqual(sorted(extra), ['Krea2-realism-V2.safetensors', 'detail_slider.safetensors'])
                self.assertEqual(sorted(folders), sorted([ckpt, krea]))
                with mock.patch.object(config, 'lora_search_paths', [loras] + folders):
                    self.assertEqual(config.resolve_lora('Krea2-realism-V2.safetensors'),
                                     os.path.realpath(os.path.join(krea, 'Krea2-realism-V2.safetensors')))
                    self.assertEqual(lf.family_of_lora(config.resolve_lora('Krea2-realism-V2.safetensors')), lf.KREA2)


if __name__ == '__main__':
    unittest.main()
