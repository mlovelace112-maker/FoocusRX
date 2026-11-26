import unittest

import torch
from torch import nn

from modules.flux import lora as fl

NAMES = {'double_blocks.0.img_attn.qkv', 'double_blocks.0.txt_attn.qkv', 'double_blocks.0.img_attn.proj',
         'double_blocks.0.img_mlp.0', 'double_blocks.0.img_mod.lin', 'single_blocks.0.linear1',
         'single_blocks.0.linear2', 'single_blocks.0.modulation.lin', 'final_layer.adaLN_modulation.1', 'img_in'}


def ab(rank=4, out=8, inp=6):
    return torch.randn(rank, inp), torch.randn(out, rank)


class TestParse(unittest.TestCase):
    def test_kohya_with_alpha(self):
        a, b = ab()
        terms, unused = fl.parse({'lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight': a,
                                  'lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight': b,
                                  'lora_unet_double_blocks_0_img_attn_qkv.alpha': torch.tensor(2.0),
                                  'lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight': a,
                                  'lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight': b}, NAMES)
        (t,) = terms['double_blocks.0.img_attn.qkv']
        self.assertAlmostEqual(t.scale, 0.5)  # alpha / rank
        self.assertEqual((t.start, t.end), (0, None))
        self.assertEqual(unused, ['lora_te1_text_model_encoder_layers_0_mlp_fc1'])  # text-encoder LoRA

    def test_bfl_prefixes(self):
        for prefix in ('diffusion_model.', 'transformer.', ''):
            a, b = ab()
            terms, unused = fl.parse({f'{prefix}single_blocks.0.linear2.lora_A.weight': a,
                                      f'{prefix}single_blocks.0.linear2.lora_B.weight': b}, NAMES)
            self.assertIn('single_blocks.0.linear2', terms, prefix)
            self.assertEqual(terms['single_blocks.0.linear2'][0].scale, 1.0)
            self.assertEqual(unused, [])

    def test_diffusers_slices(self):
        state = {}
        for name in ('transformer.transformer_blocks.0.attn.to_k', 'transformer.transformer_blocks.0.attn.add_v_proj',
                     'transformer.single_transformer_blocks.0.proj_mlp', 'transformer.single_transformer_blocks.0.attn.to_q',
                     'transformer.transformer_blocks.0.ff.net.0.proj', 'transformer.transformer_blocks.0.norm1.linear',
                     'transformer.single_transformer_blocks.0.norm.linear', 'transformer.x_embedder'):
            a, b = ab()
            state[name + '.lora_A.weight'], state[name + '.lora_B.weight'] = a, b
        terms, unused = fl.parse(state, NAMES)
        self.assertEqual(unused, [])
        H = fl.HIDDEN
        self.assertEqual((terms['double_blocks.0.img_attn.qkv'][0].start, terms['double_blocks.0.img_attn.qkv'][0].end), (H, 2 * H))
        self.assertEqual((terms['double_blocks.0.txt_attn.qkv'][0].start, terms['double_blocks.0.txt_attn.qkv'][0].end), (2 * H, 3 * H))
        linear1 = sorted((t.start, t.end) for t in terms['single_blocks.0.linear1'])
        self.assertEqual(linear1, [(0, H), (3 * H, 3 * H + fl.MLP)])
        for name in ('double_blocks.0.img_mlp.0', 'double_blocks.0.img_mod.lin', 'single_blocks.0.modulation.lin', 'img_in'):
            self.assertIn(name, terms)

    def test_norm_out_halves_are_swapped(self):
        a = torch.randn(2, 6)
        b = torch.arange(8.0).reshape(8, 1).repeat(1, 2)
        terms, _ = fl.parse({'norm_out.linear.lora_A.weight': a, 'norm_out.linear.lora_B.weight': b}, NAMES)
        up = terms['final_layer.adaLN_modulation.1'][0].up
        self.assertTrue(torch.equal(up[:4], b[4:]) and torch.equal(up[4:], b[:4]))

    def test_unknown_and_incomplete_are_unused(self):
        a, b = ab()
        terms, unused = fl.parse({'foo.bar.lora_A.weight': a, 'foo.bar.lora_B.weight': b,
                                  'img_in.lora_A.weight': a}, NAMES)
        self.assertEqual(terms, {})
        self.assertEqual(sorted(unused), ['foo.bar', 'img_in'])


class TestHooks(unittest.TestCase):
    def test_hook_adds_low_rank_term_into_slice(self):
        model = nn.Module()
        model.single_blocks = nn.ModuleList([nn.Module()])
        lin = nn.Linear(6, 12, bias=False)
        model.single_blocks[0].linear1 = lin
        x = torch.randn(3, 6)
        base = lin(x)
        a, b = torch.randn(2, 6), torch.randn(4, 2)
        term = fl.LoraTerm(a, b, scale=0.5, start=4, end=8)
        handle = lin.register_forward_hook(fl._make_hook([term]))
        with torch.no_grad():
            out = lin(x)
        handle.remove()
        expected = base.clone()
        expected[:, 4:8] += 0.5 * (x @ a.T) @ b.T
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))
        self.assertTrue(torch.allclose(lin(x), base))  # removed cleanly


if __name__ == '__main__':
    unittest.main()
