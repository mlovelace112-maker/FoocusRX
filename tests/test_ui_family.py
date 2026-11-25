import unittest

from modules import ui_family as uf

MODELS = ['juggernautXL_v8Rundiffusion.safetensors', 'ponyDiffusionV6XL.safetensors', 'flux1-schnell-Q4_K_S.gguf']


class TestFamily(unittest.TestCase):
    def test_family_of_and_models_for(self):
        self.assertEqual(uf.family_of('flux1-schnell-Q4_K_S.gguf'), uf.FLUX)
        self.assertEqual(uf.family_of('x.safetensors'), uf.SDXL)
        self.assertEqual(uf.family_of(None), uf.SDXL)
        self.assertEqual(uf.models_for(uf.FLUX, MODELS), ['flux1-schnell-Q4_K_S.gguf'])
        self.assertEqual(len(uf.models_for(uf.SDXL, MODELS)), 2)

    def test_initial_memory_prefers_current(self):
        memory = uf.initial_memory('ponyDiffusionV6XL.safetensors', MODELS)
        self.assertEqual(memory, {uf.SDXL: 'ponyDiffusionV6XL.safetensors', uf.FLUX: 'flux1-schnell-Q4_K_S.gguf'})

    def test_select_family_restores_last_model(self):
        memory = {uf.SDXL: 'ponyDiffusionV6XL.safetensors', uf.FLUX: 'flux1-schnell-Q4_K_S.gguf'}
        family, choices, model, warning = uf.select_family(uf.SDXL, memory, MODELS)
        self.assertEqual((family, model, warning), (uf.SDXL, 'ponyDiffusionV6XL.safetensors', None))
        self.assertNotIn('flux1-schnell-Q4_K_S.gguf', choices)
        _, _, model, _ = uf.select_family(uf.FLUX, memory, MODELS)
        self.assertEqual(model, 'flux1-schnell-Q4_K_S.gguf')

    def test_select_family_forgotten_model_falls_back(self):
        _, _, model, _ = uf.select_family(uf.SDXL, {uf.SDXL: 'deleted.safetensors'}, MODELS)
        self.assertEqual(model, MODELS[0])

    def test_select_flux_when_not_installed(self):
        family, choices, model, warning = uf.select_family(uf.FLUX, {}, MODELS[:2])
        self.assertEqual((family, choices, model), (uf.SDXL, None, None))
        self.assertIn('download_flux.py', warning)

    def test_model_changed_switches_family_once(self):
        family, memory, choices = uf.model_changed('flux1-schnell-Q4_K_S.gguf', uf.SDXL, {}, MODELS)
        self.assertEqual(family, uf.FLUX)
        self.assertEqual(choices, ['flux1-schnell-Q4_K_S.gguf'])
        self.assertEqual(memory[uf.FLUX], 'flux1-schnell-Q4_K_S.gguf')
        # Same family already shown: no choices update (prevents change loops).
        _, memory2, choices2 = uf.model_changed('ponyDiffusionV6XL.safetensors', uf.SDXL, memory, MODELS)
        self.assertIsNone(choices2)
        self.assertEqual(memory2[uf.SDXL], 'ponyDiffusionV6XL.safetensors')
        self.assertEqual(memory2[uf.FLUX], 'flux1-schnell-Q4_K_S.gguf')


if __name__ == '__main__':
    unittest.main()
