"""Browser tests for the running app (headless Chrome over DevTools).

Skipped unless FOOOCUS_E2E_URL points at a running FoocusRX, e.g.

    FOOOCUS_E2E_URL=http://127.0.0.1:7866/ venv/bin/python -m unittest tests.e2e.test_ui -v

Opt-in extras:
  FOOOCUS_E2E_OUTPUTS=<the app's outputs folder> + FOOOCUS_E2E_ALLOW_DELETE=1
      History delete/undo. Point the app at a scratch copy of outputs and start
      it with FOOOCUS_TRASH_DIR=<scratch dir> so the real Trash is never used.
  FOOOCUS_E2E_GENERATE=1   real FLUX and SDXL-draft generations (minutes, GPU).
  FOOOCUS_E2E_SHOTS=<dir>  save screenshots there.
"""
import asyncio
import glob
import json
import os
import unittest

URL = os.environ.get('FOOOCUS_E2E_URL')
OUTPUTS = os.environ.get('FOOOCUS_E2E_OUTPUTS')
SHOTS = os.environ.get('FOOOCUS_E2E_SHOTS')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

THUMBS = "[...document.querySelectorAll('#history_gallery button.thumbnail-item')]"
MARKED = "[...document.querySelectorAll('#history_marked_gallery button.thumbnail-item')]"
HISTORY_STATE = f"""(() => {{ const w = document.querySelector('#history_gallery .grid-wrap');
  return {{thumbs: {THUMBS}.length, marked: {MARKED}.length,
          delete_label: document.querySelector('#history_delete').innerText.trim(),
          undo_enabled: !document.querySelector('#history_undo').disabled,
          wrap_h: w && w.clientHeight, wrap_sh: w && w.scrollHeight, wrap_st: w && w.scrollTop}}; }})()"""
MAIN_STATE = """(() => { const v = s => { const e = document.querySelector(s); return !!(e && e.offsetParent !== null); };
  return {family: document.querySelector('#model_family input').value, model: document.querySelector('#base_model input').value,
          negative_visible: v('#negative_prompt_accordion'), render_full_visible: v('#render_full_button'),
          gallery_imgs: document.querySelectorAll('#final_gallery img').length,
          input_image_visible: [...document.querySelectorAll('label')].some(l => l.innerText.trim() === 'Input Image' && l.offsetParent !== null),
          ip_tab_present: [...document.querySelectorAll('button[role=tab]')].some(b => b.innerText.trim() === 'Image Prompt')}; })()"""


def run(scenario, path=''):
    from tests.e2e.cdp import run_with_page
    return run_with_page(URL.rstrip('/') + '/' + path, scenario)


def images_on_disk():
    return len([p for p in glob.glob(os.path.join(OUTPUTS, '20*', '*')) if p.lower().endswith(IMAGE_EXTS)])


async def shot(page, name):
    if SHOTS:
        os.makedirs(SHOTS, exist_ok=True)
        await page.screenshot(os.path.join(SHOTS, f'{name}.png'))


async def open_history(page):
    await page.show_advanced()
    await page.open_tab('History')
    assert await page.wait_for(f'{THUMBS}.length > 0', timeout=30), 'History shows no images'
    await asyncio.sleep(1.0)


@unittest.skipUnless(URL, 'set FOOOCUS_E2E_URL to run browser tests')
class TestPage(unittest.TestCase):
    def test_loads_light_and_dark_without_page_errors(self):
        for theme in ('light', 'dark'):
            async def scenario(page, theme=theme):
                await shot(page, f'{theme}_desktop')
                return page.exceptions, await page.js("document.querySelectorAll('#generate_button').length")
            exceptions, buttons = run(scenario, f'?__theme={theme}')
            self.assertEqual(exceptions, [], theme)
            self.assertEqual(buttons, 1)

    def test_stylesheet_applies(self):
        async def scenario(page):
            return await page.js("!!document.querySelector('#foocusrx-style') && "
                                 "getComputedStyle(document.querySelector('.rx-header')).display")
        self.assertEqual(run(scenario), 'flex')

    def test_model_family_round_trip(self):
        async def scenario(page):
            start = await page.js(MAIN_STATE)
            states = []
            if await page.pick_family('FLUX.1'):
                states.append(await page.js(MAIN_STATE))
                await page.pick_family('SDXL')
                states.append(await page.js(MAIN_STATE))
            return start, states, page.exceptions
        start, states, exceptions = run(scenario)
        if not states:
            self.skipTest('FLUX.1 is not installed')
        flux, back = states
        self.assertEqual(flux['family'], 'FLUX.1')
        self.assertTrue(flux['model'].endswith('.gguf'))
        self.assertFalse(flux['negative_visible'] or flux['render_full_visible'])
        self.assertTrue(flux['input_image_visible'])   # Vary / Upscale work with FLUX
        self.assertFalse(flux['ip_tab_present'])        # Image Prompt doesn't
        self.assertTrue(back['ip_tab_present'])
        self.assertEqual(back['family'], 'SDXL')
        self.assertTrue(back['negative_visible'])
        if start['family'] == 'SDXL':
            self.assertEqual(back['model'], start['model'])  # remembered per family
        self.assertEqual(exceptions, [])


@unittest.skipUnless(URL, 'set FOOOCUS_E2E_URL to run browser tests')
class TestHistory(unittest.TestCase):
    def test_grid_scrolls_with_mouse_wheel(self):
        async def scenario(page):
            await open_history(page)
            before = await page.js(HISTORY_STATE)
            if before['wrap_sh'] <= before['wrap_h']:
                return before, None
            await page.wheel("document.querySelector('#history_gallery .grid-wrap')", 300, times=4)
            await asyncio.sleep(0.6)
            return before, await page.js(HISTORY_STATE)
        before, after = run(scenario)
        if after is None:
            self.skipTest('not enough images to need scrolling')
        self.assertGreater(after['wrap_st'], 0)
        self.assertLessEqual(after['wrap_h'], 500)

    @unittest.skipUnless(OUTPUTS and os.environ.get('FOOOCUS_E2E_ALLOW_DELETE') == '1',
                         'set FOOOCUS_E2E_OUTPUTS and FOOOCUS_E2E_ALLOW_DELETE=1 (scratch outputs only)')
    def test_select_delete_and_undo(self):
        start = images_on_disk()

        async def scenario(page):
            await open_history(page)
            await page.click_label('Select multiple')
            await asyncio.sleep(0.6)
            for i in (0, 1, 2):
                await page.click(f'{THUMBS}[{i}]')
                await asyncio.sleep(0.8)
            selected = await page.js(HISTORY_STATE)
            await page.click(f'{MARKED}[1]')  # unselect one
            await asyncio.sleep(0.8)
            unselected = await page.js(HISTORY_STATE)
            await shot(page, 'history_selected')
            await page.click("document.querySelector('#history_delete')")
            await asyncio.sleep(1.5)
            deleted, on_disk_after_delete = await page.js(HISTORY_STATE), images_on_disk()
            await page.click("document.querySelector('#history_undo')")
            await asyncio.sleep(1.5)
            undone, on_disk_after_undo = await page.js(HISTORY_STATE), images_on_disk()
            await page.click_button('Select all shown')
            await asyncio.sleep(0.8)
            all_label = (await page.js(HISTORY_STATE))['delete_label']
            await page.click_button('Clear selection')
            await asyncio.sleep(0.8)
            cleared = await page.js(HISTORY_STATE)
            return (selected, unselected, deleted, on_disk_after_delete, undone, on_disk_after_undo, all_label,
                    cleared, page.exceptions)

        (selected, unselected, deleted, disk_deleted, undone, disk_undone, all_label, cleared,
         exceptions) = run(scenario)
        self.assertEqual(selected['marked'], 3)
        self.assertIn('Delete 3 selected', selected['delete_label'])
        self.assertEqual(unselected['marked'], 2)
        self.assertEqual(disk_deleted, start - 2)
        self.assertTrue(deleted['undo_enabled'])
        self.assertEqual(disk_undone, start)
        self.assertEqual(undone['thumbs'], selected['thumbs'])
        self.assertRegex(all_label, r'Delete \d+ selected')
        self.assertEqual(cleared['marked'], 0)
        self.assertEqual(exceptions, [])


def log_entries(day_dir):
    from modules.history import parse_log_html
    try:
        with open(os.path.join(day_dir, 'log.html'), encoding='utf-8') as f:
            return set(parse_log_html(f.read()))
    except OSError:
        return set()


@unittest.skipUnless(URL, 'set FOOOCUS_E2E_URL to run browser tests')
class TestHistoryBrowse(unittest.TestCase):
    COUNT = "document.querySelector('#history_count').innerText.trim()"

    def test_thumbnails_paging_favourite_rating_and_filters(self):
        async def scenario(page):
            out = {}
            await open_history(page)
            out['src'] = await page.js(f"{THUMBS}[0].querySelector('img').getAttribute('src')")
            out['count'] = await page.js(self.COUNT)
            out['thumbs'] = (await page.js(HISTORY_STATE))['thumbs']
            if await page.js("document.querySelector('#history_more').offsetParent !== null"):
                await page.click("document.querySelector('#history_more')")
                await asyncio.sleep(1.5)
                out['thumbs_more'] = (await page.js(HISTORY_STATE))['thumbs']
            await page.click(f'{THUMBS}[0]')
            await page.wait_for("!document.querySelector('#history_fav').disabled", timeout=10)
            await asyncio.sleep(0.5)
            if (await page.js("document.querySelector('#history_fav').innerText.trim()")).startswith('♥'):
                await page.click("document.querySelector('#history_fav')")  # left over from an earlier run
                await asyncio.sleep(0.8)
            await page.click("document.querySelector('#history_fav')")
            await asyncio.sleep(0.8)
            out['fav_label'] = await page.js("document.querySelector('#history_fav').innerText.trim()")
            await page.click_label('★★★', "document.querySelector('#history_stars')")
            await asyncio.sleep(0.8)
            await page.click_label('♥ Favourites only')
            await asyncio.sleep(1.5)
            out['fav_only'] = (await page.js(HISTORY_STATE))['thumbs']
            out['fav_caption'] = await page.js(f"{THUMBS}[0] && {THUMBS}[0].innerText")
            await page.click_label('♥ Favourites only')  # back to everything
            await asyncio.sleep(1.5)
            # Leave no rating behind: clear it on the same image.
            await page.click(f'{THUMBS}[0]')
            await page.wait_for("!document.querySelector('#history_fav').disabled", timeout=10)
            await asyncio.sleep(0.5)
            await page.click("document.querySelector('#history_fav')")
            await asyncio.sleep(0.6)
            await page.click_label('☆', "document.querySelector('#history_stars')")
            await asyncio.sleep(0.6)
            out['fav_label_after'] = await page.js("document.querySelector('#history_fav').innerText.trim()")
            out['fav_enabled_after'] = await page.js("!document.querySelector('#history_fav').disabled")
            out['exceptions'] = page.exceptions
            await shot(page, 'history_browse')
            return out
        out = run(scenario)
        # Gradio serves a cached copy, so check the thumbnail's file name (<image>.<ext>.jpg).
        self.assertRegex(out['src'], r'\.(png|jpe?g|webp)\.jpg$')
        self.assertRegex(out['count'], r'Showing \d+ of \d+')
        if 'thumbs_more' in out:
            self.assertGreater(out['thumbs_more'], out['thumbs'])
        self.assertEqual(out['fav_label'], '♥ Favourite')
        self.assertEqual(out['fav_only'], 1)
        self.assertIn('♥', out['fav_caption'])
        self.assertIn('★★★', out['fav_caption'])
        self.assertEqual(out['fav_label_after'], '♡ Favourite')
        self.assertTrue(out['fav_enabled_after'])  # re-selecting the same image after a refresh works
        if OUTPUTS:
            from modules.history import RatingsStore
            self.assertEqual(RatingsStore(OUTPUTS).load(), {})  # the test leaves no rating behind
        self.assertEqual(out['exceptions'], [])

    @unittest.skipUnless(OUTPUTS and os.environ.get('FOOOCUS_E2E_ALLOW_DELETE') == '1',
                         'set FOOOCUS_E2E_OUTPUTS and FOOOCUS_E2E_ALLOW_DELETE=1 (scratch outputs only)')
    def test_delete_tidies_log_and_undo_restores_it(self):
        async def scenario(page):
            await open_history(page)
            await page.click(f'{THUMBS}[0]')
            await asyncio.sleep(0.8)
            details = await page.js("[...document.querySelectorAll('#rx_panel textarea')].map(t => t.value).find(v => v.startsWith('File: '))")
            name = details.split('\n')[0][len('File: '):]
            day = details.split('\n')[1][len('Date: '):]
            day_dir = os.path.join(OUTPUTS, day)
            before = log_entries(day_dir)
            await page.click("document.querySelector('#history_delete')")
            await asyncio.sleep(1.5)
            after_delete = log_entries(day_dir)
            await page.click("document.querySelector('#history_undo')")
            await asyncio.sleep(1.5)
            return name, before, after_delete, log_entries(day_dir), page.exceptions
        name, before, after_delete, after_undo, exceptions = run(scenario)
        if name not in before:
            self.skipTest('the newest image has no log.html entry')
        self.assertNotIn(name, after_delete)
        self.assertEqual(after_undo, before)
        self.assertEqual(exceptions, [])


async def set_textbox(page, selector, text):
    await page.click(f"document.querySelector({json.dumps(selector)})")
    await page.js(f"(() => {{ const t = document.querySelector({json.dumps(selector)}); t.select(); }})()")
    await page.send('Input.insertText', text=text)
    await asyncio.sleep(0.3)


async def pick_dropdown(page, elem_id, text_prefix):
    await page.click(f"document.querySelector('#{elem_id} input')")
    await asyncio.sleep(0.6)
    return await page.click(f"[...document.querySelectorAll('ul.options li')].find(li => li.innerText.trim().startsWith({json.dumps(text_prefix)}))")


@unittest.skipUnless(URL, 'set FOOOCUS_E2E_URL to run browser tests')
class TestPromptLibrary(unittest.TestCase):
    def test_save_find_use_delete(self):
        marker = 'e2e prompt library check, a teal teapot on a desk'

        async def scenario(page):
            out = {}
            await set_textbox(page, '#positive_prompt textarea', marker)
            await page.show_advanced()
            await page.open_tab('Prompts')
            await asyncio.sleep(1.0)
            await set_textbox(page, '#prompts_name textarea, #prompts_name input', 'E2E teapot')
            await set_textbox(page, '#prompts_tags textarea, #prompts_tags input', 'e2e, objects')
            await page.click("document.querySelector('#prompts_save')")
            await asyncio.sleep(1.2)
            out['status'] = await page.js("document.querySelector('#prompts_status').innerText.trim()")
            # Saving selects the new entry in the list.
            out['picked'] = (await page.js("document.querySelector('#prompts_list input').value")).startswith('E2E teapot')
            out['preview'] = await page.js("document.querySelector('#prompts_preview textarea').value")
            await set_textbox(page, '#positive_prompt textarea', 'something else')
            await page.click("document.querySelector('#prompts_use')")
            await asyncio.sleep(1.0)
            out['prompt_after_use'] = await page.js("document.querySelector('#positive_prompt textarea').value")
            await page.click("document.querySelector('#prompts_delete')")
            await asyncio.sleep(1.0)
            out['status_after_delete'] = await page.js("document.querySelector('#prompts_status').innerText.trim()")
            out['exceptions'] = page.exceptions
            return out

        out = run(scenario)
        self.assertIn('Saved', out['status'])
        self.assertTrue(out['picked'])
        self.assertTrue(out['preview'].startswith(marker))
        self.assertEqual(out['prompt_after_use'], marker)
        self.assertEqual(out['status_after_delete'], 'Deleted.')
        if OUTPUTS:
            from modules.prompt_library import PromptLibrary
            self.assertEqual(PromptLibrary(OUTPUTS).search(marker), [])  # nothing left behind
        self.assertEqual(out['exceptions'], [])


@unittest.skipUnless(URL and os.environ.get('FOOOCUS_E2E_GENERATE') == '1',
                     'set FOOOCUS_E2E_URL and FOOOCUS_E2E_GENERATE=1 (uses the GPU for minutes)')
class TestCompare(unittest.TestCase):
    def test_compare_sdxl_with_flux_same_seed(self):
        async def scenario(page):
            await set_textbox(page, '#positive_prompt textarea', 'a lighthouse on a cliff at sunset')
            await page.click("[...document.querySelectorAll('#compare_accordion button, #compare_accordion .label-wrap')][0]")
            await asyncio.sleep(0.8)
            if not await pick_dropdown(page, 'compare_model_b', 'flux1'):
                return None
            await page.click("document.querySelector('#compare_button')")
            await page.wait_for("document.querySelector('#compare_button').disabled", timeout=30)
            done = await page.wait_for("!document.querySelector('#compare_button').disabled && "
                                       "document.querySelectorAll('#final_gallery img').length >= 2", timeout=900)
            captions = await page.js("[...document.querySelectorAll('#final_gallery .caption-label, #final_gallery span')]"
                                     ".map(e => e.innerText.trim()).filter(t => /^[AB] \\u00b7/.test(t))")
            await shot(page, 'compare')
            return done, captions, page.exceptions
        result = run(scenario)
        if result is None:
            self.skipTest('FLUX.1 is not installed')
        done, captions, exceptions = result
        self.assertTrue(done)
        self.assertTrue(any(c.startswith('A \u00b7') for c in captions), captions)
        self.assertTrue(any(c.startswith('B \u00b7 flux1') for c in captions), captions)
        self.assertEqual(exceptions, [])


@unittest.skipUnless(URL and os.environ.get('FOOOCUS_E2E_GENERATE') == '1',
                     'set FOOOCUS_E2E_URL and FOOOCUS_E2E_GENERATE=1 (uses the GPU for minutes)')
class TestGenerate(unittest.TestCase):
    async def _generate(self, page, timeout=900):
        await page.click("document.querySelector('#generate_button')")
        await page.wait_for("(() => { const s=document.querySelector('#stop_button'); return s && s.offsetParent !== null; })()", timeout=60)
        return await page.wait_for("document.querySelector('#generate_button').offsetParent !== null && "
                                   "document.querySelectorAll('#final_gallery img').length > 0", timeout=timeout)

    def test_flux_then_sdxl_draft_then_history_load(self):
        async def scenario(page):
            await page.click("document.querySelector('#positive_prompt textarea')")
            await page.send('Input.insertText', text='a red fox sitting in fresh snow at sunrise')
            results = {}
            if await page.pick_family('FLUX.1'):
                results['flux'] = (await self._generate(page), await page.js(MAIN_STATE))
                await shot(page, 'generate_flux')
            await page.pick_family('SDXL')
            await page.show_advanced()
            await page.click_label('Draft mode (fast previews)')
            await asyncio.sleep(0.5)
            results['sdxl'] = (await self._generate(page), await page.js(MAIN_STATE))
            await shot(page, 'generate_sdxl_draft')
            return results, page.exceptions
        results, exceptions = run(scenario)
        for name, (finished, state) in results.items():
            self.assertTrue(finished, f'{name} generation did not finish')
            self.assertGreater(state['gallery_imgs'], 0, name)
        self.assertEqual(exceptions, [])


if __name__ == '__main__':
    unittest.main()
