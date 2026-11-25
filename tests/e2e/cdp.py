"""Minimal headless-Chrome driver over the DevTools protocol for UI tests.

Only ever opens the local app URL it is given. Real mouse events (not
element.click()) so overlapping elements are caught like a user would hit
them; page exceptions are collected so a test can assert there are none.
"""
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

DEFAULT_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


def chrome_path():
    return os.environ.get('FOOOCUS_E2E_CHROME', DEFAULT_CHROME)


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0
        self.exceptions = []

    async def send(self, method, **params):
        self.n += 1
        msg_id = self.n
        await self.ws.send(json.dumps({'id': msg_id, 'method': method, 'params': params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get('method') == 'Runtime.exceptionThrown':
                d = msg['params']['exceptionDetails']
                self.exceptions.append((d.get('exception') or {}).get('description', d.get('text', ''))[:300])
            if msg.get('id') == msg_id:
                if 'error' in msg:
                    raise RuntimeError(msg['error'])
                return msg.get('result', {})

    async def js(self, expr):
        r = await self.send('Runtime.evaluate', expression=expr, returnByValue=True, awaitPromise=True)
        return r.get('result', {}).get('value')

    async def click(self, selector_js):
        """Real mouse click at the centre of the element ``selector_js`` returns."""
        box = await self.js(f"""(() => {{ const e=({selector_js}); if(!e) return null; e.scrollIntoView({{block:'center'}});
                                const r=e.getBoundingClientRect(); return [r.x+r.width/2, r.y+r.height/2]; }})()""")
        if not box:
            return False
        for kind in ('mousePressed', 'mouseReleased'):
            await self.send('Input.dispatchMouseEvent', type=kind, x=box[0], y=box[1], button='left', clickCount=1)
        return True

    async def wheel(self, selector_js, delta_y, times=1):
        box = await self.js(f"""(() => {{ const r=({selector_js}).getBoundingClientRect();
                                return [r.x+r.width/2, r.y+r.height/2]; }})()""")
        for _ in range(times):
            await self.send('Input.dispatchMouseEvent', type='mouseWheel', x=box[0], y=box[1], deltaX=0, deltaY=delta_y)
            await asyncio.sleep(0.2)

    async def wait_for(self, expr, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(expr):
                return True
            await asyncio.sleep(0.3)
        return False

    async def screenshot(self, path, width=1440, height=1000):
        import base64
        await self.send('Emulation.setDeviceMetricsOverride', width=width, height=height, deviceScaleFactor=1,
                        mobile=False)
        data = (await self.send('Page.captureScreenshot', format='png'))['data']
        with open(path, 'wb') as f:
            f.write(base64.b64decode(data))

    async def open(self, url, width=1440, height=1000):
        await self.send('Page.enable')
        await self.send('Runtime.enable')
        await self.send('Emulation.setDeviceMetricsOverride', width=width, height=height, deviceScaleFactor=1,
                        mobile=False)
        await self.send('Page.navigate', url=url)
        if not await self.wait_for("!!document.querySelector('#generate_button')", timeout=90):
            raise TimeoutError(f'{url} did not load')
        await asyncio.sleep(2.5)

    # --- app helpers -------------------------------------------------------
    async def click_label(self, text, scope='document'):
        return await self.click(f"[...{scope}.querySelectorAll('label')].find(l=>l.innerText.trim()==={json.dumps(text)})")

    async def click_button(self, text, scope='document'):
        return await self.click(f"[...{scope}.querySelectorAll('button')].find(b=>b.innerText.trim()==={json.dumps(text)} && b.offsetParent!==null)")

    async def open_tab(self, name):
        return await self.click(f"[...document.querySelectorAll('button[role=tab]')].find(b=>b.innerText.trim()==={json.dumps(name)} && b.offsetParent!==null)")

    async def show_advanced(self):
        await self.click_label('Advanced')
        await self.wait_for("(() => { const t=[...document.querySelectorAll('button[role=tab]')].find(b=>b.innerText.trim()==='Settings'); return t && t.offsetParent !== null; })()", timeout=20)
        await asyncio.sleep(0.8)

    async def pick_family(self, name):
        await self.click("document.querySelector('#model_family input')")
        await asyncio.sleep(0.8)
        ok = await self.click(f"[...document.querySelectorAll('ul.options li')].find(li=>li.innerText.trim()==={json.dumps(name)})")
        await asyncio.sleep(2.5)
        return ok


def run_with_page(url, scenario, port=9341):
    """Start headless Chrome, open ``url``, run ``await scenario(page)``."""
    import websockets

    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([chrome_path(), '--headless=new', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                             '--hide-scrollbars', '--no-first-run', f'--user-data-dir={profile}',
                             f'--remote-debugging-port={port}', 'about:blank'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        target = None
        for _ in range(60):
            try:
                targets = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/json', timeout=1).read())
                target = next(t for t in targets if t.get('type') == 'page')
                break
            except Exception:
                time.sleep(0.5)
        if target is None:
            raise RuntimeError('Chrome did not start')

        async def main():
            async with websockets.connect(target['webSocketDebuggerUrl'], max_size=50 * 2**20) as ws:
                page = Page(ws)
                await page.open(url)
                return await scenario(page)

        return asyncio.run(main())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
