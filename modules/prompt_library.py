"""Prompt library: save, tag, search and reuse prompts.

Stored as JSON in <outputs>/.prompt_library.json (next to the images, never
in the repo). Writes are atomic and serialised; entries are plain dicts:
{id, name, prompt, negative, tags, created}.
"""
import json
import os
import threading
import time
import uuid

FILE_NAME = '.prompt_library.json'
_lock = threading.Lock()


def _parse_tags(tags) -> list:
    if isinstance(tags, str):
        tags = tags.split(',')
    out = []
    for tag in tags or []:
        tag = str(tag).strip().lower()
        if tag and tag not in out:
            out.append(tag)
    return out


class PromptLibrary:
    def __init__(self, root: str):
        self.path = os.path.join(root, FILE_NAME)

    def load(self) -> list:
        try:
            with open(self.path, encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return []
        return [e for e in data if isinstance(e, dict) and e.get('id') and isinstance(e.get('prompt'), str)] \
            if isinstance(data, list) else []

    def _write(self, entries: list):
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(entries, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    def save(self, prompt: str, negative: str = '', name: str = '', tags='') -> dict:
        """Add a prompt; saving the same prompt + negative again updates its name/tags."""
        prompt = (prompt or '').strip()
        if not prompt:
            raise ValueError('Type a prompt first.')
        negative = (negative or '').strip()
        name = (name or '').strip() or (prompt[:48] + ('…' if len(prompt) > 48 else ''))
        with _lock:
            entries = self.load()
            for entry in entries:
                if entry['prompt'] == prompt and entry.get('negative', '') == negative:
                    entry.update(name=name, tags=_parse_tags(tags) or entry.get('tags', []))
                    self._write(entries)
                    return entry
            entry = {'id': uuid.uuid4().hex[:12], 'name': name, 'prompt': prompt, 'negative': negative,
                     'tags': _parse_tags(tags), 'created': time.strftime('%Y-%m-%d %H:%M')}
            entries.insert(0, entry)
            self._write(entries)
            return entry

    def delete(self, entry_id: str) -> bool:
        with _lock:
            entries = self.load()
            kept = [e for e in entries if e['id'] != entry_id]
            if len(kept) == len(entries):
                return False
            self._write(kept)
            return True

    def get(self, entry_id: str):
        return next((e for e in self.load() if e['id'] == entry_id), None)

    def search(self, query: str = '', tag: str = '') -> list:
        terms = (query or '').lower().split()
        tag = (tag or '').strip().lower()
        out = []
        for entry in self.load():
            if tag and tag not in entry.get('tags', []):
                continue
            haystack = ' '.join([entry.get('name', ''), entry['prompt'], entry.get('negative', ''),
                                 ' '.join(entry.get('tags', []))]).lower()
            if all(t in haystack for t in terms):
                out.append(entry)
        return out

    def all_tags(self) -> list:
        tags = []
        for entry in self.load():
            for tag in entry.get('tags', []):
                if tag not in tags:
                    tags.append(tag)
        return sorted(tags)


def label(entry: dict) -> str:
    tags = f"  [{', '.join(entry['tags'])}]" if entry.get('tags') else ''
    return f"{entry['name']}{tags}"
