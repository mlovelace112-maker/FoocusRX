// FoocusRX full-window image viewer.
//
// rxOpenViewer(urls, index) shows images over the whole window:
//   click the image: fit to window <-> actual size (scroll to pan)
//   left / right (or the arrows): previous / next      Esc or backdrop: close
//   buttons: open in a new tab, download, close
// rxViewGallery(selector) opens the images of a Gradio gallery, starting at the
// selected one. Independent of Gradio's internal markup beyond <img> tags.

(function () {
    let state = {urls: [], index: 0, actual: false};

    function el(tag, attrs, children) {
        const e = document.createElement(tag);
        Object.entries(attrs || {}).forEach(([k, v]) => {
            if (k === 'class') e.className = v;
            else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
            else e.setAttribute(k, v);
        });
        (children || []).forEach(c => e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
        return e;
    }

    function stop(e) { e.stopPropagation(); }

    function build() {
        let root = document.getElementById('rx_viewer');
        if (root) return root;
        const img = el('img', {id: 'rx_viewer_img', alt: '', onclick: e => { stop(e); toggleSize(); }});
        const stage = el('div', {id: 'rx_viewer_stage', onclick: close}, [img]);
        const btn = (id, label, title, fn) => el('button', {id, title, 'aria-label': title, type: 'button',
                                                            onclick: e => { stop(e); fn(); }}, [label]);
        root = el('div', {id: 'rx_viewer', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Image viewer',
                          tabindex: '-1'}, [
            stage,
            btn('rx_viewer_prev', '‹', 'Previous image (←)', () => go(-1)),
            btn('rx_viewer_next', '›', 'Next image (→)', () => go(1)),
            el('div', {id: 'rx_viewer_bar', onclick: stop}, [
                el('span', {id: 'rx_viewer_count'}),
                btn('rx_viewer_size', 'Actual size', 'Fit to window / actual size (click the image)', toggleSize),
                btn('rx_viewer_newtab', 'Open in new tab ↗', 'Open the full image in a new tab', () =>
                    window.open(state.urls[state.index], '_blank', 'noopener')),
                btn('rx_viewer_download', 'Download', 'Download this image', download),
                btn('rx_viewer_close', '✕', 'Close (Esc)', close),
            ]),
        ]);
        document.body.appendChild(root);
        document.addEventListener('keydown', e => {
            if (!root.classList.contains('open')) return;
            if (e.key === 'Escape') { e.preventDefault(); close(); }
            else if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
            else if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
        }, true);
        return root;
    }

    function render() {
        const root = build();
        const img = document.getElementById('rx_viewer_img');
        img.src = state.urls[state.index];
        root.classList.toggle('actual', state.actual);
        document.getElementById('rx_viewer_size').textContent = state.actual ? 'Fit to window' : 'Actual size';
        document.getElementById('rx_viewer_count').textContent =
            state.urls.length > 1 ? `${state.index + 1} / ${state.urls.length}` : '';
        const many = state.urls.length > 1;
        document.getElementById('rx_viewer_prev').style.display = many ? '' : 'none';
        document.getElementById('rx_viewer_next').style.display = many ? '' : 'none';
    }

    function go(step) {
        if (state.urls.length < 2) return;
        state.index = (state.index + step + state.urls.length) % state.urls.length;
        state.actual = false;
        render();
    }

    function toggleSize() { state.actual = !state.actual; render(); }

    function download() {
        const url = state.urls[state.index];
        const a = el('a', {href: url, download: decodeURIComponent(url.split('/').pop().split('?')[0])});
        document.body.appendChild(a); a.click(); a.remove();
    }

    function close() {
        const root = document.getElementById('rx_viewer');
        if (root) root.classList.remove('open');
        document.body.classList.remove('rx-viewer-open');
    }

    window.rxOpenViewer = function (urls, index) {
        urls = [...new Set((urls || []).filter(Boolean))];
        if (!urls.length) return false;
        state = {urls, index: Math.min(Math.max(index || 0, 0), urls.length - 1), actual: false};
        const root = build();
        render();
        root.classList.add('open');
        document.body.classList.add('rx-viewer-open');
        root.focus();
        return true;
    };

    function galleryImages(selector) {
        const gallery = document.querySelector(selector);
        if (!gallery) return {urls: [], index: 0};
        // Thumbnails (and the preview, when open) are <img>s with the full file URL.
        const thumbs = [...gallery.querySelectorAll('button.thumbnail-item img, .thumbnail-item img')];
        const imgs = thumbs.length ? thumbs : [...gallery.querySelectorAll('img')];
        const urls = [...new Set(imgs.map(i => i.currentSrc || i.src).filter(Boolean))];
        const selected = gallery.querySelector('.thumbnail-item.selected img');
        const preview = gallery.querySelector('.preview img, img[data-testid="detailed-image"]');
        const current = (selected && (selected.currentSrc || selected.src)) || (preview && preview.src);
        return {urls, index: Math.max(0, urls.indexOf(current))};
    }

    window.rxViewGallery = function (selector) {
        const {urls, index} = galleryImages(selector || '#final_gallery');
        return window.rxOpenViewer(urls, index);
    };

    // Double-click an image in the main gallery to view it larger.
    document.addEventListener('dblclick', e => {
        const img = e.target.closest && e.target.closest('#final_gallery img');
        if (!img) return;
        const {urls} = galleryImages('#final_gallery');
        window.rxOpenViewer(urls, Math.max(0, urls.indexOf(img.currentSrc || img.src)));
    });
})();
