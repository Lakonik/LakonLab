# Copyright (c) 2026 Hansheng Chen

import html
import json
import os
import pathlib
from urllib.parse import urlparse, urljoin, quote
from bs4 import BeautifulSoup
from mmcv.fileio import FileClient


ASSET_DIR = pathlib.Path(__file__).parent
TEMPLATE = (ASSET_DIR / "viewer.html").read_text(encoding="utf-8")


def _make_abs(base_uri: str, src: str) -> str:
    """Return an absolute URL / path for <img>/<video> src."""
    # already absolute URL?
    if urlparse(src).scheme in {"http", "https"}:
        return src

    # base is a URL → use urljoin
    if urlparse(base_uri).scheme in {"http", "https"}:
        return urljoin(base_uri, src)

    # otherwise treat both as local paths
    return os.path.abspath(os.path.join(os.path.dirname(base_uri), src))


def build_thumbnails(entries, n_cols, lazy=False):
    """return markup, sources list, captions list

    Args:
        entries: list of (data_id, src, caption) tuples
        n_cols: number of columns in the grid
        lazy: if True, use data-src instead of src for lazy loading
    """
    sources, caps, blocks = [], [], []
    for i, (d, src, cap) in enumerate(entries):
        if not (src.startswith("http://") or src.startswith("https://")):
            src = quote(src, safe="/%")
        esc_src = html.escape(src)
        sources.append(esc_src)
        caps.append(cap)

        ext = os.path.splitext(esc_src)[-1].lower()
        src_attr = "data-src" if lazy else "src"
        if ext == '.mp4':
            # preload="metadata" needed to show first frame as thumbnail
            thumb = f'<video {src_attr}="{esc_src}" preload="metadata" muted></video>'
        else:
            thumb = f'<img {src_attr}="{esc_src}" alt="thumb">'
        blocks.append(f'<div class="item" data-idx="{i}">{thumb}'
                      f'<textarea class="prompt" readonly>'
                      f'{html.escape(cap)}</textarea></div>')
    grid = (f'<div class="grid" style="grid-template-columns:repeat({n_cols},1fr);">\n'
            if n_cols else '<div class="grid">\n')
    return grid + '\n'.join(blocks) + '\n</div>', sources, caps


def grid_html(entries, n_cols=None, *, inline_assets=False, page_size=None):
    """Generate HTML page with media grid.

    Args:
        entries: list of (data_id, src, caption) tuples
        n_cols: number of columns in the grid
        inline_assets: if True, inline CSS and JS into the HTML
        page_size: if set, enables hash-based pagination with lazy loading.
            Only the current page's media loads; navigation via URL hash.
    """
    use_pagination = page_size and len(entries) > page_size
    grid_markup, srcs, caps = build_thumbnails(entries, n_cols, lazy=use_pagination)
    data = {"sources": srcs, "captions": caps}

    if use_pagination:
        total_pages = (len(entries) + page_size - 1) // page_size
        data["pageSize"] = page_size
        data["totalPages"] = total_pages

        # Pagination nav - JS handles the links via hash
        nav = (
            '<div id="pagination">'
            '<a id="navFirst">&#x23EE;</a> '
            '<a id="navPrev">&#x25C0;</a> '
            '<input type="number" id="pageInput" value="1" min="1" max="{total}" /> '
            '<span>/ {total}</span> '
            '<a id="navNext">&#x25B6;</a> '
            '<a id="navLast">&#x23ED;</a>'
            '</div>\n'
        ).format(total=total_pages)
        grid_markup = nav + grid_markup

    blob = f'<script>window.GRID_DATA={json.dumps(data)};</script>'

    page = TEMPLATE.replace("{{GRID_MARKUP}}", grid_markup + blob)
    if inline_assets:
        css = (ASSET_DIR / "viewer.css").read_text(encoding="utf-8")
        js = (ASSET_DIR / "viewer.js").read_text(encoding="utf-8")
        page = page.replace(
            '<link rel="stylesheet" href="viewer.css" />', f'<style>\n{css}\n</style>'
        ).replace(
            '<script src="viewer.js"></script>', f'<script>\n{js}\n</script>')
    return page


DEFAULT_PAGE_SIZE = 128


# ------------------------------------------------------------------ API wrappers
def write_html(html_path, entries, file_client, page_size=DEFAULT_PAGE_SIZE):
    """Write entries to a single HTML file with hash-based pagination.

    All entries are in one file. If entries exceed page_size, pagination is enabled:
      - Items use data-src (lazy), JS loads media only for the current page
      - Navigation via URL hash: grid.html#page=2
      - Browser back/forward works

    Set page_size=None to disable pagination (all media loads immediately).
    """
    if not entries:
        return
    formatted = [(d, img, f'[{d}] {name}') for d, img, name in entries]
    content = grid_html(formatted, inline_assets=True, page_size=page_size)
    file_client.put_text(content, html_path)


# ------------------------------------------------------------------ CLI merger
def parse_html(uri: str):
    """Parse HTML file and return all entries."""
    html_text = FileClient.infer_client(uri=uri).get_text(uri)
    soup = BeautifulSoup(html_text, 'html.parser')
    output = []
    for it in soup.select('.item'):
        media = it.find(['img', 'video'])
        if not media:
            continue
        # Support both eager (src) and lazy (data-src) loading
        src = media.get('src') or media.get('data-src')
        if not src:
            continue
        src = _make_abs(uri, src)
        cap = (it.find('textarea') or {}).text.strip()
        data_id = it.get('data-idx', str(len(output)))
        output.append((data_id, src, cap))
    return output
