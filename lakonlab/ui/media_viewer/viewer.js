/* viewer.js
   =========
   Light-box / zoom / keyboard logic for the result grid.
   Expects:
     • the page to contain thumbnails inside elements with class="item"
       and data-idx attributes that match the order of
         window.GRID_DATA.sources  &  window.GRID_DATA.captions
     • the CSS rules from viewer.css (same selectors as before)
*/

const INITIAL_SCALE_WHEN_FITS = 1.5;

/* ----------------------------------------------------------
   1.  Data from Python-generated <script> snippet
---------------------------------------------------------- */
const {
    sources,
    captions
} = window.GRID_DATA;

/* ----------------------------------------------------------
   2.  Build / cache overlay DOM elements
---------------------------------------------------------- */
const overlay = document.createElement('div');
overlay.id = 'overlay';
overlay.innerHTML = `
  <div id="container"></div>
  <div id="caption"></div>
`;
document.body.appendChild(overlay);

const container = overlay.querySelector('#container');
const captionEl = overlay.querySelector('#caption');

/* ----------------------------------------------------------
   3.  Runtime state
---------------------------------------------------------- */
let idx = 0;
let isZoomed = false;
let canZoom = false; // false for videos
let media = null; // <img> | <video>
let baseWidth = 0; // natural width for wheel-zoom
let scale = 1;

/* ----------------------------------------------------------
   4.  Helpers
---------------------------------------------------------- */
const mediaTag = src =>
    src.toLowerCase().endsWith('.mp4') ?
    `<video src="${src}" controls autoplay></video>` :
    `<img src="${src}" alt="preview">`;

/* Center or pin content when zoomed */
function adjustAlignment() {
    if (!isZoomed) {
        container.style.alignSelf = '';
        overlay.classList.remove('scrollY');
        return;
    }
    const hOverflow = overlay.scrollWidth > overlay.clientWidth;
    const vOverflow = overlay.scrollHeight > overlay.clientHeight;

    container.style.alignSelf = hOverflow ? 'flex-start' : '';
    overlay.classList.toggle('scrollY', vOverflow);
}

/* ----------------------------------------------------------
   5.  Open / close / zoom
---------------------------------------------------------- */
function show(i) {
    idx = (i + sources.length) % sources.length;

    container.innerHTML = mediaTag(sources[idx]);
    captionEl.textContent = captions[idx];
    overlay.style.display = 'flex';
    document.body.style.overflow = 'hidden'; // lock background scroll

    media = container.firstElementChild;
    canZoom = media.tagName === 'IMG';
    isZoomed = false;
    scale = 1;

    media.style.width = '';
    media.classList.remove('zoomed');
    overlay.classList.remove('scrollY');
    container.style.alignSelf = '';

    media.onclick = canZoom ? toggleZoom : null;
    adjustAlignment();
}

function hide() {
    overlay.style.display = 'none'; // make it invisible
    document.body.style.overflow = '';
    isZoomed = false;
    /* container will be overwritten on next show() call */
}

function toggleZoom() {
    if (!canZoom) return; // ignore for videos

    isZoomed = !isZoomed;
    media.classList.toggle('zoomed', isZoomed);

    if (isZoomed) {
        baseWidth = media.naturalWidth;
        media.style.width = (baseWidth * INITIAL_SCALE_WHEN_FITS) + 'px';
        overlay.scrollTop = overlay.scrollLeft = 0;
    } else {
        media.style.width = '';
    }
    adjustAlignment();
}

/* ----------------------------------------------------------
   6.  Event wiring
---------------------------------------------------------- */
/* Thumbnail clicks – activate only when the media itself is clicked */
document.querySelector('.grid').addEventListener('click', e => {
    if (!e.target.matches('img, video')) return; // ignore textarea, padding, …
    const idx = +e.target.closest('.item').dataset.idx; // ascend to the card
    show(idx);
});

/* Keyboard */
document.addEventListener('keydown', e => {
    if (overlay.style.display !== 'flex') return;

    /* Esc — exit zoom first, then close overlay */
    if (e.key === 'Escape') {
        e.preventDefault(); // ← stop “Stop loading” in Chrome/Firefox
        if (isZoomed) toggleZoom();
        else hide();
        return;
    }

    /* Enter toggles zoom for images */
    if (e.key === 'Enter' && canZoom) {
        toggleZoom();
        return;
    }

    /* Arrow navigation disabled while zoomed */
    if (isZoomed) return;
    if (e.key === 'ArrowRight') show(idx + 1);
    else if (e.key === 'ArrowLeft') show(idx - 1);
});

/* Click backdrop to close */
overlay.addEventListener('click', e => {
    if (e.target === overlay) hide();
});

/* Ctrl + wheel zoom for images only */
overlay.addEventListener('wheel', e => {
    if (!canZoom || !e.ctrlKey) return;

    if (!isZoomed) toggleZoom(); // auto-enter zoom
    e.preventDefault();

    scale += (e.deltaY < 0 ? 0.1 : -0.1);
    scale = Math.max(0.2, Math.min(5, scale));
    media.style.width = (baseWidth * scale) + 'px';
    adjustAlignment();
}, {
    passive: false
});

/* Keep centering correct on resize / browser zoom */
window.addEventListener('resize', adjustAlignment);

/* ----------------------------------------------------------
   7.  Pagination (hash-based, single file)
---------------------------------------------------------- */
const {pageSize, totalPages} = window.GRID_DATA;

if (pageSize && totalPages > 1) {
    const items = document.querySelectorAll('.item');
    const pageInput = document.getElementById('pageInput');
    const navFirst = document.getElementById('navFirst');
    const navPrev = document.getElementById('navPrev');
    const navNext = document.getElementById('navNext');
    const navLast = document.getElementById('navLast');

    function getPageFromHash() {
        const match = window.location.hash.match(/page=(\d+)/);
        return match ? Math.max(1, Math.min(totalPages, parseInt(match[1]))) : 1;
    }

    function showPage(page) {
        const start = (page - 1) * pageSize;
        const end = start + pageSize;

        items.forEach((item, i) => {
            const visible = i >= start && i < end;
            item.style.display = visible ? '' : 'none';

            const media = item.querySelector('img, video');
            if (!media) return;

            if (visible) {
                // Load: set src from data-src
                if (media.dataset.src && !media.src) {
                    media.src = media.dataset.src;
                }
            } else {
                // Unload: remove src to abort pending loads
                if (media.src) {
                    media.removeAttribute('src');
                }
            }
        });

        // Update UI
        pageInput.value = page;
        navFirst.classList.toggle('disabled', page === 1);
        navPrev.classList.toggle('disabled', page === 1);
        navNext.classList.toggle('disabled', page === totalPages);
        navLast.classList.toggle('disabled', page === totalPages);
    }

    function goToPage(page) {
        page = Math.max(1, Math.min(totalPages, page));
        window.location.hash = 'page=' + page;
    }

    // Navigation handlers
    navFirst.addEventListener('click', () => goToPage(1));
    navPrev.addEventListener('click', () => goToPage(getPageFromHash() - 1));
    navNext.addEventListener('click', () => goToPage(getPageFromHash() + 1));
    navLast.addEventListener('click', () => goToPage(totalPages));

    pageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            goToPage(parseInt(this.value));
        }
    });

    // React to hash changes (back/forward buttons)
    window.addEventListener('hashchange', () => showPage(getPageFromHash()));

    // Initial page load
    showPage(getPageFromHash());
}
