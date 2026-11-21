import argparse
from lakonlab.utils.io_utils import FileClient
from lakonlab.ui.media_viewer.grid_tools import (
    parse_html, grid_html
)


def shortest_distinguishing_tags(uri_a: str, uri_b: str):
    """Return (tag_a, tag_b) – shortest path suffixes that differ."""
    def split(uri):
        parts = [p for p in uri.rstrip('/').split('/') if p]
        if parts:
            parts[-1] = parts[-1].removesuffix('.html')
        return parts

    a_parts, b_parts = split(uri_a), split(uri_b)
    max_len = max(len(a_parts), len(b_parts))

    for n in range(1, max_len + 1):
        tag_a = '/'.join(a_parts[-n:])
        tag_b = '/'.join(b_parts[-n:])
        if tag_a != tag_b:
            return tag_a, tag_b

    # paths identical → fall back to suffix plus differentiator
    return tag_a + "_A", tag_b + "_B"


def main():
    parser = argparse.ArgumentParser(
        description="Interleave two existing grids for side-by-side comparison."
    )
    parser.add_argument("html_a", help="First grid file or URL")
    parser.add_argument("html_b", help="Second grid file or URL")
    parser.add_argument("--tag-a", help="Caption prefix for first grid")
    parser.add_argument("--tag-b", help="Caption prefix for second grid")
    parser.add_argument(
        "-o", "--output", required=True, help="Output HTML path")
    args = parser.parse_args()

    tag_a = args.tag_a
    tag_b = args.tag_b
    if tag_a is None and tag_b is None:
        tag_a, tag_b = shortest_distinguishing_tags(args.html_a, args.html_b)
    elif tag_a is None:
        tag_a = "A"
    elif tag_b is None:
        tag_b = "B"

    ea = [(d, s, f"[{tag_a}] {c}") for d, s, c in parse_html(args.html_a)]
    eb = [(d, s, f"[{tag_b}] {c}") for d, s, c in parse_html(args.html_b)]

    merged = [e for chunk in zip(ea, eb) for e in chunk]
    html = grid_html(
        merged,
        inline_assets=True
    )

    FileClient.infer_client(uri=args.output).put_text(html, args.output)
    print(f"Wrote merged HTML to {args.output}")


if __name__ == "__main__":
    main()
