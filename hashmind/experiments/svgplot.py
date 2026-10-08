"""Dependency-free SVG charts for Phase-4 reports (no matplotlib in the env).

Static files embedded in markdown: per-mark <title> gives a hover tooltip in
browsers; legends are always drawn for >= 2 series; colors follow series
identity in a fixed order.
"""

from __future__ import annotations

from html import escape

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, BG = "#1a1a19", "#6b6a63", "#e4e3dc", "#ffffff"
FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'


def _head(w: int, h: int, title: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT}>',
            f'<rect width="{w}" height="{h}" fill="{BG}"/>',
            f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>']


def _yaxis(out: list[str], x0: float, x1: float, y0: float, y1: float, lo: float, hi: float,
           label: str, fmt: str) -> None:
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = y1 - (v - lo) / (hi - lo) * (y1 - y0)
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">'
                   f'{format(v, fmt)}</text>')
    out.append(f'<text x="14" y="{(y0 + y1) / 2:.0f}" font-size="11" fill="{MUTED}" '
               f'transform="rotate(-90 14 {(y0 + y1) / 2:.0f})" text-anchor="middle">{escape(label)}</text>')


def _legend(out: list[str], names: list[str], x: float, y: float) -> None:
    for i, n in enumerate(names):
        out.append(f'<rect x="{x}" y="{y + i * 18 - 9}" width="10" height="10" rx="2" fill="{SERIES[i]}"/>')
        out.append(f'<text x="{x + 16}" y="{y + i * 18}" font-size="11" fill="{INK}">{escape(n)}</text>')


def grouped_bars(title: str, groups: list[str], series: list[str], values: list[list[float | None]],
                 errors: list[list[float]] | None = None, ylabel: str = "", fmt: str = ".0%",
                 lo: float | None = None) -> str:
    """values[s][g]."""
    w, h = 760, 360
    x0, x1, y0, y1 = 64, 560, 44, 300
    flat = [v for row in values for v in row if v is not None]
    hi = max(flat) * 1.05 if flat else 1
    lo = min(0.0, min(flat)) if lo is None else lo
    out = _head(w, h, title)
    _yaxis(out, x0, x1, y0, y1, lo, hi, ylabel, fmt)
    gw = (x1 - x0) / len(groups)
    bw = min(22, (gw - 12) / len(series) - 2)
    for gi, g in enumerate(groups):
        cx = x0 + gw * (gi + 0.5)
        for si, s in enumerate(series):
            v = values[si][gi]
            if v is None:
                continue
            bx = cx - (len(series) * (bw + 2)) / 2 + si * (bw + 2)
            top = y1 - (v - lo) / (hi - lo) * (y1 - y0)
            e = errors[si][gi] if errors else 0
            out.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(y1 - top, 0):.1f}" '
                       f'rx="2" fill="{SERIES[si]}"><title>{escape(s)} / {escape(g)}: {format(v, fmt)}'
                       f'{" ± " + format(e, fmt) if e else ""}</title></rect>')
            if e:
                ey0 = y1 - (v - e - lo) / (hi - lo) * (y1 - y0)
                ey1 = y1 - (v + e - lo) / (hi - lo) * (y1 - y0)
                out.append(f'<line x1="{bx + bw / 2:.1f}" x2="{bx + bw / 2:.1f}" y1="{ey0:.1f}" y2="{ey1:.1f}" '
                           f'stroke="{INK}" stroke-width="1.5"/>')
        out.append(f'<text x="{cx:.1f}" y="{y1 + 16}" font-size="11" text-anchor="middle" fill="{INK}">'
                   f'{escape(g)}</text>')
    _legend(out, series, 576, 60)
    out.append("</svg>")
    return "\n".join(out)


def lines(title: str, xs: list[float], series: dict[str, list[float | None]], xlabel: str = "", ylabel: str = "",
          fmt: str = ".0%", logx: bool = True) -> str:
    import math

    w, h = 760, 360
    x0, x1, y0, y1 = 64, 560, 44, 300
    flat = [v for vs in series.values() for v in vs if v is not None]
    lo, hi = min(flat) * 0.95, max(flat) * 1.05
    tx = (lambda v: math.log2(v)) if logx else (lambda v: v)
    a, b = tx(min(xs)), tx(max(xs))
    X = lambda v: x0 + (tx(v) - a) / (b - a or 1) * (x1 - x0)  # noqa: E731
    Y = lambda v: y1 - (v - lo) / (hi - lo or 1) * (y1 - y0)  # noqa: E731
    out = _head(w, h, title)
    _yaxis(out, x0, x1, y0, y1, lo, hi, ylabel, fmt)
    for v in xs:
        out.append(f'<text x="{X(v):.1f}" y="{y1 + 16}" font-size="11" text-anchor="middle" fill="{INK}">{v:g}</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 34}" font-size="11" text-anchor="middle" fill="{MUTED}">'
               f'{escape(xlabel)}</text>')
    for si, (name, vs) in enumerate(series.items()):
        pts = [(X(x), Y(v), v, x) for x, v in zip(xs, vs) if v is not None]
        out.append(f'<polyline fill="none" stroke="{SERIES[si]}" stroke-width="2" points="'
                   + " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts) + '"/>')
        for px, py, v, x in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{SERIES[si]}" stroke="{BG}" '
                       f'stroke-width="2"><title>{escape(name)} @ {x:g}: {format(v, fmt)}</title></circle>')
    _legend(out, list(series), 576, 60)
    out.append("</svg>")
    return "\n".join(out)


def scatter(title: str, series: dict[str, list[tuple[float, float, str]]], xlabel: str, ylabel: str,
            xfmt: str = ".0%", yfmt: str = ".0%", logx: bool = False) -> str:
    import math

    w, h = 760, 380
    x0, x1, y0, y1 = 64, 560, 44, 316
    pts = [p for ps in series.values() for p in ps]
    tx = (lambda v: math.log10(max(v, 1e-9))) if logx else (lambda v: v)
    xa, xb = min(tx(p[0]) for p in pts), max(tx(p[0]) for p in pts)
    pad = (xb - xa) * 0.05 or 1
    xa, xb = xa - pad, xb + pad
    ylo, yhi = min(p[1] for p in pts), max(p[1] for p in pts)
    ylo, yhi = ylo - (yhi - ylo) * 0.05 - 1e-6, yhi + (yhi - ylo) * 0.05 + 1e-6
    X = lambda v: x0 + (tx(v) - xa) / (xb - xa) * (x1 - x0)  # noqa: E731
    Y = lambda v: y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)  # noqa: E731
    out = _head(w, h, title)
    _yaxis(out, x0, x1, y0, y1, ylo, yhi, ylabel, yfmt)
    for i in range(5):
        tv = xa + (xb - xa) * i / 4
        v = 10 ** tv if logx else tv
        out.append(f'<text x="{x0 + (x1 - x0) * i / 4:.1f}" y="{y1 + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{MUTED}">{format(v, xfmt)}</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 34}" font-size="11" text-anchor="middle" fill="{MUTED}">'
               f'{escape(xlabel)}</text>')
    for si, (name, ps) in enumerate(series.items()):
        for x, y, lab in ps:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5" fill="{SERIES[si]}" stroke="{BG}" '
                       f'stroke-width="2"><title>{escape(lab)}: x={format(x, xfmt)}, y={format(y, yfmt)}'
                       f'</title></circle>')
    _legend(out, list(series), 576, 60)
    out.append("</svg>")
    return "\n".join(out)
