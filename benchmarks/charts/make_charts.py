#!/usr/bin/env python3
"""Draw the Qwen3.8-27B EXL3 benchmark charts from the committed evidence.

The point of generating these rather than drawing them by hand is that a chart
which restates a measurement can drift from it silently. Every series here is
read out of a file under `docs/bench-evidence/`, so a figure that disagrees with
the record is a parse error and not a quiet lie.

Two files come out per chart, `-light.svg` and `-dark.svg`, because GitHub picks
between them with `<picture>` and `prefers-color-scheme` and a standalone SVG in
an `<img>` cannot reach the page's own colours.

Usage:
    python3 benchmarks/charts/make_charts.py [--out docs/benchmarks/assets]

Standard library only. No network, no GPU.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

VARIADIC = REPO / "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/report.md"
HEADTOHEAD = REPO / "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/results.txt"

# Palette. Validated for CVD separation and contrast against each mode's own
# surface before use: worst adjacent pair reads dE 14.9 (protan) light and 17.4
# (deutan) dark, both above the 8 floor, and the normal-vision floor is 21.8 and
# 19.3 against a required 15.
THEMES = {
    "light": dict(ours="#b4530a", ref="#1f6f9c", ink="#111a20", mute="#68798a",
                  grid="#c9d2d9", face="#ffffff"),
    "dark": dict(ours="#cf7a34", ref="#3d92c8", ink="#e6ebf1", mute="#7b8997",
                 grid="#33404e", face="#161d26"),
}

MONO = ("ui-monospace,SFMono-Regular,Menlo,Consolas,"
        "'DejaVu Sans Mono',monospace")


def read_pipe_table(text: str, header_contains: str) -> list[dict[str, str]]:
    """Return the rows of the first pipe table whose header holds a substring."""
    rows: list[dict[str, str]] = []
    cols: list[str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if cols:
                break
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if cols is None:
            if header_contains in line:
                cols = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if len(cells) == len(cols):
            rows.append(dict(zip(cols, cells)))
    return rows


def variadic_series() -> tuple[dict[int, list[float]], dict[int, list[float]],
                               dict[int, list[float]], dict[int, list[float]]]:
    """Throughput and p95 TTFT per concurrency, per arm, from the leg table."""
    if not VARIADIC.exists():
        sys.exit(f"missing evidence: {VARIADIC}")
    rows = read_pipe_table(VARIADIC.read_text(), "out tok/s")
    ours_tp: dict[int, list[float]] = {}
    theirs_tp: dict[int, list[float]] = {}
    ours_p95: dict[int, list[float]] = {}
    theirs_p95: dict[int, list[float]] = {}
    for r in rows:
        if r.get("publishable") != "yes":
            continue
        c = int(r["c"])
        tp, p95 = float(r["out tok/s"]), float(r["p95 TTFT ms"])
        if r["arm"] == "OURS":
            ours_tp.setdefault(c, []).append(tp)
            ours_p95.setdefault(c, []).append(p95)
        elif r["arm"] == "THEIRS":
            theirs_tp.setdefault(c, []).append(tp)
            theirs_p95.setdefault(c, []).append(p95)
    if not ours_tp or not theirs_tp:
        sys.exit("variadic report parsed but held no publishable legs")
    return ours_tp, theirs_tp, ours_p95, theirs_p95


def headtohead_verdicts() -> dict[str, tuple[float, float]]:
    """The two VERDICT lines the head-to-head job wrote."""
    if not HEADTOHEAD.exists():
        sys.exit(f"missing evidence: {HEADTOHEAD}")
    out: dict[str, tuple[float, float]] = {}
    for line in HEADTOHEAD.read_text().splitlines():
        m = re.match(r"VERDICT (\S+(?: \S+)?)\s*:\s*ours ([\d.]+) vs theirs ([\d.]+)", line.strip())
        if m:
            out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    if len(out) < 2:
        sys.exit("head-to-head results.txt held fewer than two VERDICT lines")
    return out


def svg_open(w: int, h: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-labelledby="t d">',
        f"<title id=\"t\">{title}</title><desc id=\"d\">{desc}</desc>",
    ]


def text(x: float, y: float, s: str, fill: str, size: float = 11,
         anchor: str = "middle", weight: str = "400") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-family="{MONO}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{s}</text>')


def line_chart(cs: list[int], ours: list[float], theirs: list[float], th: dict,
               ymax: float, yticks: list[float], ylab: str, title: str,
               desc: str, fmt: str = "{:.1f}") -> str:
    W, H = 640, 300
    L, R, T, B = 92, 40, 34, 74
    px = [L + (W - L - R) * i / max(1, len(cs) - 1) for i in range(len(cs))]

    def py(v: float) -> float:
        return H - B - (H - B - T) * (v / ymax)

    o = svg_open(W, H, title, desc)
    for tk in yticks:
        y = py(tk)
        dash = '' if tk == 0 else ' stroke-dasharray="2 3"'
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" '
                 f'stroke="{th["grid"]}" stroke-width="1"{dash}/>')
        o.append(text(L - 10, y + 4, fmt.format(tk), th["mute"], 11, "end"))
    o.append(f'<text x="22" y="{(T+H-B)/2:.0f}" fill="{th["mute"]}" font-size="11" '
             f'font-family="{MONO}" text-anchor="middle" '
             f'transform="rotate(-90 22 {(T+H-B)/2:.0f})">{ylab}</text>')

    for series, colour, label in ((theirs, th["ref"], "exllamav3"),
                                  (ours, th["ours"], "vllm.cpp")):
        pts = " ".join(f"{x:.1f},{py(v):.1f}" for x, v in zip(px, series))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
        for x, v in zip(px, series):
            o.append(f'<circle cx="{x:.1f}" cy="{py(v):.1f}" r="5" fill="{colour}"/>')
        # Label only the endpoints, so the middle of the plot stays readable.
        # The first label is anchored to the RIGHT of its marker rather than
        # centred on it: centred, it runs into the y-axis tick labels, which sit
        # just left of the plot area.
        for idx in (0, len(series) - 1):
            dy = -12 if colour == th["ours"] else 20
            anchor, dx = ("start", 11) if idx == 0 else ("middle", 0)
            o.append(text(px[idx] + dx, py(series[idx]) + dy, fmt.format(series[idx]),
                          colour, 11.5, anchor, "600"))
        o.append(text(px[-1], py(series[-1]) + (-26 if colour == th["ours"] else 34),
                      label, colour, 11, "middle", "600"))

    for x, c in zip(px, cs):
        o.append(text(x, H - B + 22, str(c), th["mute"], 11))
    o.append(text((L + W - R) / 2, H - B + 44, "concurrent requests", th["mute"], 11))
    o.append("</svg>")
    return "\n".join(o)


def bar_chart(groups: list[tuple[str, float, float]], th: dict, ymax: float,
              yticks: list[float], xlab: str, title: str, desc: str) -> str:
    W, H = 640, 262
    L, R, T, B = 118, 66, 30, 62
    o = svg_open(W, H, title, desc)

    def px(v: float) -> float:
        return L + (W - L - R) * (v / ymax)

    for tk in yticks:
        x = px(tk)
        dash = '' if tk == 0 else ' stroke-dasharray="2 3"'
        o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" '
                 f'stroke="{th["grid"]}" stroke-width="1"{dash}/>')
        o.append(text(x, H - B + 20, f"{tk:g}", th["mute"], 11))
    o.append(text((L + W - R) / 2, H - B + 42, xlab, th["mute"], 11))

    gh, bh, gap = 60, 21, 4
    for gi, (name, ours, theirs) in enumerate(groups):
        top = T + 14 + gi * gh
        o.append(text(L - 12, top + 14, name, th["mute"], 11, "end"))
        for bi, (v, colour, who) in enumerate(((ours, th["ours"], "vllm.cpp"),
                                               (theirs, th["ref"], "exllamav3"))):
            y = top + bi * (bh + gap)
            o.append(f'<rect x="{L}" y="{y}" width="{px(v)-L:.1f}" height="{bh}" '
                     f'rx="4" fill="{colour}"/>')
            o.append(text(px(v) + 8, y + bh - 6, f"{v:.2f}", colour, 11.5, "start", "600"))
            if gi == 0:
                o.append(text(W - R + 24, y + bh - 6, who, colour, 10, "start", "600"))
    o.append("</svg>")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/benchmarks/assets")
    args = ap.parse_args()
    out = (REPO / args.out) if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ours_tp, theirs_tp, ours_p95, theirs_p95 = variadic_series()
    cs = sorted(set(ours_tp) & set(theirs_tp))
    o_tp = [sum(ours_tp[c]) / len(ours_tp[c]) for c in cs]
    t_tp = [sum(theirs_tp[c]) / len(theirs_tp[c]) for c in cs]
    o_95 = [sum(ours_p95[c]) / len(ours_p95[c]) for c in cs]
    t_95 = [sum(theirs_p95[c]) / len(theirs_p95[c]) for c in cs]

    verd = headtohead_verdicts()
    key_dec = next(k for k in verd if "decode" in k)
    key_run = next(k for k in verd if "whole" in k)

    written = []
    for mode, th in THEMES.items():
        jobs = [
            (f"variadic-concurrency-{mode}.svg", line_chart(
                cs, o_tp, t_tp, th, 60, [0, 15, 30, 45, 60],
                "aggregate output tok/s",
                "Aggregate throughput against concurrency",
                "vllm.cpp rises with offered load while exllamav3 stays flat.")),
            (f"variadic-ttft-p95-{mode}.svg", line_chart(
                cs, o_95, t_95, th, 200000, [0, 50000, 100000, 150000, 200000],
                "p95 time to first token (ms)",
                "p95 time to first token against concurrency",
                "exllamav3's p95 time to first token grows steeply with load; "
                "vllm.cpp's rises more slowly.",
                fmt="{:.0f}")),
            (f"headtohead-throughput-{mode}.svg", bar_chart(
                [("decode only", *verd[key_dec]), ("whole run", *verd[key_run])],
                th, 60, [0, 15, 30, 45, 60], "tokens per second",
                "Throughput by counting convention",
                "vllm.cpp leads exllamav3 on both counting conventions.")),
        ]
        for name, body in jobs:
            (out / name).write_text(body + "\n")
            written.append(name)

    for n in sorted(written):
        print(f"wrote {args.out}/{n}")
    print(f"\nconcurrency rungs: {cs}")
    print(f"ours   tok/s: {[round(v,2) for v in o_tp]}  (n per rung: "
          f"{[len(ours_tp[c]) for c in cs]})")
    print(f"theirs tok/s: {[round(v,2) for v in t_tp]}  (n per rung: "
          f"{[len(theirs_tp[c]) for c in cs]})")
    print(f"head-to-head: {key_dec} {verd[key_dec]}, {key_run} {verd[key_run]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
