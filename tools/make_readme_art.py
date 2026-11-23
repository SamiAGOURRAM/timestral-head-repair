"""Draws the README's illustrations (banner, key-number strip, chapter dividers) as SVG.

Pure drawing, no model runs and no data: the series in the banner is an illustration of the
failure (a dip that repeats, lost by the published model after about 512 steps ahead and kept
by the head edit), not a forecast. Every number printed comes from the README's results.

Run: python tools/make_readme_art.py   (writes assets/*.svg)
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "assets"

BG, LINE = "#0f1216", "#262b33"
INK, INK2, MUTED = "#f2f1ec", "#b9b7b0", "#7d7b75"
ORANGE, GREY, GREEN, RED = "#eb6834", "#8c8a84", "#1baf7a", "#e5484d"
SANS = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', SFMono-Regular, Consolas, 'Liberation Mono', monospace"

STYLE = f"""<style>
  .sans {{ font-family: {SANS}; }}
  .mono {{ font-family: {MONO}; }}
  .draw {{ stroke-dasharray: 3000; animation: draw 3.2s ease-out 0.4s both; }}
  .fade {{ animation: fade 0.8s ease-out 2.8s both; }}
  @keyframes draw {{ from {{ stroke-dashoffset: 3000; }} to {{ stroke-dashoffset: 0; }} }}
  @keyframes fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  @media (prefers-reduced-motion: reduce) {{ .draw, .fade {{ animation: none; }} }}
</style>"""


def path(xs, ys) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def series(x0, x1, level, depth, period, width, phase, noise, rng, keep_until=None, step=3.0):
    """A level with a dip every `period` px; after `keep_until` (px) the dips are gone (flat line)."""
    xs = np.arange(x0, x1 + 0.1, step)
    in_dip = ((xs - phase) % period) < width
    if keep_until is not None:
        in_dip &= xs < keep_until
    ys = level + depth * in_dip + rng.normal(0, noise, len(xs))
    return xs, ys


def wrench(cx, cy, s, color) -> str:
    """A small open-end wrench, drawn from basic shapes, rotated 45 degrees."""
    return (f'<g transform="translate({cx},{cy}) rotate(-45) scale({s})">'
            f'<rect x="-3.2" y="2" width="6.4" height="22" rx="3.2" fill="{color}"/>'
            f'<circle cx="0" cy="-4" r="9" fill="{color}"/>'
            f'<rect x="-3.4" y="-15" width="6.8" height="10" fill="{BG}"/></g>')


def banner() -> str:
    W, H = 1200, 420
    rng = np.random.default_rng(7)
    now, far = 640, 640 + 0.5 * (1150 - 640)  # "512 steps ahead" = half of a 1024-step forecast
    hx, hy = series(50, now, 330, 34, 57.3, 11, 20, 1.6, rng)
    px, py = series(now, 1150, 330, 34, 57.3, 11, 20, 1.2, rng, keep_until=far + 20)
    ex, ey = series(now, 1150, 330, 34, 57.3, 11, 20, 1.2, rng)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="Illustration: the published model loses a repeating dip about 512 steps ahead; the head edit keeps it">',
         STYLE,
         f'<rect width="{W}" height="{H}" rx="18" fill="{BG}"/>',
         # faint patch grid: the model reads the series in patches of 32 steps
         *[f'<line x1="{x}" y1="250" x2="{x}" y2="380" stroke="{LINE}" stroke-width="1"/>' for x in range(50, 1151, 32)],
         f'<rect x="{now}" y="236" width="{1150 - now}" height="152" fill="{ORANGE}" opacity="0.05"/>',
         f'<line x1="{now}" y1="236" x2="{now}" y2="388" stroke="{INK2}" stroke-width="1.5"/>',
         f'<line x1="{far}" y1="236" x2="{far}" y2="388" stroke="{ORANGE}" stroke-width="1.5" stroke-dasharray="4 5"/>',
         f'<text x="{now - 10}" y="402" text-anchor="end" class="sans" font-size="13" fill="{MUTED}">history</text>',
         f'<text x="{now + 10}" y="402" class="sans" font-size="13" fill="{MUTED}">forecast, in one pass</text>',
         f'<text x="{far + 8}" y="252" class="sans" font-size="13" fill="{ORANGE}">512 steps ahead</text>',
         f'<path d="{path(hx, hy)}" fill="none" stroke="{INK}" stroke-width="2" stroke-linejoin="round"/>',
         f'<path class="draw" d="{path(ex, ey)}" fill="none" stroke="{ORANGE}" stroke-width="2.6" stroke-linejoin="round"/>',
         f'<path d="{path(px, py)}" fill="none" stroke="#c9c7c0" stroke-width="2.2" stroke-dasharray="6 5" stroke-linejoin="round"/>',
         # legend
         f'<line x1="852" y1="64" x2="884" y2="64" stroke="#c9c7c0" stroke-width="2.4" stroke-dasharray="6 5"/>',
         f'<text x="894" y="69" class="sans" font-size="15" fill="{INK2}">published: the pattern is lost</text>',
         f'<line x1="852" y1="94" x2="884" y2="94" stroke="{ORANGE}" stroke-width="3"/>',
         f'<text x="894" y="99" class="sans" font-size="15" fill="{INK}">head edit: the pattern is kept</text>',
         # title block
         wrench(70, 74, 1.25, ORANGE),
         f'<text x="100" y="66" class="mono" font-size="14" letter-spacing="3" fill="{ORANGE}">WEIGHT-SPACE REPAIR</text>',
         f'<text x="100" y="86" class="mono" font-size="14" letter-spacing="3" fill="{MUTED}">TIMESTRAL-5M · HEAD 5.1</text>',
         f'<text x="50" y="150" class="sans" font-size="40" font-weight="700" fill="{INK}">One attention head was looking</text>',
         f'<text x="50" y="198" class="sans" font-size="40" font-weight="700" fill="{INK}">in the wrong place.</text>',
         f'<text x="452" y="198" class="sans fade" font-size="40" font-weight="700" fill="{ORANGE}">We moved it.</text>',
         "</svg>"]
    return "\n".join(s)


def stats() -> str:
    tiles = [("6,144", "weights changed", "0.12 % of the model"),
             ("0", "gradients", "edited by reasoning"),
             ("82 %", "of each distant dip drawn", "published model: none"),
             ("22–54 %", "less distant error", "on real ocean tides"),
             ("0.999", "benchmark error ratio", "97 real tasks: unchanged")]
    W, H, gap = 1200, 150, 12
    w = (W - gap * (len(tiles) - 1)) / len(tiles)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="Key numbers: {"; ".join(f"{a} {b} ({c})" for a, b, c in tiles)}">', STYLE]
    for i, (big, label, sub) in enumerate(tiles):
        x = i * (w + gap)
        s += [f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{H}" rx="14" fill="{BG}"/>',
              f'<rect x="{x:.1f}" y="0" width="5" height="{H}" rx="2" fill="{ORANGE if i != 1 else GREEN}"/>',
              f'<text x="{x + 24:.1f}" y="62" class="mono" font-size="38" font-weight="700" fill="{INK}">{big}</text>',
              f'<text x="{x + 24:.1f}" y="96" class="sans" font-size="15" fill="{INK2}">{label}</text>',
              f'<text x="{x + 24:.1f}" y="120" class="sans" font-size="13" fill="{MUTED}">{sub}</text>']
    s.append("</svg>")
    return "\n".join(s)


def patches(x0, y, n, size, gap, filled_until) -> list[str]:
    """A row of patch squares; the ones before `filled_until` hold data (history), the rest are empty (future)."""
    out = []
    for i in range(n):
        x = x0 + i * (size + gap)
        if i < filled_until:
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="2" fill="{INK2}" opacity="0.75"/>')
        else:
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="2" fill="none" stroke="{MUTED}" stroke-width="1.2"/>')
    return out


def arc(x_from, x_to, y, height, color, dashed=False) -> str:
    mid = (x_from + x_to) / 2
    d = ' stroke-dasharray="4 4"' if dashed else ""
    return (f'<path d="M{x_from},{y} Q{mid},{y - height} {x_to},{y}" fill="none" stroke="{color}" stroke-width="2.4"{d}/>'
            f'<circle cx="{x_to}" cy="{y}" r="4.5" fill="{color}"/>')


def chapter(num: str, name: str, line: str, motif: list[str], accent: str = ORANGE) -> str:
    W, H = 1200, 120
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="Chapter {num}: {name}. {line}">', STYLE,
        f'<rect width="{W}" height="{H}" rx="16" fill="{BG}"/>',
        f'<text x="36" y="78" class="mono" font-size="52" font-weight="700" fill="{accent}">{num}</text>',
        f'<line x1="124" y1="30" x2="124" y2="90" stroke="{LINE}" stroke-width="2"/>',
        f'<text x="148" y="56" class="mono" font-size="15" letter-spacing="4" fill="{accent}">{name.upper()}</text>',
        f'<text x="148" y="86" class="sans" font-size="20" fill="{INK}">{line}</text>',
        *motif, "</svg>"])


def chapters() -> dict:
    rng = np.random.default_rng(3)
    out = {}
    # 01 symptom: dips that fade into a flat line
    xs, ys = series(760, 1160, 66, 22, 40, 8, 6, 0.8, rng, keep_until=960)
    out["01_symptom"] = chapter("01", "Symptom", "The forecast forgets the cycle about 500 steps ahead.", [
        f'<line x1="960" y1="22" x2="960" y2="100" stroke="{ORANGE}" stroke-width="1.4" stroke-dasharray="4 4"/>',
        f'<path d="{path(xs, ys)}" fill="none" stroke="{GREY}" stroke-width="2.2"/>'])
    # 02 diagnosis: a future patch reads 11 back and lands on another empty future patch
    # 36 patches, the first 4 hold history; patch 35 reads 11 back (patch 24, future) or 32 back (patch 3, history)
    sz, gp, x0, y, n = 8, 2.6, 776, 82, 36
    xs_ = [x0 + i * (sz + gp) + sz / 2 for i in range(n)]
    out["02_diagnosis"] = chapter("02", "Diagnosis", "One head always reads 11 patches back, into empty future.", [
        *patches(x0, y, n, sz, gp, 4),
        arc(xs_[35], xs_[24], y - 2, 50, RED, dashed=True),
        f'<text x="{xs_[24]}" y="{y + 26}" text-anchor="middle" class="mono" font-size="12" fill="{RED}">11 back: empty</text>'], accent=RED)
    # 03 repair: the same patch now reads 32 back and lands on real history
    out["03_repair"] = chapter("03", "Repair", "6,144 weights rewritten: it now reads 32 back, into real history.", [
        *patches(x0, y, n, sz, gp, 4),
        arc(xs_[35], xs_[3], y - 2, 80, ORANGE),
        f'<text x="{x0}" y="{y + 26}" class="mono" font-size="12" fill="{ORANGE}">32 back: history</text>',
        wrench(1160, 30, 0.8, ORANGE)])
    # 04 tests: three check rows
    checks = ["fine-tunes", "real benchmark", "ocean tides"]
    m = []
    for i, c in enumerate(checks):
        cx = 790 + i * 130
        m += [f'<circle cx="{cx}" cy="58" r="16" fill="none" stroke="{GREEN}" stroke-width="2.4"/>',
              f'<path d="M{cx - 7},{58} l5,6 l10,-12" fill="none" stroke="{GREEN}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>',
              f'<text x="{cx}" y="98" text-anchor="middle" class="sans" font-size="13" fill="{INK2}">{c}</text>']
    out["04_tests"] = chapter("04", "Tests", "Against fine-tuning, a 97-task benchmark and real tides.", m, accent=GREEN)
    # 05 fine print: a cost curve over history length
    hx = np.linspace(780, 1160, 80)
    hy = 80 - 40 * np.exp(-((hx - 880) / 45) ** 2)  # flat, with a bump where histories are short
    out["05_fine_print"] = chapter("05", "Fine print", "Why 32, who finds the head, what it costs, and bigger models.", [
        f'<line x1="780" y1="80" x2="1160" y2="80" stroke="{LINE}" stroke-width="1.5"/>',
        f'<path d="{path(hx, hy)}" fill="none" stroke="{INK2}" stroke-width="2.2"/>',
        f'<text x="880" y="104" text-anchor="middle" class="sans" font-size="12" fill="{MUTED}">short histories</text>'], accent=INK2)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = {"banner": banner(), "key_numbers": stats(), **{f"chapter_{k}": v for k, v in chapters().items()}}
    for name, svg in files.items():
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8", newline="\n")
        print(f"wrote assets/{name}.svg")


if __name__ == "__main__":
    main()
