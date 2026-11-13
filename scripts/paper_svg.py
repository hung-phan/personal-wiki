"""Paper-style SVG helpers for wiki diagrams (cream paper, typewriter titles, red hand notes, dark mode).

Usage: import paper_svg as ps; ps.OUT = "wiki/<cat>/assets"; ps.svg("name.svg", 960, 560, "".join(parts))
"""
import os
OUT = "."

STYLE = """
  <style>
    .bg    { fill: #f7f0e1; }
    .ink   { fill: #2b2b2b; }
    .mut   { fill: #6b6256; font-size: 11px; }
    .h1    { fill: #2b2b2b; font-family: 'Courier New', Courier, 'DejaVu Sans Mono', monospace; font-size: 24px; font-weight: 700; }
    .h2    { fill: #2b2b2b; font-family: 'Courier New', Courier, 'DejaVu Sans Mono', monospace; font-size: 15px; font-weight: 700; }
    .kick  { fill: #6b6256; font-family: 'Courier New', Courier, 'DejaVu Sans Mono', monospace; font-size: 13px; }
    .hand  { fill: #b3261e; font-family: 'Bradley Hand', 'Segoe Print', 'Comic Sans MS', 'DejaVu Sans', cursive; font-size: 13px; }
    .handk { fill: #2b2b2b; font-family: 'Bradley Hand', 'Segoe Print', 'Comic Sans MS', 'DejaVu Sans', cursive; font-size: 13px; }
    .num   { fill: #2b2b2b; font-family: ui-monospace, Menlo, 'DejaVu Sans Mono', monospace; font-size: 11px; }
    .numb  { fill: #2b2b2b; font-family: ui-monospace, Menlo, 'DejaVu Sans Mono', monospace; font-size: 11px; font-weight: 700; }
    .numf  { fill: #a39a8c; font-family: ui-monospace, Menlo, 'DejaVu Sans Mono', monospace; font-size: 11px; }
    .red   { fill: #b3261e; }
    .uline { stroke: #b3261e; stroke-width: 2; }
    .ln    { stroke: #2b2b2b; stroke-width: 1.4; fill: none; }
    .lnr   { stroke: #b3261e; stroke-width: 1.8; fill: none; }
    .lnd   { stroke: #6b6256; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; }
    .cell  { fill: #fbf7ee; stroke: #8a8174; stroke-width: 0.8; }
    .grey  { fill: #e3dccd; stroke: #8a8174; stroke-width: 0.8; }
    .pur   { fill: #e7dcf3; stroke: #7b5ea7; stroke-width: 1; }
    .grn   { fill: #e1edd5; stroke: #5f8a3a; stroke-width: 1; }
    .yel   { fill: #fbe9c4; stroke: #b8860b; stroke-width: 1; }
    .blu   { fill: #dce8f5; stroke: #3d6fa8; stroke-width: 1; }
    .pnk   { fill: #f6d9d4; stroke: #b3261e; stroke-width: 1; }
    .hi    { fill: none; stroke: #b3261e; stroke-width: 2.2; }
    .note  { fill: #fdf9f0; stroke: #8a8174; stroke-width: 1; stroke-dasharray: 6 4; }
    .bar   { fill: #7b5ea7; }
    .barr  { fill: #b3261e; }
    @media (prefers-color-scheme: dark) {
      .bg { fill: #1f1d1a; }
      .ink, .h1, .h2, .handk, .num, .numb { fill: #ece6d9; }
      .mut, .kick { fill: #b5ab9b; }
      .numf { fill: #6f675c; }
      .hand, .red { fill: #f28b82; }
      .uline, .lnr, .hi { stroke: #f28b82; }
      .ln { stroke: #ece6d9; }
      .lnd { stroke: #b5ab9b; }
      .cell { fill: #2a2723; stroke: #6f675c; }
      .grey { fill: #3a362f; stroke: #6f675c; }
      .note { fill: #26231f; stroke: #8a8174; }
      .pur { fill: #3b2d52; stroke: #b39ddb; }
      .grn { fill: #2c3f22; stroke: #9ccc65; }
      .yel { fill: #4a3a14; stroke: #e0b44c; }
      .blu { fill: #1d3550; stroke: #7fb0e6; }
      .pnk { fill: #4d2420; stroke: #f28b82; }
      .bar { fill: #b39ddb; }
      .barr { fill: #f28b82; }
    }
  </style>
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#8a8174"/>
    </marker>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#c0392b"/>
    </marker>
  </defs>
"""

def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def T(x, y, s, c="ink", a="start", extra=""):
    return f'<text x="{x}" y="{y}" class="{c}" text-anchor="{a}" {extra}>{esc(s)}</text>'
def R(x, y, w, h, c="cell", rx=0): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{c}"/>'
def A(x1, y1, x2, y2, c="ln", m="ar"): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{c}" marker-end="url(#{m})"/>'
def P(d, c="ln", m=None): return f'<path d="{d}" class="{c}"' + (f' marker-end="url(#{m})"' if m else "") + "/>"
def C(x, y, r, c="cell"): return f'<circle cx="{x}" cy="{y}" r="{r}" class="{c}"/>'

def header(kicker, title, W):
    return T(30, 32, kicker, "kick") + T(30, 62, title, "h1") + f'<line x1="30" y1="72" x2="{30 + 15 * len(title)}" y2="72" class="uline"/>'

def matrix(x, y, vals, cw, ch, cls=lambda i, j: "cell", tcls=lambda i, j: "num"):
    out = []
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            out.append(R(x + j * cw, y + i * ch, cw, ch, cls(i, j)))
            out.append(T(x + j * cw + cw / 2, y + i * ch + ch / 2 + 4, v, tcls(i, j), "middle"))
    return "".join(out)

def colvec(x, y, vals, w, ch, c):
    out = [R(x, y, w, ch * len(vals), c, 3)]
    for i, v in enumerate(vals):
        out.append(T(x + w / 2, y + i * ch + ch / 2 + 4, v, "num", "middle"))
    return "".join(out)

def note(x, y, w, h, title, lines, tc="hand"):
    out = [R(x, y, w, h, "note", 6), T(x + 14, y + 22, title, "h2")]
    for i, l in enumerate(lines):
        out.append(T(x + 14, y + 42 + i * 18, l, tc))
    return "".join(out)

def svg(name, W, H, body):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, DejaVu Sans, sans-serif" font-size="13">'
         + STYLE + R(0, 0, W, H, "bg") + body + "</svg>\n")
    open(os.path.join(OUT, name), "w").write(s)

