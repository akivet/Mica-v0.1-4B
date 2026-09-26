"""Render two bench.py logs as a head-to-head 1080p60 video (needs ffmpeg on PATH).

Decision time on screen is each judge's recorded latency; piece motion and line clears are fixed-length animations
around it. With Bahnschrift and Cascadia Mono installed (Windows) the output matches our published videos exactly;
elsewhere DejaVu is used (pass --font-dir to point at other TTFs).

python render.py runs/easy/mica_s11.jsonl runs/easy/kev_s11.jsonl --pieces 100 --gpu "RTX 3090" --out mica_vs_kev.mp4
python render.py A.jsonl B.jsonl --still 40 --out frame.png          (one frame, for checking)
"""
import argparse, bisect, json, math, os, subprocess, sys
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import SHAPES, W, H, rotations  # noqa: E402

FPS, SW, SH = 60, 1920, 1080
CS = 38
INTRO, OUTRO = 2.4, 5.0
BG, EDGE, WELL, GRIDC = (8, 9, 12), (34, 38, 47), (11, 13, 17), (22, 25, 31)
TEXT, DIM, FAINT, RED = (234, 237, 242), (128, 136, 150), (66, 72, 85), (240, 97, 109)
PANEL = (14, 16, 22)
PIECE = {'I': (80, 205, 230), 'O': (240, 206, 80), 'T': (176, 124, 238), 'S': (104, 214, 124), 'Z': (238, 98, 110),
         'J': (80, 128, 238), 'L': (240, 152, 78)}
KNOWN = {'mica': ((126, 240, 200), 'Qwen3.5-4B fine-tune  ·  Q5_K_M  ·  llama.cpp'),
         'kev': ((168, 140, 255), 'Qwen3.5-4B-Base + LoRA  ·  PyTorch'),
         'laya': ((255, 150, 90), 'ModernBERT-large 421M  ·  PyTorch')}
PALETTE = [(255, 150, 90), (168, 140, 255), (110, 180, 255), (255, 120, 170), (240, 206, 80)]

# ------------------------------------------------------------------ fonts
FONT_DIRS = [Path(p) for p in os.environ.get('RENDER_FONT_DIR', '').split(os.pathsep) if p] + [
    Path('C:/Windows/Fonts'), Path.home() / 'AppData/Local/Microsoft/Windows/Fonts', Path('/usr/share/fonts'),
    Path('/usr/local/share/fonts'), Path.home() / '.fonts', Path.home() / '.local/share/fonts', Path('/Library/Fonts'),
    Path.home() / 'Library/Fonts', Path('/System/Library/Fonts')]
try:
    import matplotlib
    FONT_DIRS.append(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf')
except Exception:
    pass


@lru_cache(None)
def find_font(*names):
    for d in FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return str(d / n)
    for d in FONT_DIRS:
        if d.is_dir():
            for n in names:
                hit = next(d.rglob(n), None)
                if hit:
                    return str(hit)
    return None


BOLD = {'Bold', 'SemiBold'}


@lru_cache(None)
def font(size, var='Regular', fam='bahnschrift.ttf'):
    path = find_font(fam)
    if path is None:  # DejaVu fallback (no weight variations)
        path = find_font('DejaVuSansMono-Bold.ttf' if 'Mono' in fam and var in BOLD else 'DejaVuSansMono.ttf' if 'Mono' in fam else
                         'DejaVuSans-Bold.ttf' if var in BOLD else 'DejaVuSans.ttf')
        if path is None:
            return ImageFont.load_default(size)
        return ImageFont.truetype(path, size)
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_name(var)
    except Exception:
        pass
    return f


def mono(size, var='Regular'):
    return font(size, var, 'CascadiaMono.ttf')


# ------------------------------------------------------------------ small helpers
def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_in(p):
    return p * p


def ease_io(p):
    return 3 * p * p - 2 * p * p * p


def mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


SS = 4


@lru_cache(None)
def block(color, size):
    s = size * SS
    im = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    top, bot = mix(color, (255, 255, 255), 0.18), mix(color, (0, 0, 0), 0.22)
    grad = Image.new('RGBA', (s, s))
    gd = ImageDraw.Draw(grad)
    for y in range(s):
        gd.line([(0, y), (s, y)], fill=mix(top, bot, y / s) + (255,))
    mask = Image.new('L', (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([SS, SS, s - SS - 1, s - SS - 1], radius=int(size * 0.16 * SS), fill=255)
    im.paste(grad, (0, 0), mask)
    d.rounded_rectangle([SS * 3, SS * 3, s - SS * 3, SS * 3 + int(size * 0.14 * SS)], radius=int(size * 0.07 * SS),
                        fill=mix(color, (255, 255, 255), 0.45) + (110,))
    return im.resize((size, size), Image.LANCZOS)


@lru_cache(None)
def cap(h, color, alpha=255):
    s = h * SS
    im = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([0, 0, s - 1, s - 1], fill=tuple(color) + (alpha,))
    return im.resize((h, h), Image.LANCZOS)


def pill(img, x, y, w, h, color, alpha=255):
    w = int(round(w))
    if w <= 0:
        return
    if w < h:
        c = cap(h, color, alpha).crop((0, 0, max(1, w), h))
        img.paste(c, (int(x), int(y)), c)
        return
    c = cap(h, color, alpha)
    hl = h // 2
    left, right = c.crop((0, 0, hl, h)), c.crop((hl, 0, h, h))
    img.paste(left, (int(x), int(y)), left)
    if w - h > 0:
        mid = Image.new('RGBA', (w - h, h), tuple(color) + (alpha,))
        img.paste(mid, (int(x) + hl, int(y)), mid)
    img.paste(right, (int(x) + w - (h - hl), int(y)), right)


def rrect_ss(size, radius, fill, outline=None, width=1):
    w, h = size
    im = Image.new('RGBA', (w * SS, h * SS), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * SS - 1, h * SS - 1], radius=radius * SS, fill=fill, outline=outline, width=width * SS)
    return im.resize((w, h), Image.LANCZOS)


def label(d, xy, text, size=17, color=DIM, anchor='la', var='SemiBold', spacing=2):
    f = font(size, var)
    x, y = xy
    if anchor[0] == 'r':
        x -= sum(f.getlength(ch) + spacing for ch in text) - spacing
    for ch in text:
        d.text((x, y), ch, font=f, fill=color, anchor='l' + anchor[1])
        x += f.getlength(ch) + spacing


def shape(piece, ri):
    return rotations(SHAPES[piece])[ri]


def piece_icon(img, piece, ri, cx, cy, cs):
    cells = shape(piece, ri)
    w, h = max(x for x, _ in cells) + 1, max(y for _, y in cells) + 1
    ox, oy = cx - w * cs / 2, cy - h * cs / 2
    b = block(PIECE[piece], cs)
    for x, y in cells:
        img.paste(b, (int(ox + x * cs), int(oy + y * cs)), b)


def strip(img, piece, o, x, cy, c, col):
    """10-column ruler with the option's cells: where the piece lands, at a glance."""
    d = ImageDraw.Draw(img)
    cells = shape(piece, o['ri'])
    rows = max(y for _, y in cells) + 1
    y0 = cy - rows * c / 2
    for gx in range(W):
        d.rectangle([x + gx * c, cy + rows * c / 2 + 3, x + gx * c + c - 2, cy + rows * c / 2 + 4], fill=(40, 44, 54))
    for dx, dy in cells:
        X = x + (o['c'] + dx) * c
        Y = y0 + dy * c
        d.rectangle([X, Y, X + c - 2, Y + c - 2], fill=col)


def sparkline(img, x, y, w, h, vals, acc, scale=None):
    if not vals:
        return
    n = 48
    vals = vals[-n:]
    top = scale or max(max(vals) * 1.15, 1)
    bw = w / n
    d = ImageDraw.Draw(img)
    for i, v in enumerate(vals):
        bh = max(2, h * min(v, top) / top)
        c = acc if i == len(vals) - 1 else (58, 64, 78)
        xx, yy, ww = x + i * bw + 1, y + h - bh, max(2, bw - 3)
        d.rectangle([int(xx), int(yy), int(xx + ww) - 1, int(yy + bh) - 1], fill=c)


def fmt_clock(t):
    return f'{int(t // 60):02d}:{t % 60:04.1f}'


@lru_cache(None)
def glow(w, h, color, strength=0.55, r=22):
    pad = 48
    im = Image.new('RGBA', (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([pad, pad, pad + w, pad + h], radius=r, outline=color + (255,), width=5)
    im = im.filter(ImageFilter.GaussianBlur(16))
    im.putalpha(im.getchannel('A').point(lambda v: int(v * strength)))
    return im


def background():
    img = Image.new('RGB', (SW, SH), BG)
    g = Image.linear_gradient('L').rotate(30, expand=True).resize((SW, SH))
    img = Image.composite(Image.new('RGB', (SW, SH), (19, 22, 31)), img, g.point(lambda v: int(v * 0.55)))
    noise = Image.effect_noise((SW // 2, SH // 2), 16).resize((SW, SH)).convert('RGB')
    img = Image.blend(img, noise, 0.03)
    d = ImageDraw.Draw(img)
    for i in range(0, SW + 600, 110):
        d.line([(i, 0), (i - 500, SH)], fill=(22, 25, 34), width=1)
    return img


# ------------------------------------------------------------------ timeline of one recorded run
class Pace:
    def __init__(self, reveal, hold, slide, drop, lock, flash, collapse):
        self.reveal, self.hold, self.slide, self.drop, self.lock, self.flash, self.collapse = reveal, hold, slide, drop, lock, flash, collapse


def apply_move(grid, piece, cells, cleared):
    g = [row[:] for row in grid]
    for x, y in cells:
        g[y][x] = piece
    keep = [row for i, row in enumerate(g) if i not in cleared]
    return [[0] * W for _ in range(H - len(keep))] + keep


class Run:
    def __init__(self, path, pace, pieces=None, color=None, name=None, spec=None):
        rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
        self.meta = rows[0].get('meta', {})
        key = self.meta.get('judge', 'model')
        acc, sp = KNOWN.get(key.lower(), (None, ''))
        self.accent = color or acc or PALETTE[0]
        self.name = (name or key).upper()
        self.spec = sp if spec is None else spec
        moves = [r for r in rows if 'i' in r]
        self.top_out = any('game_over' in r for r in rows) and (pieces is None or len(moves) < pieces)
        self.moves = moves[:pieces] if pieces else moves
        self.starts, self.phases, t = [], [], 0.0
        for m in self.moves:
            m['opt'] = next(o for o in m['opts'] if o['id'] == m['choice'])
            ph = [('think', max(m['ms'], 0) / 1000), ('reveal', pace.reveal), ('hold', pace.hold), ('slide', pace.slide),
                  ('drop', pace.drop), ('lock', pace.lock)]
            if m['cleared_rows']:
                ph += [('flash', pace.flash), ('collapse', pace.collapse)]
            self.starts.append(t)
            self.phases.append(ph)
            t += sum(d for _, d in ph)
        self.end = t
        last = self.moves[-1]
        self.final_grid = apply_move(last['grid'], last['piece'], last['opt']['cells'], last['cleared_rows'])
        self.lat = [m['ms'] for m in self.moves if m['ms'] > 0]

    def at(self, t):
        """-> (k, phase, progress); k == len(moves) means finished."""
        if t >= self.end:
            return len(self.moves), 'end', min(1.0, (t - self.end) / 0.6)
        k = max(0, bisect.bisect_right(self.starts, t) - 1)
        u = t - self.starts[k]
        for name, d in self.phases[k]:
            if u < d:
                return k, name, (u / d if d > 0 else 1.0)
            u -= d
        return k, 'lock', 1.0

    def lines_at(self, k, phase):
        if k >= len(self.moves):
            return self.moves[-1]['lines']
        m = self.moves[k]
        done = phase in ('flash', 'collapse') or (phase == 'lock' and not m['cleared_rows'])
        return m['lines'] if done and phase == 'collapse' else m['lines'] - len(m['cleared_rows'])


# ------------------------------------------------------------------ board drawing (calm: no candidate fan-out, no strobe)
class View:
    def __init__(self, run, bx, by, cs):
        self.r, self.bx, self.by, self.cs = run, bx, by, cs
        self.acc = run.accent

    def cell_xy(self, x, y):
        return self.bx + x * self.cs, self.by + y * self.cs

    def draw_cells(self, img, cells, piece):
        b = block(PIECE[piece], self.cs)
        for x, y in cells:
            if y <= -1:
                continue
            px, py = self.cell_xy(x, y)
            if y < 0:
                c = b.crop((0, int(-y * self.cs), self.cs, self.cs))
                img.paste(c, (int(px), self.by), c)
            else:
                img.paste(b, (int(px), int(round(py))), b)

    def draw_cells_float(self, img, cells, piece):
        b = block(PIECE[piece], self.cs)
        for x, y in cells:
            if y <= -1:
                continue
            img.paste(b, (int(round(self.bx + x * self.cs)), int(round(self.by + y * self.cs))), b)

    def draw_grid(self, img, grid, shift=None, dim_rows=(), flash_rows=(), flash=0.0, grey=0.0):
        for y in range(H):
            if y in dim_rows:
                continue
            for x in range(W):
                v = grid[y][x]
                if not v:
                    continue
                col = PIECE[v] if not grey else mix(PIECE[v], (60, 64, 72), grey)
                b = block(col, self.cs)
                px, py = self.cell_xy(x, y + (shift[y] if shift else 0))
                img.paste(b, (int(px), int(round(py))), b)
        if flash_rows and flash > 0:
            ov = Image.new('RGBA', (W * self.cs, self.cs), (255, 255, 255, int(235 * flash)))
            for y in flash_rows:
                img.paste(ov, self.cell_xy(0, y), ov)

    def ghost(self, img, cells, a):
        cs = self.cs
        ov = Image.new('RGBA', (W * cs, H * cs), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for x, y in cells:
            d.rounded_rectangle([x * cs + 3, y * cs + 3, (x + 1) * cs - 4, (y + 1) * cs - 4], radius=5,
                                fill=self.acc + (int(40 * a),), outline=self.acc + (int(200 * a),), width=2)
        img.paste(ov, (self.bx, self.by), ov)

    def board(self, img, t):
        r = self.r
        k, ph, p = r.at(t)
        if k >= len(r.moves):
            self.draw_grid(img, r.final_grid, grey=(p if r.top_out else 0.0))
            return k, ph, p
        m = r.moves[k]
        piece, o = m['piece'], m['opt']
        if ph in ('think', 'reveal', 'hold', 'slide', 'drop'):
            self.draw_grid(img, m['grid'])
            a = {'think': 0.0, 'reveal': ease_out(p) * 0.6, 'hold': 0.6, 'slide': 0.6, 'drop': 0.6 * (1 - p)}[ph]
            if a > 0.01:
                self.ghost(img, o['cells'], a)
            if ph in ('think', 'reveal', 'hold'):
                cells0 = shape(piece, 0)
                sc = (W - (max(x for x, _ in cells0) + 1)) // 2
                self.draw_cells(img, [(sc + x, y) for x, y in cells0], piece)
            else:
                rot = shape(piece, o['ri'])
                sc = (W - (max(x for x, _ in rot) + 1)) // 2
                land_top = min(y for _, y in o['cells'])
                if ph == 'slide':
                    xo = sc + (o['c'] - sc) * ease_io(p)
                    self.draw_cells_float(img, [(xo + x, y) for x, y in rot], piece)
                else:
                    self.draw_cells_float(img, [(o['c'] + x, land_top * ease_in(p) + y) for x, y in rot], piece)
        else:
            g = [row[:] for row in m['grid']]
            for x, y in o['cells']:
                g[y][x] = piece
            if ph == 'lock':
                self.draw_grid(img, g)
                ov = Image.new('RGBA', (self.cs, self.cs), (255, 255, 255, int(70 * (1 - p))))
                for x, y in o['cells']:
                    img.paste(ov, self.cell_xy(x, y), ov)
            elif ph == 'flash':
                self.draw_grid(img, g, flash_rows=m['cleared_rows'], flash=0.55 * math.sin(math.pi * p))
            elif ph == 'collapse':
                cl = set(m['cleared_rows'])
                shift = {y: sum(1 for c in cl if c > y) * ease_in(p) for y in range(H)}
                self.draw_grid(img, g, shift=shift, dim_rows=cl)
        return k, ph, p


# ------------------------------------------------------------------ the broadcast frame
class Broadcast:
    def __init__(self, runs, gpu=None, title='TETRIS'):
        self.rs, self.title = runs, title
        self.sub = 'same seed  ·  same pieces' + (f'  ·  one {gpu}' if gpu else '')
        self.k = max(len(m['opts']) for r in runs for m in r.moves)
        bw, bh = W * CS, H * CS
        self.bxs = [SW // 2 - 36 - bw, SW // 2 + 36]
        self.by = 190
        self.vs = [View(r, bx, self.by, CS) for r, bx in zip(runs, self.bxs)]
        self.panels = [(70, self.bxs[0] - 44), (self.bxs[1] + bw + 44, SW - 70)]
        self.game_T = max(r.end for r in runs)
        self.T = INTRO + self.game_T + OUTRO
        allv = sorted(v for r in runs for v in r.lat)
        self.lat_scale = allv[int(len(allv) * 0.98)] * 1.1 if allv else 1
        self.bg = background()
        self.static = self._static()

    def _static(self):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        bw, bh = W * CS, H * CS
        for r, v, (px0, px1) in zip(self.rs, self.vs, self.panels):
            acc = r.accent
            g = glow(bw + 24, bh + 24, acc, 0.45)
            img.paste(g, (v.bx - 12 - 48, self.by - 12 - 48), g)
            fr = rrect_ss((bw + 24, bh + 24), 16, WELL + (255,), mix(EDGE, acc, 0.35) + (255,), 2)
            img.paste(fr, (v.bx - 12, self.by - 12), fr)
            for x in range(1, W):
                d.line([(v.bx + x * CS, self.by), (v.bx + x * CS, self.by + bh - 1)], fill=GRIDC)
            for y in range(1, H):
                d.line([(v.bx, self.by + y * CS), (v.bx + bw - 1, self.by + y * CS)], fill=GRIDC)
            pw = px1 - px0
            sp = rrect_ss((pw, bh + 24), 22, PANEL + (235,), mix((40, 44, 54), acc, 0.35) + (255,), 1)
            img.paste(sp, (px0, self.by - 12), sp)
            band = rrect_ss((pw - 16, 116), 16, mix((20, 22, 28), acc, 0.16) + (255,))
            img.paste(band, (px0 + 8, self.by - 4), band)
            d.text((px0 + 28, self.by + 58), r.name, font=font(52, 'Bold'), anchor='ls', fill=TEXT)
            d.text((px0 + 30, self.by + 92), r.spec, font=font(17), anchor='ls', fill=mix(DIM, acc, 0.25))
            label(d, (px0 + 28, self.by + 138), 'LINES', 13)
            label(d, (px0 + 170, self.by + 138), 'PIECES', 13)
            label(d, (px1 - 110, self.by + 138), 'NEXT', 13)
            label(d, (px0 + 28, self.by + 250), 'DECISION', 13)
            d.line([(px0 + 24, self.by + 596), (px1 - 24, self.by + 596)], fill=(34, 38, 47))
            label(d, (px0 + 28, self.by + 614), 'DECISION TIME', 13)
        return img

    def decision(self, img, d, r, v, k, ph, p, x0, x1, y0):
        """Probability rows; values tween from the previous decision so nothing flashes between moves."""
        n = len(r.moves)
        cur = r.moves[min(k, n - 1)]
        prev = r.moves[k - 1] if 0 < k < n else None
        tw = 1.0 if k >= n else 0.0 if ph == 'think' else ease_io(p) if ph == 'reveal' else 1.0
        acc = v.acc
        order = sorted(cur['opts'], key=lambda o: -cur['probs'].get(o['id'], 0))[:4]
        pv = sorted(prev['probs'].values(), reverse=True)[:4] if prev else [0.0] * 4
        pv += [0.0] * (4 - len(pv))
        rh = 78
        for i, o in enumerate(order):
            y = y0 + i * rh
            pr = cur['probs'].get(o['id'], 0.0)
            val = pv[i] + (pr - pv[i]) * tw
            chosen = o['id'] == cur['choice'] and tw > 0.5
            if chosen:
                a = min(1.0, (tw - 0.5) * 2)
                hl = rrect_ss((x1 - x0 + 20, rh - 10), 12, mix(PANEL, acc, 0.14 * a) + (255,), mix((40, 44, 54), acc, 0.8 * a) + (255,), 2)
                img.paste(hl, (x0 - 10, y - 8), hl)
            strip(img, cur['piece'], o, x0 + 4, y + 18, 6, mix((110, 118, 132), acc, 1.0 if i == 0 else 0.0))
            holes = max(0, o['new_holes'])
            d.text((x0 + 84, y + 20), f"{o['full']} line{'' if o['full'] == 1 else 's'}  ·  {holes} hole{'' if holes == 1 else 's'}",
                   font=font(17), anchor='lm', fill=DIM)
            d.text((x1, y + 20), f'{val * 100:.0f}%', font=mono(24, 'SemiBold'), anchor='rm', fill=acc if chosen else TEXT)
            pill(img, x0, y + 44, x1 - x0, 7, (30, 34, 42))
            pill(img, x0, y + 44, max(7, (x1 - x0) * val), 7, acc if i == 0 else (112, 120, 136))

    def panel(self, img, d, i, t):
        r, v = self.rs[i], self.vs[i]
        px0, px1 = self.panels[i]
        x0, x1 = px0 + 28, px1 - 28
        k, ph, p = r.at(t)
        n = len(r.moves)
        cur = r.moves[min(k, n - 1)]
        pieces = min(n, k + (1 if ph in ('lock', 'flash', 'collapse', 'end') else 0))
        d.text((x0, self.by + 206), str(r.lines_at(k, ph)), font=font(64, 'Bold'), anchor='ls', fill=TEXT)
        d.text((px0 + 170, self.by + 206), str(pieces), font=font(64, 'Bold'), anchor='ls', fill=mix(TEXT, DIM, 0.3))
        if k < n:
            piece_icon(img, cur['next'], 0, px1 - 76, self.by + 180, 20)
        self.decision(img, d, r, v, k, ph, p, x0, x1, self.by + 280)
        if k < n:
            shown = cur['ms'] * (min(1.0, p) if ph == 'think' else 1.0)
            hist = [m['ms'] for m in r.moves[:k + (0 if ph == 'think' else 1)] if m['ms'] > 0]
        else:
            shown, hist = (r.lat[-1] if r.lat else 0), r.lat
        big = font(72, 'Bold')
        d.text((x0, self.by + 700), f'{shown:.0f}', font=big, anchor='ls', fill=TEXT)
        d.text((x0 + big.getlength(f'{shown:.0f}') + 8, self.by + 700), 'ms', font=font(26, 'SemiLight'), anchor='ls', fill=DIM)
        if hist:
            d.text((x0, self.by + 738), f'avg {sum(hist) / len(hist):.0f} ms', font=font(17), anchor='ls', fill=DIM)
        sparkline(img, x0 + 190, self.by + 640, x1 - x0 - 190, 64, hist, v.acc, scale=self.lat_scale)

    def scoreboard(self, img, d, t):
        cx = SW // 2
        la = [r.lines_at(*r.at(t)[:2]) for r in self.rs]
        box = rrect_ss((420, 104), 20, (12, 14, 19, 240), (44, 48, 58, 255), 1)
        img.paste(box, (cx - 210, 40), box)
        for i, r in enumerate(self.rs):
            sgn = -1 if i == 0 else 1
            d.text((cx + sgn * 130, 76), r.name, font=font(22, 'Bold'), anchor='mm', fill=r.accent)
            d.text((cx + sgn * 130, 116), str(la[i]), font=font(44, 'Bold'), anchor='mm', fill=TEXT)
        d.text((cx, 80), 'LINES', font=font(14, 'SemiBold'), anchor='mm', fill=FAINT)
        d.text((cx, 114), fmt_clock(min(t, self.game_T)), font=mono(18), anchor='mm', fill=DIM)
        d.text((70, 92), self.title, font=font(30, 'Bold'), anchor='ls', fill=TEXT)
        d.text((72, 122), self.sub, font=font(17), anchor='ls', fill=DIM)
        d.text((SW - 70, 92), 'REPLAY', font=font(16, 'Bold'), anchor='rs', fill=DIM)
        d.text((SW - 70, 122), 'real decision times', font=font(17), anchor='rs', fill=FAINT)

    def race(self, img, d, t):
        x0, x1, y0, y1 = 150, SW - 70, 996, 1052
        label(d, (70, 1008), 'LINES', 12, FAINT)
        d.line([(x0, y1), (x1, y1)], fill=(34, 38, 47))
        top = max(4, max(r.moves[-1]['lines'] for r in self.rs))
        n = max(len(r.moves) for r in self.rs)
        for r, v in zip(self.rs, self.vs):
            k, ph, _ = r.at(t)
            upto = min(len(r.moves), k + (1 if ph in ('lock', 'flash', 'collapse', 'end') else 0))
            pts = [(x0, y1)] + [(x0 + (x1 - x0) * (j + 1) / n, y1 - (y1 - y0) * r.moves[j]['lines'] / top) for j in range(upto)]
            if len(pts) > 1:
                d.line(pts, fill=v.acc, width=3, joint='curve')
                ex, ey = pts[-1]
                dot = cap(10, v.acc)
                img.paste(dot, (int(ex) - 5, int(ey) - 5), dot)

    def game_frame(self, t):
        img = self.static.copy()
        d = ImageDraw.Draw(img)
        for i, (r, v) in enumerate(zip(self.rs, self.vs)):
            k, ph, p = v.board(img, t)
            if k >= len(r.moves) and r.top_out:
                ov = Image.new('RGBA', (W * CS, H * CS), (8, 9, 12, int(160 * p)))
                img.paste(ov, (v.bx, v.by), ov)
                d.text((v.bx + W * CS // 2, v.by + H * CS // 2), 'TOPPED OUT', font=font(42, 'Bold'), anchor='mm', fill=mix(BG, RED, p))
            self.panel(img, d, i, t)
        self.scoreboard(img, d, t)
        self.race(img, d, t)
        return img

    def intro(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        a = ease_out(min(1, t / 0.6))
        for i, r in enumerate(self.rs):
            sgn = -1 if i == 0 else 1
            cx = SW // 2 + sgn * int(380 + 140 * (1 - a))
            d.text((cx, SH // 2 + 10), r.name, font=font(150, 'Bold'), anchor='ms', fill=mix(BG, r.accent, a))
            d.text((cx, SH // 2 + 64), r.spec, font=font(24), anchor='ms', fill=mix(BG, DIM, a))
        d.text((SW // 2, SH // 2 - 20), 'vs', font=font(52, 'SemiLight'), anchor='ms', fill=mix(BG, DIM, a))
        d.text((SW // 2, SH // 2 - 250), f'{self.title}  ·  HEAD TO HEAD', font=font(26, 'SemiBold'), anchor='ms', fill=mix(BG, FAINT, a))
        d.text((SW // 2, SH // 2 + 220), f'Each move: pick one of {self.k} placements, same options for both', font=font(24), anchor='ms',
               fill=mix(BG, DIM, a))
        return img

    def outro(self, img, u):
        a = ease_out(min(1, u / 0.7))
        dim = Image.new('RGBA', (SW, SH), (6, 7, 10, int(170 * a)))
        img.paste(dim, (0, 0), dim)
        w, h = 900, 420
        x0, y0 = (SW - w) // 2, (SH - h) // 2 + int(30 * (1 - a))
        la = [r.moves[-1]['lines'] for r in self.rs]
        win = 0 if la[0] > la[1] else 1 if la[1] > la[0] else None
        acc = self.rs[win].accent if win is not None else TEXT
        sp = rrect_ss((w, h), 26, (13, 15, 20, int(245 * a)), acc + (int(255 * a),), 2)
        img.paste(sp, (x0, y0), sp)
        d = ImageDraw.Draw(img)
        d.text((SW // 2, y0 + 70), 'FINAL', font=font(20, 'Bold'), anchor='mm', fill=mix(BG, acc, a))
        d.text((SW // 2, y0 + 150), f'{self.rs[win].name} WINS' if win is not None else 'DRAW', font=font(84, 'Bold'), anchor='mm',
               fill=mix(BG, acc, a))
        for i, r in enumerate(self.rs):
            cx = SW // 2 + (-1 if i == 0 else 1) * 220
            d.text((cx, y0 + 250), r.name, font=font(28, 'Bold'), anchor='mm', fill=mix(BG, r.accent, a))
            d.text((cx, y0 + 312), f"{r.moves[-1]['lines']} lines", font=font(40, 'Bold'), anchor='mm', fill=mix(BG, TEXT, a))
            avg = sum(r.lat) / len(r.lat) if r.lat else 0
            tail = 'topped out' if r.top_out else 'still alive'
            d.text((cx, y0 + 362), f"{len(r.moves)} pieces · {tail} · avg {avg:.0f} ms", font=font(20), anchor='mm', fill=mix(BG, DIM, a))
        return img

    def frame(self, t):
        if t < INTRO:
            it = self.intro(t)
            if t > INTRO - 0.5:  # cross-fade into the game
                return Image.blend(it, self.game_frame(0.0), ease_io((t - (INTRO - 0.5)) / 0.5))
            return it
        g = t - INTRO
        img = self.game_frame(min(g, self.game_T + 0.8))
        if g > self.game_T + 0.8:
            img = self.outro(img, g - self.game_T - 0.8)
        return img


def encode(scene, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    n = int(scene.T * FPS)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SW}x{SH}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
           '-color_trc', 'bt709', '-movflags', '+faststart', str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    black = Image.new('RGB', (SW, SH), (0, 0, 0))
    for i in range(n):
        tt = i / FPS
        img = scene.frame(tt)
        if tt < 0.35:
            img = Image.blend(black, img, tt / 0.35)
        elif tt > scene.T - 0.6:
            img = Image.blend(black, img, max(0.0, (scene.T - tt) / 0.6))
        ff.stdin.write(img.tobytes())
        if i % 900 == 0:
            print(f'{i}/{n}', flush=True)
    ff.stdin.close()
    ff.wait()
    print('saved', out, f'{n / FPS:.1f}s')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs=2, help='two bench.py logs (same seed and scaffold)')
    ap.add_argument('--out', required=True, help='.mp4 (or .png with --still)')
    ap.add_argument('--pieces', type=int, help='cut both runs at this many pieces')
    ap.add_argument('--gpu', help='shown in the header, e.g. "RTX 3090"')
    ap.add_argument('--name', nargs=2, metavar=('LEFT', 'RIGHT'), help='display names (default: judge names from the logs)')
    ap.add_argument('--spec', nargs=2, metavar=('LEFT', 'RIGHT'), help='one-line model descriptions under the names')
    ap.add_argument('--still', type=float, help='write one PNG at this time (seconds) instead of a video')
    a = ap.parse_args()
    pace = Pace(0.16, 0.18, 0.12, 0.09, 0.07, 0.18, 0.12)
    runs = []
    for i, p in enumerate(a.runs):
        runs.append(Run(p, pace, a.pieces, name=a.name[i] if a.name else None, spec=a.spec[i] if a.spec else None))
    if runs[0].accent == runs[1].accent:
        runs[1].accent = next(c for c in PALETTE if c != runs[0].accent)
    scene = Broadcast(runs, gpu=a.gpu)
    out = Path(a.out)
    if a.still is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        scene.frame(a.still).save(out)
        print('saved', out, 'T', round(scene.T, 1))
        return
    encode(scene, out)


if __name__ == '__main__':
    main()
