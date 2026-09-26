"""Mica vs N on a Tetris-style game: every judge plays the same seeds, pieces and shuffled options.

Judges (repeat --judge, as many as you like):
  NAME=http://HOST:PORT/v1/systemone   any TypeSafe /v1/systemone server (Mica, Kev, ...); --model NAME=... sets "model"
  NAME=py:path/to/file.py[:ARG]        a Python file with make_judge(arg) -> fn(state, questions) -> /v1/systemone response
  random | greedy                      baselines: uniform random pick / always the heuristic's top placement

python bench.py --judge mica=http://127.0.0.1:8010/v1/systemone --judge kev=http://127.0.0.1:8009/v1/systemone \
                --judge random --judge greedy --seeds 7 11 23 --scaffold easy --out runs
python bench.py summarize runs          (rebuild runs/summary.md from the logs)
"""
import argparse, importlib.util, json, random, statistics, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Tetris, questions, questions_easy  # noqa: E402

DEFAULT_MODEL = {'mica': 'mica-v0.1-4b', 'kev': 'kev-latest'}
SCAFFOLDS = {'easy': (questions_easy, 4), 'base': (questions, 6)}


def parse_answer(res, wall_ms):
    a = res['answers']
    place = a['place']
    probs = place.get('probabilities') or {}
    choice = place.get('choice') or place.get('answer') or (max(probs, key=probs.get) if probs else None)
    if not probs and choice:
        probs = {choice: 1.0}

    def noul(q):
        v = (a.get(q) or {}).get('noul')
        return float(v) if isinstance(v, (int, float)) else None
    return {'probs': probs, 'choice': choice, 'risk': noul('risk'), 'clear': noul('clear'),
            'ms': float(res.get('latency_ms', wall_ms)), 'wall_ms': wall_ms, 'tokens': (res.get('usage') or {}).get('input_tokens')}


class HttpJudge:
    def __init__(self, url, model):
        self.url, self.model = url, model

    def __call__(self, state, qs, opts):
        req = urllib.request.Request(self.url, data=json.dumps({'model': self.model, 'state': state, 'questions': qs}).encode(),
                                     headers={'Content-Type': 'application/json'})
        t = time.perf_counter()
        res = json.load(urllib.request.urlopen(req, timeout=300))
        return parse_answer(res, (time.perf_counter() - t) * 1000)


class PyJudge:
    def __init__(self, path, arg=None):
        spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.fn = mod.make_judge(arg)

    def __call__(self, state, qs, opts):
        t = time.perf_counter()
        res = self.fn(state, qs)
        return parse_answer(res, (time.perf_counter() - t) * 1000)


class RandomJudge:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self, state, qs, opts):
        c = self.rng.choice(opts)['id']
        return {'probs': {o['id']: 1 / len(opts) for o in opts}, 'choice': c, 'risk': None, 'clear': None, 'ms': 0.0, 'wall_ms': 0.0}


class GreedyJudge:
    def __call__(self, state, qs, opts):
        c = min(opts, key=lambda o: o['rank'])['id']
        return {'probs': {o['id']: float(o['id'] == c) for o in opts}, 'choice': c, 'risk': None, 'clear': None, 'ms': 0.0, 'wall_ms': 0.0}


def make_judge(spec, models, seed):
    name, _, target = spec.partition('=')
    if not target:
        name, target = spec, spec
    if target == 'random':
        return name, RandomJudge(seed)
    if target == 'greedy':
        return name, GreedyJudge()
    if target.startswith('py:'):
        path, _, arg = target[3:].partition(':')
        return name, PyJudge(path, arg or None)
    if target.startswith('http'):
        return name, HttpJudge(target, models.get(name, DEFAULT_MODEL.get(name, name)))
    raise SystemExit(f'unknown judge spec: {spec}')


def play(name, judge, seed, pieces, scaffold, out, warmup=3):
    qfn, k = SCAFFOLDS[scaffold]
    g = Tetris(seed)
    warm = g.offered(k, salt=seed)
    for _ in range(warmup):  # warm kernels/caches so the first recorded move is not a cold start
        judge(g.state_text(), qfn(warm), warm)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'meta': {'judge': name, 'seed': seed, 'pieces': pieces, 'scaffold': scaffold}}) + '\n')
        while g.pieces < pieces:
            opts = g.offered(k, salt=seed)
            if not opts:
                f.write(json.dumps({'game_over': True, 'pieces': g.pieces, 'lines': g.lines}) + '\n')
                break
            before = [row[:] for row in g.grid]
            piece, nxt = g.piece, g.queue[0]
            if len(opts) == 1:
                res = {'probs': {opts[0]['id']: 1.0}, 'choice': opts[0]['id'], 'risk': None, 'clear': None, 'ms': 0, 'wall_ms': 0}
            else:
                res = judge(g.state_text(), qfn(opts), opts)
            move = next((m for m in opts if m['id'] == res['choice']), opts[0])
            g.play(move)
            f.write(json.dumps({'i': g.pieces, 'piece': piece, 'next': nxt, 'grid': before,
                                'opts': [{'id': m['id'], 'ri': m['ri'], 'c': m['c'], 'cells': m['cells'], 'full': m['o']['full'],
                                          'new_holes': m['o']['new_holes'], 'max': m['o']['max'], 'rank': m['rank']} for m in opts],
                                'probs': res['probs'], 'choice': move['id'], 'risk': res['risk'], 'clear': res['clear'],
                                'ms': res['ms'], 'wall_ms': res['wall_ms'], 'tokens': res.get('tokens'),
                                'cleared_rows': move['o']['cleared_rows'], 'lines': g.lines, 'height': move['o']['max']}) + '\n')
            f.flush()
    return g


# ------------------------------------------------------------------ summary
def load(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    meta = rows[0]['meta']
    mv = [r for r in rows if 'i' in r]
    multi = [r for r in mv if len(r['opts']) > 1]
    lat = [r['ms'] for r in mv if r['ms'] > 0]
    return {'judge': meta['judge'], 'seed': meta['seed'], 'scaffold': meta.get('scaffold', 'base'), 'cap': meta.get('pieces'),
            'lines': mv[-1]['lines'] if mv else 0, 'pieces': len(mv), 'topped_out': any('game_over' in r for r in rows),
            'best': sum(1 for r in multi if min(r['opts'], key=lambda o: o['rank'])['id'] == r['choice']) / max(1, len(multi)),
            'p50': statistics.median(lat) if lat else None}


def summarize(root):
    root = Path(root)
    runs = [load(p) for p in sorted(root.rglob('*.jsonl'))]
    out = ['# Results', '', 'Same seeds, same pieces, same shuffled options for every judge. "best" = picked the heuristic\'s top',
           'placement. Latency is per decision (3 questions share one request), as reported by each server.', '']
    for sc in sorted({r['scaffold'] for r in runs}):
        rs = [r for r in runs if r['scaffold'] == sc]
        seeds = sorted({r['seed'] for r in rs})
        out += [f'## {sc} ({SCAFFOLDS[sc][1]} options per move)', '',
                '| Judge | Lines per seed (' + ' / '.join(map(str, seeds)) + ') | Total lines | Pieces | Topped out | Best | p50 ms |',
                '|---|---|---:|---|---:|---:|---:|']
        by = {}
        for r in rs:
            by.setdefault(r['judge'], {})[r['seed']] = r
        for j, d in sorted(by.items(), key=lambda kv: -sum(x['lines'] for x in kv[1].values())):
            row = [d.get(s) for s in seeds]
            lines = ' / '.join(str(x['lines']) if x else '–' for x in row)
            pieces = ' / '.join(str(x['pieces']) if x else '–' for x in row)
            tops = sum(1 for x in row if x and x['topped_out'])
            best = statistics.mean(x['best'] for x in row if x)
            lat = [x['p50'] for x in row if x and x['p50']]
            lat_s = f'{statistics.median(lat):.0f}' if lat else '–'
            out.append(f"| {j} | {lines} | {sum(x['lines'] for x in row if x)} | {pieces} | {tops}/{sum(1 for x in row if x)} | "
                       f"{best * 100:.0f}% | {lat_s} |")
        out.append('')
    (root / 'summary.md').write_text('\n'.join(out), encoding='utf-8')
    json.dump(runs, open(root / 'summary.json', 'w', encoding='utf-8'), indent=1)
    print('\n'.join(out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'summarize':
        summarize(sys.argv[2] if len(sys.argv) > 2 else 'runs')
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--judge', action='append', required=True, help='NAME=URL | NAME=py:file.py[:arg] | random | greedy')
    ap.add_argument('--model', action='append', default=[], help='NAME=MODEL, the "model" field sent to an HTTP judge')
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 11, 23])
    ap.add_argument('--pieces', type=int, default=250)
    ap.add_argument('--scaffold', choices=['easy', 'base', 'both'], default='easy',
                    help='easy: 4 options with numeric outcomes and an explicit goal; base: 6 options, worded outcomes')
    ap.add_argument('--out', default='runs')
    a = ap.parse_args()
    models = dict(m.split('=', 1) for m in a.model)
    scaffolds = ['easy', 'base'] if a.scaffold == 'both' else [a.scaffold]
    out = Path(a.out)
    for spec in a.judge:
        for sc in scaffolds:
            for seed in a.seeds:
                name, judge = make_judge(spec, models, seed)
                g = play(name, judge, seed, a.pieces, sc, out / sc / f'{name}_s{seed}.jsonl')
                print(json.dumps({'judge': name, 'scaffold': sc, 'seed': seed, 'pieces': g.pieces, 'lines': g.lines}), flush=True)
    summarize(out)


if __name__ == '__main__':
    main()
