"""Tetris-style 10x20 game for decision-model benchmarks.

Each turn: every reachable placement (rotation x column, hard drop) is simulated, ranked by a classic heuristic
(0.76*lines - 0.51*aggregate height - 0.36*holes - 0.18*bumpiness), and the top K are offered to the judge in shuffled
order. The piece sequence and the shuffle depend only on the seed, so every judge sees exactly the same game."""
import random

W, H = 10, 20
SHAPES = {'I': [(0, 0), (1, 0), (2, 0), (3, 0)], 'O': [(0, 0), (1, 0), (0, 1), (1, 1)], 'T': [(0, 0), (1, 0), (2, 0), (1, 1)],
          'S': [(1, 0), (2, 0), (0, 1), (1, 1)], 'Z': [(0, 0), (1, 0), (1, 1), (2, 1)], 'J': [(0, 0), (0, 1), (1, 1), (2, 1)],
          'L': [(2, 0), (0, 1), (1, 1), (2, 1)]}


def rotations(cells):
    out, cur = [], list(cells)
    for _ in range(4):
        mx, my = min(x for x, _ in cur), min(y for _, y in cur)
        norm = tuple(sorted((x - mx, y - my) for x, y in cur))
        if norm not in out:
            out.append(norm)
        cur = [(-y, x) for x, y in cur]
    return out


def heights(g):
    return [next((H - y for y in range(H) if g[y][x]), 0) for x in range(W)]


def holes(g):
    n = 0
    for x in range(W):
        seen = False
        for y in range(H):
            if g[y][x]:
                seen = True
            elif seen:
                n += 1
    return n


class Tetris:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.grid = [[0] * W for _ in range(H)]
        self.lines = self.pieces = 0
        self.queue = [self._rand(), self._rand()]
        self.piece = self.queue.pop(0)

    def _rand(self):
        return self.rng.choice(list(SHAPES))

    def _drop(self, rot, col):
        width = max(x for x, _ in rot) + 1
        if col + width > W:
            return None
        y = -1
        while all(y + 1 + dy < H and not self.grid[y + 1 + dy][col + dx] for dx, dy in rot):
            y += 1
        return None if y < 0 else [(col + dx, y + dy) for dx, dy in rot]

    def _outcome(self, cells):
        g = [row[:] for row in self.grid]
        for x, y in cells:
            g[y][x] = self.piece
        cleared = [y for y in range(H) if all(g[y])]
        g = [row for row in g if not all(row)]
        g = [[0] * W for _ in range(H - len(g))] + g
        h = heights(g)
        agg, bump, hl = sum(h), sum(abs(h[i + 1] - h[i]) for i in range(W - 1)), holes(g)
        return {'grid': g, 'full': len(cleared), 'cleared_rows': cleared, 'max': max(h), 'bump': bump, 'agg': agg,
                'new_holes': hl - holes(self.grid), 'score': 0.76 * len(cleared) - 0.51 * agg - 0.36 * hl - 0.18 * bump}

    def moves(self):
        out = []
        for ri, rot in enumerate(rotations(SHAPES[self.piece])):
            for c in range(W):
                cells = self._drop(rot, c)
                if cells:
                    out.append({'id': f'r{ri}c{c}', 'ri': ri, 'c': c, 'cells': cells, 'o': self._outcome(cells)})
        return sorted(out, key=lambda m: -m['o']['score'])

    def offered(self, k=6, salt=0):
        opts = self.moves()[:k]
        for r, m in enumerate(opts):
            m['rank'] = r
        random.Random(salt * 7919 + self.pieces).shuffle(opts)
        return opts

    def play(self, move):
        self.grid = move['o']['grid']
        self.lines += move['o']['full']
        self.pieces += 1
        self.piece = self.queue.pop(0)
        self.queue.append(self._rand())

    def state_text(self):
        rows = '\n'.join(''.join('#' if v else '.' for v in row) for row in self.grid)
        return (f'Tetris board, {W} columns x {H} rows (row 0 is the top). # = filled, . = empty. A full row is cleared.\n'
                f'Current piece: {self.piece}. Next piece: {self.queue[0]}. Lines cleared: {self.lines}.\n{rows}')


def describe(o):
    surf = 'flat' if o['bump'] <= 4 else 'slightly uneven' if o['bump'] <= 9 else 'very uneven'
    lines = f"clears {o['full']} line{'' if o['full'] == 1 else 's'}"
    hl = 'creates no new holes' if o['new_holes'] <= 0 else f"creates {o['new_holes']} new hole{'s' if o['new_holes'] > 1 else ''}"
    return f"{lines}; {hl}; stack height after: {o['max']}; surface after: {surf}"


def describe_easy(o):
    hl = max(0, o['new_holes'])
    return (f"clears {o['full']} line{'' if o['full'] == 1 else 's'}; new holes: {hl}; tallest column after: {o['max']}; "
            f"total stack height after: {o['agg']}; bumpiness after: {o['bump']}")


def questions_easy(opts):
    return {'place': {'type': 'choice',
                      'instructions': 'Which placement is best? Clear lines when possible, avoid creating holes, and keep the stack low and flat.',
                      'criteria': {m['id']: describe_easy(m['o']) for m in opts}},
            'risk': {'type': 'noul', 'instructions': 'Is the stack close to reaching the top of the board?'},
            'clear': {'type': 'noul', 'instructions': 'Can the current piece complete at least one full row somewhere on this board?'}}


def questions(opts):
    return {'place': {'type': 'choice', 'instructions': 'Where should the current piece go to survive longest and clear lines?',
                      'criteria': {m['id']: describe(m['o']) for m in opts}},
            'risk': {'type': 'noul', 'instructions': 'Is the stack close to reaching the top of the board?'},
            'clear': {'type': 'noul', 'instructions': 'Can the current piece complete at least one full row somewhere on this board?'}}
