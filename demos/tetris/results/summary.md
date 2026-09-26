# Results

Same seeds, same pieces, same shuffled options for every judge. "best" = picked the heuristic's top
placement. Latency is per decision (3 questions share one request), as reported by each server.

## base (6 options per move)

| Judge | Lines per seed (7 / 11 / 23) | Total lines | Pieces | Topped out | Best | p50 ms |
|---|---|---:|---|---:|---:|---:|
| mica | 5 / 11 / 9 | 25 | 51 / 65 / 60 | 3/3 | 47% | 140 |
| kev | 6 / 6 / 6 | 18 | 54 / 53 / 52 | 3/3 | 30% | 138 |
| laya | 0 / 1 / 3 | 4 | 37 / 40 / 44 | 3/3 | 13% | 37 |

## easy (4 options per move)

| Judge | Lines per seed (7 / 11 / 23) | Total lines | Pieces | Topped out | Best | p50 ms |
|---|---|---:|---|---:|---:|---:|
| mica | 33 / 97 / 93 | 223 | 121 / 250 / 250 | 1/3 | 75% | 136 |
| kev | 27 / 17 / 11 | 55 | 107 / 86 / 69 | 3/3 | 49% | 124 |
| laya | 4 / 8 / 5 | 17 | 50 / 59 / 52 | 3/3 | 27% | 37 |
