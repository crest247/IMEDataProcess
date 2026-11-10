import csv
import os
from dataclasses import dataclass

from training.units import SOURCE_DATA_DIR, UnitTable


TONE_MARK = {1: "ˉ", 2: "ˊ", 3: "ˇ", 4: "ˋ", 5: "˙"}
MARK_TONE = {m: t for t, m in TONE_MARK.items() if m}
STANDALONE = set("ㄓㄔㄕㄖㄗㄘㄙ" "ㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ")


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    initial: str
    remain: str | None
    tone: int | None

    @property
    def text(self):
        return self.initial + (self.remain or "") + (TONE_MARK[self.tone] if self.tone else "")

    def matches(self, syl):
        i, r, t = syl
        if i != self.initial:
            return False
        if self.remain is not None and r != self.remain:
            return False
        return self.tone is None or t == self.tone


class Bopomofo:
    def __init__(self, data_dir=SOURCE_DATA_DIR):
        self.syl = {}
        for r in csv.DictReader(open(os.path.join(data_dir, "chars.csv"), encoding="utf-8")):
            self.syl[int(r["char_id"])] = (r["initial"], r["remain"], int(r["tone"]))
        self.char_list = {}
        for r in csv.DictReader(open(os.path.join(data_dir, "all_chars.csv"), encoding="utf-8")):
            key = (r["initial"], r["remain"], int(r["tone"]))
            lst = self.char_list.setdefault(key, [])
            if r["char"] not in lst:
                lst.append(r["char"])
        self.table_syl = set(self.char_list)
        self._table_keys = {}

        self.syl_set = set(self.syl.values()) | self.table_syl
        self.initials = {s[0] for s in self.syl_set}
        self.tones_of_initial = {}
        self.tones_of_full = {}
        for i, r, t in self.syl_set:
            self.tones_of_initial.setdefault(i, set()).add(t)
            self.tones_of_full.setdefault((i, r), set()).add(t)
        self.remains = {}
        for i, r in self.tones_of_full:
            if r:
                self.remains.setdefault(i, []).append(r)
        self.complete = {i + r + TONE_MARK[t]: (i, r, t) for i, r, t in self.table_syl}

    def tokens_at(self, s, p):
        out = []
        i = s[p]
        if i not in self.initials:
            return out
        forms = [(None, p + 1)]
        for r in self.remains.get(i, []):
            if s.startswith(r, p + 1):
                forms.append((r, p + 1 + len(r)))
        for r, q in forms:
            tones = self.tones_of_initial[i] if r is None else self.tones_of_full[(i, r)]
            if q < len(s) and s[q] in MARK_TONE:
                t = MARK_TONE[s[q]]
                if r is None and i in STANDALONE and (i, "", t) in self.table_syl:
                    out.append(Token(p, q + 1, i, "", t))
                elif t in tones:
                    out.append(Token(p, q + 1, i, r, t))
            else:
                out.append(Token(p, q, i, r, None))
        return out

    def all_tokens(self, s):
        n = len(s)
        toks = [self.tokens_at(s, p) for p in range(n)]
        eaten = [False] * n
        for p in range(n):
            if eaten[p]:
                toks[p] = []
                continue
            for t in toks[p]:
                for q in range(p + 1, min(t.end, n)):
                    eaten[q] = True
        return toks

    def lattice(self, s):
        n = len(s)
        raw = self.all_tokens(s)
        ok = [False] * (n + 1)
        ok[n] = True
        tok = [[] for _ in range(n)]
        for p in range(n - 1, -1, -1):
            tok[p] = [t for t in raw[p] if ok[t.end]]
            ok[p] = bool(tok[p])
        return tok, ok

    def segmentations(self, s, limit=50):
        tok, ok = self.lattice(s)
        res = []

        def dfs(p, path):
            if len(res) >= limit:
                return
            if p == len(s):
                res.append(list(path))
                return
            for t in tok[p]:
                path.append(t)
                dfs(t.end, path)
                path.pop()

        if s and ok[0]:
            dfs(0, [])
        return res

    def count_segmentations(self, s):
        tok, ok = self.lattice(s)
        n = len(s)
        cnt = [0] * (n + 1)
        cnt[n] = 1
        for p in range(n - 1, -1, -1):
            cnt[p] = sum(cnt[t.end] for t in tok[p])
        return cnt[0] if n else 0

    def classify(self, s, continuation=False):
        if not s:
            return "空"
        if not continuation and s in self.complete:
            return "單字完整注音"
        return "混打" if self.tokens_at(s, 0) else "不合法"

    def table_keys(self, t):
        k = (t.initial, t.remain, t.tone)
        if k in self._table_keys:
            return self._table_keys[k]
        match = [key for key in self.char_list
                 if key[0] == t.initial and (t.remain is None or key[1] == t.remain)]
        keys = [key for key in match if key[2] == t.tone] if t.tone is not None else match
        self._table_keys[k] = keys
        return keys

    def reading_keys(self, t):
        r = t.remain or ""
        return [key for key in self.char_list
                if key[0] == t.initial and key[1] == r and (t.tone is None or key[2] == t.tone)]


class Lexicon(UnitTable):
    def __init__(self, bpmf: Bopomofo, data_dir=SOURCE_DATA_DIR):
        super().__init__(data_dir)
        self.bpmf = bpmf
        self.syl_id = {v: k for k, v in bpmf.syl.items()}
        self.single_char_unit = {}
        for uid, (unit, cids, _) in enumerate(self.units):
            if len(cids) == 1:
                self.single_char_unit.setdefault((unit, cids[0]), uid)

    def table_chars(self, t):
        return self.chars_of(self.bpmf.table_keys(t))

    def chars_of(self, keys):
        out, seen = [], set()
        for key in keys:
            cid = self.syl_id.get(key)
            for ch in self.bpmf.char_list[key]:
                if ch not in seen:
                    seen.add(ch)
                    out.append((ch, self.single_char_unit.get((ch, cid), -1) if cid is not None else -1))
        return out

    def homophones(self, s):
        best = None
        for t in self.bpmf.tokens_at(s, 0):
            keys = self.bpmf.reading_keys(t)
            if keys and (best is None or t.end > best[0]):
                best = (t.end, keys)
        return [(best[0], self.chars_of(best[1]))] if best else []

    def edges_ext(self, s):
        n = len(s)
        toks = self.bpmf.all_tokens(s)
        edges = [[] for _ in range(n)]
        groups = []
        reach = [False] * (n + 1)
        if n:
            reach[0] = True
        for p in range(n):
            if not reach[p]:
                continue
            seen = set()

            def dfs(q, node, depth):
                if depth and None in node:
                    for _, _, uid in node[None]:
                        if (uid, q) not in seen:
                            seen.add((uid, q))
                            edges[p].append((uid, q))
                if q == n:
                    return
                for t in toks[q]:
                    for cid, child in node.items():
                        if cid is not None and t.matches(self.bpmf.syl[cid]):
                            dfs(t.end, child, depth + 1)

            dfs(p, self.root, 0)
            for t in toks[p]:
                if self.bpmf.table_keys(t):
                    groups.append(t)
                    edges[p].append((-len(groups), t.end))
            for _, e in edges[p]:
                reach[e] = True
        target = n if reach[n] else max([q for q in range(1, n + 1) if reach[q]], default=0)
        INF = float("inf")
        cost = [INF] * (n + 1)
        cost[target] = 0
        for p in range(target - 1, -1, -1):
            for u, e in edges[p]:
                if e <= target:
                    cost[p] = min(cost[p], (u < 0) + cost[e])
        for p in range(n):
            edges[p] = [(u, e) for u, e in edges[p]
                        if p < target and e <= target and (u < 0) + cost[e] == cost[p] < INF]
        return edges, [self.table_chars(t) for t in groups], target

    def edges(self, s):
        tok, ok = self.bpmf.lattice(s)
        n = len(s)
        edges = [[] for _ in range(n)]
        reach = [False] * (n + 1)
        if n and ok[0]:
            reach[0] = True
        for p in range(n):
            if not reach[p]:
                continue
            seen = set()

            def dfs(q, node, depth):
                if depth and None in node:
                    for _, _, uid in node[None]:
                        if (uid, q) not in seen:
                            seen.add((uid, q))
                            edges[p].append((uid, q))
                if q == n:
                    return
                for t in tok[q]:
                    for cid, child in node.items():
                        if cid is not None and t.matches(self.bpmf.syl[cid]):
                            dfs(t.end, child, depth + 1)

            dfs(p, self.root, 0)
            for _, e in edges[p]:
                reach[e] = True
        done = [False] * (n + 1)
        done[n] = True
        for p in range(n - 1, -1, -1):
            edges[p] = [(u, e) for u, e in edges[p] if done[e]]
            done[p] = bool(edges[p])
        return edges

    def prefix_candidates(self, s):
        tok, ok = self.bpmf.lattice(s)
        best = {}

        def dfs(p, node, path):
            if path and None in node:
                for unit, cnt, uid in node[None]:
                    k = (unit, p)
                    if k not in best or best[k][1] < cnt:
                        best[k] = (unit, cnt, p, tuple(path))
            if p == len(s):
                return
            for t in tok[p]:
                for cid, child in node.items():
                    if cid is not None and t.matches(self.bpmf.syl[cid]):
                        path.append(t)
                        dfs(t.end, child, path)
                        path.pop()

        if s and ok[0]:
            dfs(0, self.root, [])
        return list(best.values())
