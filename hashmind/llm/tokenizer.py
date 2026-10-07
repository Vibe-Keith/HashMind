"""SentencePiece-style (llama "SPM") tokenizer driven by GGUF vocab + scores.

Mirrors llama.cpp's ``llm_tokenizer_spm``: start from UTF-8 characters,
repeatedly merge the adjacent pair whose concatenation is a vocab token with
the highest score, then fall back to ``<0xXX>`` byte tokens for leftovers.
Words are processed independently (llama vocab pieces only carry ``▁`` at the
start), which keeps it fast enough for experiment-sized corpora.
"""

from __future__ import annotations

import heapq
from functools import lru_cache

SPACE = "▁"


class SPMTokenizer:
    def __init__(self, tokens: list[str], scores: list[float], bos_id: int = 1, eos_id: int = 2,
                 add_space_prefix: bool = True) -> None:
        self.tokens = tokens
        self.scores = scores
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.bos_id, self.eos_id = bos_id, eos_id
        self.add_space_prefix = add_space_prefix
        self._byte = {b: self.vocab.get(f"<0x{b:02X}>") for b in range(256)}
        self._piece = lru_cache(maxsize=200_000)(self._encode_piece)

    @classmethod
    def from_gguf_metadata(cls, md: dict) -> "SPMTokenizer":
        if md.get("tokenizer.ggml.model") != "llama":
            raise ValueError(f"SPM tokenizer needs tokenizer.ggml.model == 'llama', got "
                             f"{md.get('tokenizer.ggml.model')!r}")
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.scores"]),
                   int(md.get("tokenizer.ggml.bos_token_id", 1)), int(md.get("tokenizer.ggml.eos_token_id", 2)),
                   bool(md.get("tokenizer.ggml.add_space_prefix", True)))

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        sym = list(piece)
        # doubly linked list over symbols; heap of (-score, left_index, left_text, right_text)
        prev = list(range(-1, len(sym) - 1))
        nxt = list(range(1, len(sym) + 1))
        nxt[-1] = -1
        alive = [True] * len(sym)
        heap: list[tuple[float, int, str, str]] = []

        def push(i: int) -> None:
            j = nxt[i]
            if j == -1:
                return
            tid = self.vocab.get(sym[i] + sym[j])
            if tid is not None:
                heapq.heappush(heap, (-self.scores[tid], i, sym[i], sym[j]))

        for i in range(len(sym) - 1):
            push(i)
        while heap:
            _, i, a, b = heapq.heappop(heap)
            j = nxt[i]
            if not alive[i] or j == -1 or sym[i] != a or sym[j] != b:
                continue  # stale entry
            sym[i] = a + b
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] != -1:
                prev[nxt[j]] = i
            if prev[i] != -1:
                push(prev[i])
            push(i)
        out: list[int] = []
        i = 0
        while i != -1:
            s = sym[i]
            tid = self.vocab.get(s)
            if tid is not None:
                out.append(tid)
            else:
                out.extend(self._byte[b] or 0 for b in s.encode("utf-8"))
            i = nxt[i]
        return tuple(out)

    def encode(self, text: str, bos: bool = True) -> list[int]:
        t = text.replace(" ", SPACE)
        if self.add_space_prefix and not t.startswith(SPACE):
            t = SPACE + t
        ids = [self.bos_id] if bos else []
        start = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or (t[k] == SPACE and t[k - 1] != SPACE):
                ids.extend(self._piece(t[start:k]))
                start = k
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            if i in (self.bos_id, self.eos_id):
                continue
            t = self.tokens[i]
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                out.append(int(t[3:5], 16))
            else:
                out.extend(t.replace(SPACE, " ").encode("utf-8"))
        s = out.decode("utf-8", errors="replace")
        return s[1:] if s.startswith(" ") else s
