# search_stem_exact.py (standard library only)
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Callable, List, Sequence, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())

def stem_word(w: str) -> str:
    """Tiny heuristic stemmer (English-ish). Good enough for coursework demos."""
    w = w.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]  # running -> run
        return base
    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w

def stem_tokens(tokens: Sequence[str]) -> List[str]:
    return [stem_word(t) for t in tokens]

_QUOTE_PATTERNS = [r'"([^"]+)"', r'“([^”]+)”']

def _normalize_phrase(p: str) -> str:
    p = p.strip().lower()
    p = re.sub(r"\s+", " ", p)
    return p

def extract_quoted_phrases(query: str) -> List[str]:
    phrases: List[str] = []
    for pat in _QUOTE_PATTERNS:
        phrases += re.findall(pat, query)
    return [_normalize_phrase(p) for p in phrases if p.strip()]

def remove_quoted_spans(query: str) -> str:
    q = re.sub(r'"[^"]+"', " ", query)
    q = re.sub(r'“[^”]+”', " ", q)
    return q

@dataclass(frozen=True)
class QueryParts:
    raw_query: str
    phrases: Tuple[str, ...]
    terms: Tuple[str, ...]
    stemmed_terms: Tuple[str, ...]

    def stemmed_query_string(self) -> str:
        return " ".join(self.stemmed_terms)

def parse_query_stem_exact(query: str) -> QueryParts:
    phrases = extract_quoted_phrases(query)
    unquoted = remove_quoted_spans(query)
    terms = tokenize(unquoted)
    stemmed = stem_tokens(terms)
    return QueryParts(
        raw_query=query,
        phrases=tuple(phrases),
        terms=tuple(terms),
        stemmed_terms=tuple(stemmed),
    )

def contains_exact_phrase(text: str, phrase: str) -> bool:
    """Multi-word: substring. Single word: word boundary."""
    if not phrase:
        return True
    t = re.sub(r"\s+", " ", text.lower()).strip()
    p = _normalize_phrase(phrase)
    if " " in p:
        return p in t
    return re.search(rf"\b{re.escape(p)}\b", t) is not None

def filter_doc_ids_by_phrases(
    doc_ids: Sequence[int],
    get_text: Callable[[int], str],
    phrases: Sequence[str],
) -> List[int]:
    if not phrases:
        return list(doc_ids)
    kept = []
    for i in doc_ids:
        txt = get_text(i)
        if all(contains_exact_phrase(txt, ph) for ph in phrases):
            kept.append(i)
    return kept

if __name__ == "__main__":
    q = 'clean "new york" houses'
    parts = parse_query_stem_exact(q)
    print(parts.phrases)        # ('new york',)
    print(parts.stemmed_terms)  # ('clean', 'house' or 'hous'..., depends)
