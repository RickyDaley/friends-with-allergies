import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import csv
from collections import defaultdict

# from your reusable helper module
from search_stem_exact import tokenize, stem_tokens, parse_query_stem_exact, filter_doc_ids_by_phrases

# parser for operators
OP_MAP = {
    "and": "&",
    "or": "|",
    "not": "1 -",
}

def load_restaurant_docs(menu_file: str, review_file: str):
    """
    Reads menu_highlights.csv and review_data.csv from the SAME DIRECTORY as this script,
    and builds one document per restaurant by concatenating menu + reviews.
    Both files are tab-separated.
    """
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    menu_path = os.path.join(__location__, menu_file)
    review_path = os.path.join(__location__, review_file)

    menus = defaultdict(list)
    with open(menu_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            r = (row.get("Restaurant") or "").strip()
            if not r:
                continue
            cat = row.get("Category") or ""
            dish = row.get("Dish") or ""
            desc = row.get("Description") or ""
            fr = row.get("Food restrictions") or ""
            # keep it simple: one menu line per dish
            menus[r].append(f"{cat} | {dish} | {desc} | {fr}")

    reviews = defaultdict(list)
    with open(review_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            r = (row.get("Restaurant") or "").strip()
            if not r:
                continue
            score = (row.get("Review Score") or "").strip()
            txt = (row.get("Review Text") or "").strip()
            if txt:
                reviews[r].append(f"[score={score}] {txt}" if score else txt)

    restaurants = sorted(set(menus.keys()) | set(reviews.keys()))

    documents = []
    for r in restaurants:
        # put restaurant name up front so snippets are useful
        doc = [f"Restaurant: {r}"]
        if menus.get(r):
            doc.append("MENU:")
            doc.extend(menus[r])
        if reviews.get(r):
            doc.append("REVIEWS:")
            doc.extend(reviews[r])
        documents.append("\n".join(doc))

    return documents, restaurants

def stem_analyzer(text: str):
    # standard-library stemming via your helper module
    return stem_tokens(tokenize(text))


def get_term_vector(term, t2i, td_matrix):
    term = term.lower()
    if term in t2i:
        return td_matrix[t2i[term]]
    return np.zeros(td_matrix.shape[1], dtype=bool)


def _normalize_query_for_bool(query: str) -> str:
    # remove quoted spans: "..." or “...”
    q = re.sub(r'"[^"]+"|“[^”]+”', " ", query)
    # ensure parentheses are tokenized separately
    q = re.sub(r"([()])", r" \1 ", q)
    # collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    return q


def rewrite_query(query_for_bool, t2i, td_matrix):
    """
    Turn a boolean query string into an eval-able expression:
      clean AND (house OR home)
    -> get_term_vector("clean",...) & (get_term_vector("hous",...) | get_term_vector("home",...))
    IMPORTANT: we stem query terms here to match stemmed vocabulary in CountVectorizer.
    """
    tokens = query_for_bool.split()
    parts = []

    for t in tokens:
        tl = t.lower()
        if tl in OP_MAP:
            parts.append(OP_MAP[tl])
        elif t in ("(", ")"):
            parts.append(t)
        else:
            # stem the query term to match stemmed index
            stemmed = stem_tokens([tl])[0]
            parts.append(f'get_term_vector("{stemmed}", t2i, td_matrix)')

    return "(" + " ".join(parts) + ")"

def split_fragments(text: str):
    """
    Split a long restaurant doc into smaller fragments.
    We treat newline and ' | ' as boundaries too, because your docs contain lots of them.
    """
    t = text.replace(" | ", ". ").replace("\n", ". ")
    # split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', t)
    return [p.strip() for p in parts if p.strip()]

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def fragment_matches(fragment: str, qparts):
    # exact phrase match (quoted)
    frag_norm = normalize_ws(fragment)
    for ph in qparts.phrases:
        if ph and ph in frag_norm:
            return True

    # stemming match (unquoted)
    frag_stems = set(stem_tokens(tokenize(fragment)))
    return bool(frag_stems.intersection(set(qparts.stemmed_terms)))

def highlight(fragment: str, qparts):
    """
    Very light highlighting:
    - highlight quoted phrases (exact)
    - highlight raw unquoted terms (not stems) by word boundary, case-insensitive
    """
    out = fragment

    # highlight phrases first
    for ph in qparts.phrases:
        if not ph:
            continue
        out = re.sub(re.escape(ph), lambda m: f"[{m.group(0)}]", out, flags=re.IGNORECASE)

    # highlight raw terms (optional, helps readability)
    for term in qparts.terms:
        if not term:
            continue
        out = re.sub(rf"\b{re.escape(term)}\b", lambda m: f"[{m.group(0)}]", out, flags=re.IGNORECASE)

    return out

def make_context_snippet(doc: str, qparts, max_hits: int = 3):
    header = doc.splitlines()[0] if doc else "Restaurant: (unknown)"
    hits = []

    for frag in split_fragments(doc):
        if fragment_matches(frag, qparts):
            hits.append(highlight(frag, qparts))
            if len(hits) >= max_hits:
                break

    if not hits:
        return header  # fallback: at least show restaurant name

    return header + "\n  - " + "\n  - ".join(hits)

def split_fragments(text: str):
    """
    Split a long restaurant doc into smaller fragments.
    We treat newline and ' | ' as boundaries too, because your docs contain lots of them.
    """
    t = text.replace(" | ", ". ").replace("\n", ". ")
    # split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', t)
    return [p.strip() for p in parts if p.strip()]

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def fragment_matches(fragment: str, qparts):
    # exact phrase match (quoted)
    frag_norm = normalize_ws(fragment)
    for ph in qparts.phrases:
        if ph and ph in frag_norm:
            return True

    # stemming match (unquoted)
    frag_stems = set(stem_tokens(tokenize(fragment)))
    return bool(frag_stems.intersection(set(qparts.stemmed_terms)))

def highlight(fragment: str, qparts):
    """
    Very light highlighting:
    - highlight quoted phrases (exact)
    - highlight raw unquoted terms (not stems) by word boundary, case-insensitive
    """
    out = fragment

    # highlight phrases first
    for ph in qparts.phrases:
        if not ph:
            continue
        out = re.sub(re.escape(ph), lambda m: f"[{m.group(0)}]", out, flags=re.IGNORECASE)

    # highlight raw terms (optional, helps readability)
    for term in qparts.terms:
        if not term:
            continue
        out = re.sub(rf"\b{re.escape(term)}\b", lambda m: f"[{m.group(0)}]", out, flags=re.IGNORECASE)

    return out

def make_context_snippet(doc: str, qparts, max_hits: int = 3):
    header = doc.splitlines()[0] if doc else "Restaurant: (unknown)"
    hits = []

    for frag in split_fragments(doc):
        if fragment_matches(frag, qparts):
            hits.append(highlight(frag, qparts))
            if len(hits) >= max_hits:
                break

    if not hits:
        return header  # fallback: at least show restaurant name

    return header + "\n  - " + "\n  - ".join(hits)


def print_doc_snippet(doc):
    words = doc.split()
    return " ".join(words[:50]) + " ..."


def search_tool(t2i, td_matrix, documents):
    print()
    print("*Boolean search tool (stemming + quoted exact match)*")
    print('(Enter empty line to stop.)')
    print()

    while True:
        query = input("Please enter the query: ").strip()
        if query == "":
            print("Good bye!!!")
            break

        # 1) parse quoted phrases vs unquoted terms
        parts = parse_query_stem_exact(query)

        # 2) boolean part ignores quoted phrases (they are enforced later)
        query_for_bool = _normalize_query_for_bool(query)

        rewritten = rewrite_query(query_for_bool, t2i, td_matrix)

        hits_vector = eval(rewritten, {
            "td_matrix": td_matrix,
            "t2i": t2i,
            "get_term_vector": get_term_vector,
            "np": np,
            "__builtins__": {}
        })
        hits_vector = np.asarray(hits_vector).ravel().astype(bool)
        hits_list = np.where(hits_vector)[0]

        # 3) enforce quoted phrases as exact matches
        hits_list = filter_doc_ids_by_phrases(
            hits_list,
            get_text=lambda i: documents[i],
            phrases=parts.phrases
        )

        count = len(hits_list)
        print(f"Found {count} matching document(s)")
        if count == 0:
            print()
            continue

        n = 5
        shown = min(n, count)
        print(f"Showing first {shown} of {count}:\n")
        for rank, doc_idx in enumerate(hits_list[:n], 1):
            print(f"Matching doc #{rank}:")
            print(make_context_snippet(documents[doc_idx], parts, max_hits=3))
            


def main():
    print('Loading restaurant documents from menu_highlights.csv + review_data.csv ...')

    menu_file = "menu_highlights.csv"
    review_file = "review_data.csv"
    documents, restaurants = load_restaurant_docs(menu_file, review_file)

    cv = CountVectorizer(analyzer=stem_analyzer, binary=True)
    sparse_matrix = cv.fit_transform(documents)

    td_matrix = np.asarray(sparse_matrix.T.todense(), dtype=bool)
    t2i = cv.vocabulary_

    print(f"Done! Restaurants: {len(restaurants)} | Vocabulary size: {len(t2i)}")
    search_tool(t2i, td_matrix, documents)


if __name__ == "__main__":
    main()
