import os
import nltk
import numpy as np
import pandas as pd
import translators as ts
from math import inf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
lemmatizer = nltk.wordnet.WordNetLemmatizer()
N = 10  # How many top matches to show


def translate_chunk(chunk):
    try:
        transl_chunk = ts.translate_text(chunk, from_language='fi', to_language='en', translator='yandex')
        return transl_chunk
    except:
        return chunk
    

def translate_batch(text_entries):
    transl_entries = []
    limit = 10000
    chunk = ''
    for entry in text_entries:
        test_chunk = chunk + '\n' + entry if chunk else entry
        if len(test_chunk) > limit:
            chunk = test_chunk[:limit] if not chunk else chunk
            transl_chunk = translate_chunk(chunk)
            transl_entries.append(transl_chunk)
            chunk = ''
        elif len(test_chunk) == limit:
            transl_chunk = translate_chunk(test_chunk)
            transl_entries.append(transl_chunk)
            chunk = ''
        else:
            chunk = test_chunk
    if chunk:
        transl_chunk = translate_chunk(chunk)
        transl_entries.append(transl_chunk)
    return transl_entries
    

def initialise_index():
    documents = []
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = lambda x: os.path.join(__location__, x)
    data = pd.read_csv(absolute_path("restaurant_data.csv"), sep="\t", index_col=0)
    reviews = pd.read_csv(absolute_path("translated_review_data.csv"), sep="\t")
    highlights = pd.read_csv(absolute_path("menu_highlights.csv"), sep="\t", index_col=0)
    for _, row in data.iterrows():
        rest_reviews = reviews[reviews.Restaurant == row.Name]["Reviews"].values[0]
        rest_reviews = rest_reviews if isinstance(rest_reviews, str) else ''
        full_rest_text = '\n'.join([row.Name, row.Location, row.Cuisine, str(row.Meals)]) + '\n' + rest_reviews
        documents.append(full_rest_text)
    doc_embeddings = model.encode(documents)
    return data, documents, doc_embeddings


def extract_lemmas(docs):
    lem_sentence = lambda doc: ' '.join([lemmatizer.lemmatize(word)for word in doc.lower().split()])
    if isinstance(docs, str):
        return lem_sentence(docs)
    return [lem_sentence(doc) for doc in docs]


def rewrite_query(query):
    tokens = query.split()
    operators = ("and", "or", "not", "&", "|", "(", ")")
    parts = []
    min_ngram_size, max_ngram_size = inf, 1
    current_ngram = []
    for i, t in enumerate(tokens):
        if t in operators or i == len(tokens) - 1:
            if t not in operators:
                current_ngram.append(t.lower())
            if len(current_ngram) > 0:
                min_ngram_size = len(current_ngram) if len(current_ngram) < min_ngram_size else min_ngram_size
                max_ngram_size = len(current_ngram) if len(current_ngram) > max_ngram_size else max_ngram_size
                ngram = ' '.join(current_ngram)
                parts.append(f'get_term_vector("{ngram}", t2i, td_matrix)')
                current_ngram = []
            if t in operators:
                parts.append({"and":"&", "or":"|", "not":"1 -"}.get(t.lower(), t))
        else:
            current_ngram.append(t.lower())
    return "(" + " ".join(parts) + ")", min_ngram_size, max_ngram_size


def get_term_vector(term, t2i, td_matrix):
    #Returns zero vector (no matches) for unknown terms
    term = term.lower()
    if term in t2i:
        return td_matrix[t2i[term]]
    else:
        return np.zeros(td_matrix.shape[1], dtype=bool)


def boolean_search(query_yes, query_no, documents):
    if not (query_yes and query_no):
        return []
    if query_yes:
        transl_query_yes = translate_chunk(query_yes)
        lemmatized_query_yes = extract_lemmas(transl_query_yes)
        rewritten_yes, min_ngram_size_yes, max_ngram_size_yes = rewrite_query(lemmatized_query_yes)
        rewritten, min_ngram_size, max_ngram_size = rewritten_yes, min_ngram_size_yes, max_ngram_size_yes
    if query_no:
        transl_query_no = translate_chunk(query_no)
        lemmatized_query_no = extract_lemmas(transl_query_no)
        rewritten_no, min_ngram_size_no, max_ngram_size_no = rewrite_query(lemmatized_query_no)
        rewritten, min_ngram_size, max_ngram_size = rewritten_no, min_ngram_size_no, max_ngram_size_no
    if query_yes and query_no:
        min_ngram_size = min(min_ngram_size_yes, min_ngram_size_no)
        max_ngram_size = max(max_ngram_size_yes, max_ngram_size_no)
        rewritten = f"({rewritten_yes}) & (1 – ({rewritten_no}))"

    cv = CountVectorizer(lowercase=True, binary=True, preprocessor=extract_lemmas, ngram_range=(min_ngram_size, max_ngram_size))
    sparse_matrix = cv.fit_transform(documents)
    dense_matrix = sparse_matrix.todense()
    td_matrix = np.asarray(dense_matrix.T,dtype=bool)
    t2i = cv.vocabulary_
    
    hits_vector = eval(rewritten, {
        "td_matrix": td_matrix,
        "t2i": t2i,
        "get_term_vector": get_term_vector,
        "np": np,
        "__builtins__": {}
    })
    hits_vector = np.asarray(hits_vector).ravel().astype(bool)
    hits_list = np.where(hits_vector)[0]
    shown = min(N,len(hits_list))
    fitting_restaurants = []
    for doc_idx in hits_list[:shown]:
        fitting_restaurants.append(documents[doc_idx].split('\n')[0])
    return fitting_restaurants


def get_tf_idf_scores(query, documents):
    ngram_size = len(query.split())
    lemmatized_query = extract_lemmas(query)

    tfv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", ngram_range=(1, ngram_size))
    tf_matrix = tfv.fit_transform(documents).T.tocsr()
    query_vec = tfv.transform([lemmatized_query]).tocsc()
    hits = np.dot(query_vec, tf_matrix) 

    doc_ids_and_scores = dict(zip(hits.nonzero()[1], np.array(hits[hits.nonzero()])[0]))
    return doc_ids_and_scores
 

def tf_idf_search(query_yes, query_no, documents):
    if not (query_yes and query_no):
        return []
    if query_yes:
        ids_and_scores_yes = get_tf_idf_scores(query_yes, documents)
    if query_no:
        ids_and_scores_no = get_tf_idf_scores(query_no, documents)
    if query_yes and query_no:
        ids_and_scores = [(doc_idx, score - ids_and_scores_no.get(doc_idx, 0)) for (doc_idx, score) in ids_and_scores_yes.items()]
        ids_and_scores = sorted(ids_and_scores, key=lambda x: -x[1])
    elif query_yes:
        ids_and_scores = sorted(ids_and_scores_yes.items(), key=lambda x: -x[1])
    else:
        ids_and_scores = sorted(ids_and_scores_no.items(), key=lambda x: x[1])

    fitting_restaurants = []
    for (doc_idx, score) in ids_and_scores[:N]:
        fitting_restaurants.append(documents[doc_idx].split('\n')[0])
    return fitting_restaurants


def semantic_search(query, documents, doc_embeddings):
    query_embedding = model.encode(query)
    cosine_similarities = np.dot(query_embedding, doc_embeddings.T)
    ranked_doc_indices = np.argsort(cosine_similarities)[::-1]

    fitting_restaurants = []
    for i in range(N):
        doc_idx = ranked_doc_indices[i]
        fitting_restaurants.append(documents[doc_idx].split('\n')[0])
    return fitting_restaurants