import os
from sklearn.feature_extraction.text import CountVectorizer


def load_documents(filename):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = os.path.join(__location__, filename)
    with open(absolute_path) as f:
        full_text = f.read()
        documents = full_text.split('</article>')
        documents = [' '.join(doc.strip().split("\n")[1:]) for doc in documents]
        return documents


filename = "enwiki-20181001-corpus.1000-articles.txt"
documents = load_documents(filename)

cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(documents)
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names_out()

t2i = cv.vocabulary_

# parser
d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

def rewrite_token(t):
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query))) # Eval runs the string as a Python command
    print()