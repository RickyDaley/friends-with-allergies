import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# parser
d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

def load_documents(filename):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = os.path.join(__location__, filename)
    with open(absolute_path) as f:
        full_text = f.read()
        documents = full_text.split('</article>')
        documents = [' '.join(doc.strip().split("\n")[1:]) for doc in documents]
        return documents

def rewrite_token(t):
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))

def rewrite_query(query, t2i, td_matrix):
    tokens = query.split()
    parts = []
    for t in tokens:
        if t.lower() in ("and", "or", "not"):
            parts.append({"and":"&", "or":"|", "not":"1 -"}[t.lower()])
        elif t in ("(", ")"):
            parts.append(t)
        else:
            parts.append(f'get_term_vector("{t.lower()}", t2i, td_matrix)')
    return "(" + " ".join(parts) + ")"

def test_query(query,t2i,td_matrix):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query,t2i,td_matrix))


def print_doc_snippet(doc):
    #Show beginning 50 words of the document(s) found
    words = doc.split()
    return " ".join(words[:50]) + " ..."

def get_term_vector(term, t2i, td_matrix):
    #Returns zero vector (no matches) for unknown terms
    term = term.lower()
    if term in t2i:
        return td_matrix[t2i[term]]
    else:
        return np.zeros(td_matrix.shape[1], dtype=bool)

def search_tool(t2i,td_matrix,documents):
    
    print()
    print("*Boolean search tool*")
    print("(Enter empty line to stop.)")
    print()
    
    while True:
        query = input("Please enter the query: ")
        if query == "":
            print("Good bye!!!")
            break
        
        print("...")
        test_query(query,t2i,td_matrix)
        print("...")
        
        rewritten = rewrite_query(query,t2i,td_matrix)

       
        
        hits_vector = eval(rewritten, {
            "td_matrix": td_matrix,
            "t2i": t2i,
            "get_term_vector": get_term_vector,
            "np": np,
            "__builtins__": {}
        })
        hits_vector = np.asarray(hits_vector).ravel().astype(bool)

        hits_list = np.where(hits_vector)[0]

        count = len(hits_list)
        print(f"Found {count} matching document(s)")
        if count == 0:
            print()
            continue
       
       
        #Show top 5 matches
        n = 5
        shown = min(n,count)
        print(f"Showing first {shown} of {count}:")
        print()
        for rank, doc_idx in enumerate(hits_list[:n],1):
            print(f"Matching doc #{rank}: ")
            print(print_doc_snippet(documents[doc_idx]))
            print()

def main():
    #Load the document
    print('Loading the document: "1000 articles extracted from English Wikipedia..."')
    filename = "enwiki-20181001-corpus.1000-articles.txt"
    documents = load_documents(filename)

    
    cv = CountVectorizer(lowercase=True, binary=True)
    sparse_matrix = cv.fit_transform(documents)
    dense_matrix = sparse_matrix.todense()
    td_matrix = np.asarray(dense_matrix.T,dtype=bool)
    terms = cv.get_feature_names_out()
    t2i = cv.vocabulary_

    """
    print("Term-document matrix :")
    print(td_matrix)
    print(t2i)
    """
    
    print(f"Done! Vocabulary size: {len(terms)}")

    search_tool(t2i,td_matrix,documents)

if __name__ == "__main__":
    main()