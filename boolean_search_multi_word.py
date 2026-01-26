import os
import numpy as np
import nltk
from math import inf
from sklearn.feature_extraction.text import CountVectorizer

stemmer = nltk.stem.SnowballStemmer('english') # supports finnish and swedish as well
nltk.download("wordnet")
lemmatizer = nltk.wordnet.WordNetLemmatizer()


def load_documents(filename):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = os.path.join(__location__, filename)
    with open(absolute_path) as f:
        full_text = f.read()
        documents = full_text.split('</article>')
        documents = [' '.join(doc.strip().split("\n")[1:]) for doc in documents]
        return documents
    

def extract_stems(docs):
    stem_sentence = lambda doc: ' '.join([stemmer.stem(word)for word in doc.lower().split()])
    if isinstance(docs, str):
        return stem_sentence(docs)
    return [stem_sentence(doc) for doc in docs]


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


def search_tool(documents):
    print()
    print("*Boolean search tool*")
    print("(Enter empty line to stop.)")
    print()
    
    while True:
        query = input("Please enter the query: ")
        if query == "":
            print("Good bye!!!")
            break

        # stemmed_query = extract_stems(query)
        lemmatized_query = extract_lemmas(query)
        
        rewritten, min_ngram_size, max_ngram_size = rewrite_query(lemmatized_query) #or stemmed_query

        print("...")
        print("Query: '" + query + "'")
        print("Rewritten:", rewritten)
        print("...")

        cv = CountVectorizer(lowercase=True, binary=True, preprocessor=extract_lemmas, ngram_range=(min_ngram_size, max_ngram_size)) # or preprocessor=extract_stems
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
    search_tool(documents)


if __name__ == "__main__":
    main()