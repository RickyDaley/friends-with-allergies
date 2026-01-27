import os
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_documents(filename):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = os.path.join(__location__, filename)
    with open(absolute_path) as f:
        full_text = f.read()
        documents = full_text.split('</article>')
        documents = [' '.join(doc.strip().split("\n")[1:]) for doc in documents]
        return documents
    
def print_doc_snippet(doc):
    #Show beginning 50 words of the document(s) found
    words = doc.split()
    return " ".join(words[:50]) + " ..."

def semantic_search(query, query_embedding, documents, doc_embeddings):
    cosine_similarities = np.dot(query_embedding, doc_embeddings.T)
    ranked_doc_indices = np.argsort(cosine_similarities)[::-1]  # Sort descending   

    # Output results, print 5 most relevant results
    print(f"Showing 5 most relevant matches:")
    print()
    for i, doc_idx in enumerate(ranked_doc_indices[:6]):
        print(f"Doc #{i} (score: {cosine_similarities[doc_idx]:.4f})")
        print(print_doc_snippet(documents[doc_idx]))
        print()

    
def main():
    #Load and embed the document
    print('Loading the document: "1000 articles extracted from English Wikipedia..."')
    print('Please wait a second...')
    filename = "enwiki-20181001-corpus.1000-articles.txt"
    documents = load_documents(filename)    
    doc_embeddings = model.encode(documents)

    # Query input
    query = input("Please enter the query: ")
    print()
    query_embedding = model.encode(query)

    # Perfoming semantic search
    semantic_search(query, query_embedding, documents, doc_embeddings)


if __name__ == "__main__":
    main()

