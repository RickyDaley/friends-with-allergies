# tf-idf: relevance-ranked search
import numpy as np
import os
import nltk
from math import inf
from sklearn.feature_extraction.text import TfidfVectorizer


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


def first_words_in_doc(file):
    words = " ".join(file.split()[:50])
    return words


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


def tf_idf_search(query, file):
    ngram_size = len(query.split())
    # stemmed_query = extract_stems(query)
    lemmatized_query = extract_lemmas(query)

    # Vectorize query string
    tfv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", ngram_range=(ngram_size, ngram_size)) # ngram_range=(1, ngram_size) for a non-strict search
    tf_matrix = tfv.fit_transform(file).T.tocsr()
    query_vec = tfv.transform([lemmatized_query]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, tf_matrix) 

    # Rank hits
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]),
               reverse=True)

    # Output results
    print("Your query '{:s}' matches the following documents:".format(query))
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids[:6]):
        print("Doc #{:d} (score: {:.4f})".format(i, score))
        print(first_words_in_doc(file[doc_idx]), "\n")


def main():
    file = "enwiki-20181001-corpus.1000-articles.txt"
    file = load_documents(file)
    
    query = input("Please enter an query:")
    
    tf_idf_search(query, file)


main()