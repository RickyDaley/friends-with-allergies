from flask import Flask, render_template, request
import data_processing as dp
import pandas as pd
import json


app = Flask(__name__)
engine = 'semantic'
sem_show, bool_tfidf_show = True, False

data = pd.DataFrame()
documents = []
doc_embeddings = pd.DataFrame()


@app.route('/')
def init():
    global data, documents, doc_embeddings
    data, documents, doc_embeddings = dp.initialise_index()
    return render_template('index.html', sem_show=sem_show, bool_tfidf_show=bool_tfidf_show)


@app.route('/switch', methods=['POST'])
def switch_engine():
    global engine, sem_show, bool_tfidf_show
    if 'semantic' in request.form:
        engine = 'semantic'
        sem_show = True
        bool_tfidf_show = False
    elif 'boolean' in request.form:
        engine = 'boolean'
        sem_show = False
        bool_tfidf_show = True
    else:
        engine = 'tf_idf'
        sem_show = False
        bool_tfidf_show = True
    return render_template('index.html', sem_show=sem_show, bool_tfidf_show=bool_tfidf_show)


@app.route('/search', methods=['POST'])
def search():
    if engine == 'semantic':
        query = request.form.get('query')
        matches = dp.semantic_search(query, documents, doc_embeddings)
    elif engine in ('boolean', 'tf_idf'):
        query_yes = request.form.get('query_yes')
        query_no = request.form.get('query_no')
        matches = dp.boolean_search(query_yes, query_no, documents) if engine == 'boolean' else dp.tf_idf_search(query_yes, query_no, documents)
    else:
        return render_template('error.html', error_msg="Wrong search engine name or no search engine provided")
    matching_entries = list(json.loads(data[data.Name.isin(matches)].T.to_json()).values())
    print(matching_entries)
    return render_template('index.html', sem_show=sem_show, bool_tfidf_show=bool_tfidf_show, matches=matching_entries)


if __name__ == "__main__":
    app.run(debug=True)