import os
import json
import math
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import translators as ts
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
nltk.download('wordnet')
nltk.download('vader_lexicon')

model = SentenceTransformer('all-MiniLM-L6-v2')
sia = SentimentIntensityAnalyzer()
lemmatizer = nltk.wordnet.WordNetLemmatizer()


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


def translate_list(text_entries):
    transl_entries = []
    limit = 10000
    chunk = ''
    for entry in text_entries:
        test_chunk = chunk + '\n' + entry if chunk else entry
        if len(test_chunk) > limit:
            transl_chunk = translate_chunk(chunk)
            transl_entries.extend(transl_chunk.split('\n'))
            chunk = entry
        elif len(test_chunk) == limit:
            transl_chunk = translate_chunk(test_chunk)
            transl_entries.extend(transl_chunk.split('\n'))
            chunk = ''
        else:
            chunk = test_chunk
    if chunk:
        transl_chunk = translate_chunk(chunk)
        transl_entries.extend(transl_chunk.split('\n'))
    return transl_entries


def json_file_to_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        contents = f.read()
        d = json.loads(contents)
    for key, val in d.items():
        d[key] = np.asarray(val)
    return d


def load_review_embeds(dir_path):
    dict_readed = {}
    for filename in os.listdir(dir_path):
        with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
            contents = f.read()
            val = json.loads(contents)
        key = filename.replace(".txt", "").replace("_", "/")
        dict_readed[key] = np.asarray(val)
    return dict_readed
    

def initialise_index():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = lambda x: os.path.join(__location__, x)
    data = pd.read_csv(absolute_path("data/restaurant_data.csv"), sep="\t", index_col=0)
    reviews = pd.read_csv(absolute_path("data/translated_review_data.csv"), sep="\t", index_col=0, lineterminator='\n')
    menus = pd.read_csv(absolute_path("data/translated_menu_data.csv"), sep="\t", index_col=0)
    review_dict, menu_dict, embed_review_dict, embed_menu_dict = {}, {}, {}, {}
    for _, row in data.iterrows():
        rest_reviews = reviews[reviews.Restaurant == row.Name]["Review Text Eng"].values
        review_dict[row.Name] = rest_reviews
        #embed_review_dict[row.Name] = model.encode(rest_reviews)
        rest_menu = menus[menus.Restaurant == row.Name]["Menu Eng"].values
        menu_dict[row.Name] = rest_menu
        #embed_menu_dict[row.Name] = model.encode(rest_menu)
    # Load pre-saved embeddings
    embed_review_dict = load_review_embeds(absolute_path("data/embed_review_dict"))
    embed_menu_dict = json_file_to_dict(absolute_path("data/embed_menu_dict.json"))
    return data, review_dict, menu_dict, embed_review_dict, embed_menu_dict


def extract_lemmas(docs):
    lem_sentence = lambda doc: ' '.join([lemmatizer.lemmatize(word)for word in doc.lower().split()])
    if isinstance(docs, str):
        return lem_sentence(docs)
    return [lem_sentence(doc) for doc in docs]


def rewrite_query(query):
    tokens = query.split()
    operators = ("and", "or", "not", "&", "|", "(", ")")
    parts = []
    min_ngram_size, max_ngram_size = math.inf, 1
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


def boolean_search(query, documents):
    if not query:
        return []
    transl_query = translate_chunk(query)
    lemmatized_query = extract_lemmas(transl_query)
    rewritten, min_ngram_size, max_ngram_size = rewrite_query(lemmatized_query)

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
    return hits_list
 

def tf_idf_search(query, documents):
    if not query:
        return []
    ngram_size = len(query.split())
    lemmatized_query = extract_lemmas(query)

    tfv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", ngram_range=(1, ngram_size))
    tf_matrix = tfv.fit_transform(documents).T.tocsr()
    query_vec = tfv.transform([lemmatized_query]).tocsc()
    hits = np.dot(query_vec, tf_matrix) 

    doc_ids_and_scores = dict(zip(hits.nonzero()[1], np.array(hits[hits.nonzero()])[0]))
    doc_ids_and_scores = sorted(doc_ids_and_scores.items(), key=lambda x: -x[1])
    return [pair[0] for pair in doc_ids_and_scores]


def semantic_search(query, doc_embeddings, threshold=0.25):
    if query is None or doc_embeddings is None or np.array(doc_embeddings).shape == (0,):
        return []
    query_embedding = model.encode(query)
    cosine_similarities = np.dot(query_embedding, doc_embeddings.T)
    ranked_doc_indices = np.argsort(cosine_similarities)[::-1]

    best_similarity = cosine_similarities[ranked_doc_indices[0]]
    if best_similarity < threshold:
        return []
    
    doc_ids = []
    for i in range(len(doc_embeddings)):
        doc_idx = ranked_doc_indices[i]
        similarity = cosine_similarities[doc_idx]
        if similarity < threshold:
            break
        doc_ids.append(doc_idx)

    return doc_ids


def sentiment_analysis(text):
    #Input a text and use sentiment analysis to output if the review is negative (0) or positive (1)
    score = sia.polarity_scores(text)["compound"]
    return 1 if score >= 0 else 0


def get_all_allergy_reviews(all_reviews, embed_reviews, threshold=0.3):
    #Extract all allergy-related reviews from all the review of a restaurant and return them in a list
    allergy_query = """
    allergy allergic reaction anaphylaxis hypersensitivity
    gluten celiac
    dairy lactose
    food intolerance sensitivity dietary restrictions
    cross-contamination epinephrine EpiPen stomach ache, diarrhea """

    doc_ids = semantic_search(allergy_query, embed_reviews, threshold=threshold)
    allergy_reviews = [all_reviews[i] for i in doc_ids]

    return allergy_reviews


def general_allergy_score(all_reviews, embed_reviews, threshold=0.3):
    # Classify the allergy reviews into positive or negative
    # => Calculate the proportion of positive/total allergy reviews => the larger this proportion, the better result
    allergy_reviews = get_all_allergy_reviews(all_reviews, embed_reviews, threshold)

    positive_count = 0
    negative_count = 0

    for review in allergy_reviews:
        sentiment_score = sentiment_analysis(review)
        if sentiment_score == 1:
            positive_count += 1
        else:
            negative_count += 1
    
    total_allergy_reviews_number = len(allergy_reviews)
    if total_allergy_reviews_number > 0:
        positive_proportion = positive_count/total_allergy_reviews_number
    else:
        positive_proportion = "Neutral" # return "Neutral" if the reviews doesn't have any related to allergy

    return positive_proportion


def specific_allergy_score(specific_reviews, embed_specific_reviews, allergen, threshold=0.3):
    specific_query = f"{allergen} allergy safety cross-contamination" 

    specific_ids = semantic_search(specific_query, embed_specific_reviews, threshold=threshold)

    verified_reviews = []
    for i in specific_ids:
        text = specific_reviews[i].lower()
        if allergen.lower() in text or "allergy" in text:
            verified_reviews.append(specific_reviews[i])

    positive_count = 0
    negative_count = 0

    for review in verified_reviews:
        sentiment_score = sentiment_analysis(review)
        if sentiment_score == 1:
            positive_count += 1
        else:
            negative_count += 1
    
    total_allergy_reviews_number = len(verified_reviews)
    if total_allergy_reviews_number > 0:
        positive_proportion = positive_count/total_allergy_reviews_number
    else:
        positive_proportion = "Neutral" # return "Neutral" as a neutral score if the reviews doesn't have any related to allergy

    return  positive_proportion


def rank_restaurants(restaurant_data, allergen, threshold=0.3):
    final_ranking_list = []

    for restaurant in restaurant_data:
        current_reviews = restaurant['reviews']
        current_embeddings = None # got questions about embeddings

        general_allergy_score = general_allergy_score(current_reviews, current_embeddings, threshold)
        base_score = 0.4 if general_allergy_score == "Neutral" else general_allergy_score

        if general_allergy_score != "Neutral" and general_allergy_score < 0.7:
            continue
        specific_allergy_score = specific_allergy_score(current_reviews, current_embeddings, allergen, threshold)

        total_socre = general_allergy_score * 0.3 + specific_allergy_score * 0.7
        final_ranking_list.append({
                "name": restaurant['name'],
                "total_score": total_socre,
            })
        
    # return ranks based on the total score
    return sorted(final_ranking_list, key=lambda x: x["total_score"], reverse=True)
        

def plot_dist(data, column, nbins):
    x_raw = data[column]
    if column == 'Rating (out of 6)':
        x_raw = x_raw.apply(lambda x: 0 if math.isnan(x) else x)
    if column == 'General Allergy Score':
        def recalculate_gas(score):
            threshold = 0.25
            if score == 'Neutral':
                return 0
            score = float(score)
            if score <= threshold:
                return -2 * (score + threshold) - 0.5
            return (score - threshold) * 2  - 0.5
        x_raw = x_raw.apply(recalculate_gas)
    x_raw = x_raw.values
    counts, bin_edges = np.histogram(x_raw, bins=nbins)
    bins = []
    ndigits = 0 if column == "Review Count" else 1
    for i in range(nbins):
        bins.append(f"{round(bin_edges[i], ndigits)} – {round(bin_edges[i + 1], ndigits)}")

    #x, y = np.unique(x_raw, return_counts=True)
    p = figure(x_range=bins, height=350, width=800, title="Distribution of " + column,
           toolbar_location=None, tools="")
    p.vbar(x=bins, top=counts, width=0.9)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    
    script, div = components(p)
    return script, div, CDN.render()


# ============== RED FLAG FILTERING (Issue #36) ==============

# Configurable threshold - restaurants with negative proportion >= this value will be filtered out
RED_FLAG_NEGATIVE_THRESHOLD = 0.25  # 25% negative reviews threshold

def filter_by_general_allergy_score(restaurant_names, review_dict, embed_review_dict, threshold=RED_FLAG_NEGATIVE_THRESHOLD):
    """
    Filter restaurants based on their general allergy score.
    Removes restaurants where the proportion of negative allergy-related reviews >= threshold.
    
    Args:
        restaurant_names: List of restaurant names to filter
        review_dict: Dictionary mapping restaurant names to their reviews
        embed_review_dict: Dictionary mapping restaurant names to embedded reviews
        threshold: Maximum allowed proportion of negative allergy reviews (default 0.25)
    
    Returns:
        filtered_names: List of restaurant names that passed the filter
        filtered_out: List of tuples (name, negative_proportion, reason) for restaurants that were filtered out
    """
    filtered_names = []
    filtered_out = []
    
    for name in restaurant_names:
        if name not in review_dict or name not in embed_review_dict:
            # Keep restaurants with no review data (benefit of the doubt)
            filtered_names.append(name)
            continue
        
        all_reviews = review_dict[name]
        embed_reviews = embed_review_dict[name]
        
        if len(all_reviews) == 0 or embed_reviews.shape[0] == 0:
            # Keep restaurants with no reviews
            filtered_names.append(name)
            continue
        
        # Calculate general allergy score
        score = general_allergy_score(all_reviews, embed_reviews, threshold=0.3)
        
        if score == "Neutral":
            # No allergy-related reviews found - keep the restaurant
            filtered_names.append(name)
        else:
            positive_proportion = score
            negative_proportion = 1 - positive_proportion
            
            if negative_proportion >= threshold:
                # Too many negative allergy reviews - filter out
                filtered_out.append((
                    name, 
                    negative_proportion, 
                    f"Filtered: {negative_proportion:.0%} negative allergy reviews (threshold: {threshold:.0%})"
                ))
            else:
                # Passed the filter
                filtered_names.append(name)
    
    return filtered_names, filtered_out


def get_matching_dishes(query, dishes_list, rest_to_dish_map, search_func):
    """
    Find matching dishes for each restaurant based on the search query.
    
    Args:
        query: Search query string
        dishes_list: List of all dishes across all restaurants
        rest_to_dish_map: Mapping from restaurant name to (start_idx, end_idx) in dishes_list
        search_func: Search function to use (boolean_search or tf_idf_search)
    
    Returns:
        Dictionary mapping restaurant names to list of matching dishes
    """
    if not query:
        return {}
    
    matched_indices = search_func(query, dishes_list)
    
    matching_dishes = {}
    for idx in matched_indices:
        # Find which restaurant this dish belongs to
        for rest_name, (start, end) in rest_to_dish_map.items():
            if start <= idx < end:
                if rest_name not in matching_dishes:
                    matching_dishes[rest_name] = []
                matching_dishes[rest_name].append(dishes_list[idx])
                break
    
    return matching_dishes


if __name__ == "__main__":
    pass