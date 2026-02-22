import os
import json
import math
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import translators as ts
from bokeh.plotting import figure
from bokeh.layouts import row, Spacer
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

def load_restaurant_pictures():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = lambda x: os.path.join(__location__, x)
    
    pictures_path = absolute_path("data_raw/restaurant_pictures.csv")
    
    try:
        df = pd.read_csv(pictures_path, sep='\t',quotechar='"',doublequote=True,escapechar='\\',on_bad_lines='warn')
        pictures_dict = {}
        
        for _, row in df.iterrows():
            name = str(row['Restaurant']).strip()
            images_str = str(row.get('Images', '')).strip()
            
            if not images_str or images_str.lower() in ['nan', 'none', '']:
                continue
                
            urls = [u.strip() for u in images_str.split(', ') if u.strip()]
            pictures_dict[name] = urls
        
        print(f"Loaded {len(pictures_dict)} restaurants with additional photos")
        return pictures_dict
    
    except FileNotFoundError:
        print(f"Warning: {pictures_path} not found. No extra photos available.")
        return {}
    except Exception as e:
        print(f"Error loading restaurant_pictures.csv: {e}")
        return {}


def initialise_index():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = lambda x: os.path.join(__location__, x)
    data = pd.read_csv(absolute_path("data/restaurant_data.csv"), sep="\t", index_col=0)
    reviews = pd.read_csv(absolute_path("data/translated_review_data.csv"), sep="\t", index_col=0, lineterminator='\n')
    menus = pd.read_csv(absolute_path("data/translated_menu_data.csv"), sep="\t", index_col=0)
    review_dict, menu_dict, embed_review_dict, embed_menu_dict, pictures_dict = {}, {}, {}, {}, {}
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
    
    # Load pictures
    pictures_dict = load_restaurant_pictures()
    
    return data, review_dict, menu_dict, embed_review_dict, embed_menu_dict, pictures_dict


def extract_lemmas(docs):
    lem_sentence = lambda doc: ' '.join([lemmatizer.lemmatize(word)for word in doc.lower().split()])
    if isinstance(docs, str):
        return lem_sentence(docs)
    return [lem_sentence(doc) for doc in docs]


def fix_spaces_around_brackets(query_string):
    reconstructed_string = ''
    for i, c in enumerate(query_string):
        if c == '(':
            if i == len(query_string) - 1:
                break
            if i > 0 and query_string[i - 1] != ' ':
                reconstructed_string += ' '
            reconstructed_string += c
            if query_string[i + 1] != ' ':
                reconstructed_string += ' '
        elif c == ')':
            if i == 0:
                continue
            if query_string[i - 1] != ' ':
                reconstructed_string += ' '
            reconstructed_string += c
            if i < len(query_string) - 1 and query_string[i + 1] != ' ':
                reconstructed_string += ' '
        else:
            reconstructed_string += c
    return reconstructed_string.strip()


def rewrite_query(query):
    query = fix_spaces_around_brackets(query)
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
    if not documents or all(not doc.strip() for doc in documents):
        return []
    transl_query = translate_chunk(query)
    lemmatized_query = extract_lemmas(transl_query)
    rewritten, min_ngram_size, max_ngram_size = rewrite_query(lemmatized_query)
    print(rewritten)
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
    
    # added check for empty hits to avoid errors when converting to dict and sorting
    if hits is None or hits.size == 0 or hits.nnz == 0:
        return {}

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


def specific_allergy_score(all_reviews, embed_reviews, allergen, threshold=0.3):
    # Semantic search to match specific allergen
    specific_ids = semantic_search(allergen, embed_reviews, threshold=threshold)

    specific_allergy_reviews = []
    for i in specific_ids:
        specific_allergy_reviews.append(all_reviews[i])

    positive_count = 0
    # Sentiment Analysis on specific allergy reviews: positive for 1, neutral for 0
    for review in specific_allergy_reviews:
        sentiment_score = sentiment_analysis(review)
        if sentiment_score == 1:
            positive_count += 1
    
    total_reviews = len(specific_allergy_reviews)
    if total_reviews > 0:
        return positive_count / total_reviews
    else:
        return "Neutral"


def rank_restaurants(restaurant_names, review_dict, embed_review_dict, allergen, threshold=0.3):
    final_ranking_list = []

    for name in restaurant_names:
        if name not in review_dict or name not in embed_review_dict:
            continue

        current_reviews = review_dict[name]
        current_embeddings = embed_review_dict[name]

        general_score = general_allergy_score(current_reviews, current_embeddings, threshold)
        
        # If no allergy reviews for general reviews, give a default base score 0.4
        if general_score == "Neutral":
            base_score = 0.4 
        else:
            base_score = general_score

        specific_score = specific_allergy_score(current_reviews, current_embeddings, allergen, threshold)
        
        # If no allergy reviews for specific allergen, give a default base score 0.4
        if specific_score == "Neutral":
            specific_score = 0.4 
        else:
            specific_score = specific_score

        total_score = base_score * 0.35 + specific_score * 0.65
        final_ranking_list.append({
                "name": name,
                "total_score": total_score,
            })
        
    return sorted(final_ranking_list, key=lambda x: x["total_score"], reverse=True)
        

def plot_stats(data):
    def recalculate_gas(score):
            if score == 'Neutral':
                return 0
            score = float(score)
            if score < 0.7:
                return score + 0.3
            return score
    ratings = data['Rating (out of 6)'].apply(lambda x: 0 if math.isnan(x) else x).values
    review_counts = data['Review Count'].values
    allergy_scores = data['General Allergy Score'].apply(recalculate_gas).values
    counts, bin_edges = np.histogram(allergy_scores, bins=3, range=(0, 1))
    bins = []
    bin_names = ['Neutral', 'Safe', 'Safest']
    for i in range(3):
        bins.append(f"{round(bin_edges[i], 1)} – {round(bin_edges[i + 1], 1)}\n{bin_names[i]}")

    p1 = figure(height=300, title="Rating Density",
           toolbar_location=None)
    p2 = figure(x_range=bins, height=300, title="Distribution of General Allergy Scores",
           toolbar_location=None)
    
    p1.scatter(x=ratings, y=review_counts, color="#6CAF61", size=10, alpha=0.3)
    p1.title.text_font = "Futura"
    p1.title.text_font_style = "bold"
    p1.yaxis.axis_label = "Number of reviews"
    p1.yaxis.axis_label_text_font = "Futura"
    p1.xaxis.axis_label = "Rating (out of 6)"
    p1.xaxis.axis_label_text_font = "Futura"
    p1.background_fill_alpha = 0
    p1.border_fill_alpha = 0
    p1.y_range.start = 0
    p1.sizing_mode = "stretch_width"

    p2.vbar(x=bins, top=counts, width=0.9, color="#6CAF61", alpha=0.3)
    p2.title.text_font = "Futura"
    p2.title.text_font_style = "bold"
    p2.yaxis.axis_label = "Number of restaurants"
    p2.yaxis.axis_label_text_font = "Futura"
    p2.xaxis.axis_label = "General Allergy Score"
    p2.xaxis.axis_label_text_font = "Futura"
    p2.xgrid.grid_line_color = None
    p2.background_fill_alpha = 0
    p2.border_fill_alpha = 0
    p2.y_range.start = 0
    p2.sizing_mode = "stretch_width"

    viz_row = row(p1, Spacer(width=20), p2, sizing_mode="stretch_width")
    script, div = components(viz_row)
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