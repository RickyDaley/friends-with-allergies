from flask import Flask, render_template, request, jsonify
import data_processing as dp
import pandas as pd
import json


app = Flask(__name__)
engine = 'semantic'
sem_show, bool_tfidf_show = True, False

data = pd.DataFrame()
review_dict, menu_dict, embed_review_dict, embed_menu_dict = {}, {}, {}, {}
embed_compiled_documents = []


@app.route('/')
def init():
    global data, review_dict, menu_dict, embed_review_dict, embed_menu_dict, compiled_documents, embed_compiled_documents
    data, review_dict, menu_dict, embed_review_dict, embed_menu_dict = dp.initialise_index()
    compiled_documents = ['\n'.join(menu_dict[key]) + '\n' + '\n'.join(review_dict[key]) for key in menu_dict.keys()]
    embed_compiled_documents = dp.model.encode(compiled_documents)
    return render_template('index.html', engine=engine, sem_show=sem_show, bool_tfidf_show=bool_tfidf_show)


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
    elif 'tf_idf' in request.form:
        engine = 'tf_idf'
        sem_show = False
        bool_tfidf_show = True
    else:
        return render_template('error.html', error_msg="Search engine is invalid or does not exist")
    return render_template('index.html', sem_show=sem_show, bool_tfidf_show=bool_tfidf_show,engine=engine)


def doc_ids_to_data_entries(matched_docs):
    matched_rests = [list(menu_dict.keys())[idx] for idx in matched_docs]
    matches_table = data[data.Name.isin(matched_rests)].sort_values(by='Name', key=lambda col: [matched_rests.index(v) for v in col.values])
    matching_entries = list(json.loads(matches_table.T.to_json()).values())
    return matches_table, matching_entries


def dict_values_to_list(dictionary):
    key_map = {}
    val_list = []
    idx = 0
    for key, val in dictionary.items():
        val_list.extend(val)
        key_map[key] = (idx, idx + len(val))
        idx += len(val)
    return key_map, val_list


def find_rest_for_idx(key_map, idx):
    for key, borders in key_map.items():
        if idx in range(*borders):
            return key
    return None

def get_matching_scores(key_map, matched_docs):
    # Score is assigned according to how many dishes containing a query term have been found for a given restaurant
    scores = {key: 0 for key in key_map.keys()}
    for idx in matched_docs:
        rest = find_rest_for_idx(key_map, idx)
        scores[rest] += 1
    return scores


@app.route('/search_single', methods=['POST'])
def search_single():
    query = request.form.get('query', '')
    if not query:
        return render_template('error.html', error_msg="You forgot to enter a search term.")
    matched_docs = dp.semantic_search(query, embed_compiled_documents, threshold=0.3)
    matches_table, matching_entries = doc_ids_to_data_entries(matched_docs)
    print(matching_entries)
    script, div, resources = dp.plot_freq(pd.DataFrame(matches_table), 'Rating (out of 6)')
    
    # Pass the search terms back to template
    return render_template('index.html', 
                         sem_show=sem_show, 
                         bool_tfidf_show=bool_tfidf_show, 
                         matches=matching_entries,
                         engine=engine,
                         chart_script=script,
                         chart_div=div,
                         chart_resources=resources,
                         query=query)


@app.route('/search_double', methods=['POST'])
def search_double():
    query_yes = request.form.get('query_yes', '')
    query_no = request.form.get('query_no', '')
    
    # Validate input - at least one query should be provided
    if not query_yes and not query_no:
        return render_template('error.html', error_msg="Please enter at least one search term (green flag or red flag)")

    # ============== GREEN FLAG FILTERING (Issue #35) ==============
    # Search for dishes/menus that match the green flag query
    rest_to_dish_map, dishes_list = dict_values_to_list(menu_dict)
    
    matching_dishes = {}  # Store matched dishes for display
    
    if query_yes:
        # Perform search based on selected engine
        if engine == 'boolean':
            matched_docs = dp.boolean_search(query_yes, dishes_list)
        elif engine == 'tf_idf':
            matched_docs = dp.tf_idf_search(query_yes, dishes_list)
        else:
            return render_template('error.html', error_msg="Wrong search engine name or no search engine provided")
        
        # Calculate matching scores and get matched dishes for each restaurant
        matching_scores = get_matching_scores(rest_to_dish_map, matched_docs)
        
        # Store matched dishes for display (evidence of match)
        for idx in matched_docs:
            rest_name = find_rest_for_idx(rest_to_dish_map, idx)
            if rest_name:
                if rest_name not in matching_dishes:
                    matching_dishes[rest_name] = []
                matching_dishes[rest_name].append(dishes_list[idx])
        
        # Filter to only restaurants with matches (score > 0)
        matching_scores = [(key, val) for key, val in matching_scores.items() if val > 0]
        matching_scores.sort(key=lambda x: -x[1])  # Sort by match count descending
        matched_restaurant_names = [key for key, _ in matching_scores]
    else:
        # No green flag - start with all restaurants
        matched_restaurant_names = list(menu_dict.keys())
    
    # ============== RED FLAG FILTERING (Issue #36) ==============
    # If red flag is specified, filter out restaurants with low general allergy score
    filtered_out_info = []
    if query_no:
        # Apply general allergy score filtering
        matched_restaurant_names, filtered_out_info = dp.filter_by_general_allergy_score(
            matched_restaurant_names, 
            review_dict, 
            embed_review_dict,
            threshold=dp.RED_FLAG_NEGATIVE_THRESHOLD
        )
    
    # Handle case where no restaurants match
    if not matched_restaurant_names:
        return render_template('index.html', 
                             sem_show=sem_show, 
                             bool_tfidf_show=bool_tfidf_show, 
                             matches=[],
                             engine=engine,
                             query_yes=query_yes,
                             query_no=query_no,
                             no_results_message="No restaurants found matching your criteria. Try different search terms.")
    
    # Convert restaurant names to data entries
    matching_ids = [list(menu_dict.keys()).index(name) for name in matched_restaurant_names if name in menu_dict.keys()]
    matches_table, matching_entries = doc_ids_to_data_entries(matching_ids)
    
    # Add matched dishes info to each restaurant entry for display
    for entry in matching_entries:
        rest_name = entry.get('Name', '')
        if rest_name in matching_dishes:
            # Limit to first 5 matched dishes for display
            entry['matched_dishes'] = matching_dishes[rest_name][:5]
            entry['total_matches'] = len(matching_dishes[rest_name])

    script, div, resources = dp.plot_freq(pd.DataFrame(matches_table), 'Rating (out of 6)')
    
    # Pass the search terms back to template
    return render_template('index.html', 
                         sem_show=sem_show, 
                         bool_tfidf_show=bool_tfidf_show, 
                         matches=matching_entries,
                         engine=engine,
                         chart_script=script,
                         chart_div=div,
                         chart_resources=resources,
                         query_yes=query_yes,
                         query_no=query_no,
                         filtered_count=len(filtered_out_info))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_msg="A page with the URL you've entered does not exist.")

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', error_msg="Internal Server Error. Our bad, sorry!")


# ============== API ENDPOINTS ==============

@app.route('/api/search', methods=['POST'])
def api_search():
    """
    JSON API endpoint for search
    
    Request body (JSON):
    {
        "type": "semantic" | "boolean" | "tf_idf",
        "query": "search term" (for semantic),
        "query_yes": "include term" (for boolean/tf_idf),
        "query_no": "exclude term" (for boolean/tf_idf)
    }
    
    Returns: JSON with search results
    """
    data_json = request.get_json()
    
    if not data_json:
        return jsonify({'error': 'No JSON data provided', 'results': []}), 400
    
    search_type = data_json.get('type', 'semantic')

    if search_type == 'semantic':
        query = data_json.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided', 'results': []}), 400
        matches = dp.semantic_search(query, documents, doc_embeddings)
    elif search_type in ('boolean', 'tf_idf'):
        query_yes = data_json.get('query_yes', '')
        query_no = data_json.get('query_no', '')
        if not query_yes and not query_no:
            return jsonify({'error': 'No query provided', 'results': []}), 400
        if search_type == 'boolean':
            matches = dp.boolean_search(query_yes, query_no, documents)
        else:
            matches = dp.tf_idf_search(query_yes, query_no, documents)
    else:
        return jsonify({'error': f'Invalid search type: {search_type}', 'results': []}), 400
    
    matching_entries = list(json.loads(data[data.Name.isin(matches)].T.to_json()).values())
    
    return jsonify({
        'search_type': search_type,
        'count': len(matching_entries),
        'results': matching_entries
    })

@app.route('/api/restaurants', methods=['GET'])
def api_restaurants():
    """
    Get all restaurants with optional filters
    
    Query parameters:
    - cuisine: Filter by cuisine type (e.g., "Italian")
    - location: Filter by location (e.g., "Kallio")
    - limit: Max number of results (default: 50)
    """
    cuisine = request.args.get('cuisine', '')
    location = request.args.get('location', '')
    limit = request.args.get('limit', 50, type=int)
    
    filtered = data.copy()
    
    if cuisine:
        filtered = filtered[filtered['Cuisine'].str.contains(cuisine, case=False, na=False)]
    if location:
        filtered = filtered[filtered['Location'].str.contains(location, case=False, na=False)]
    
    filtered = filtered.head(limit)
    results = list(json.loads(filtered.T.to_json()).values())
    
    return jsonify({
        'count': len(results),
        'filters': {'cuisine': cuisine, 'location': location},
        'restaurants': results
    })

@app.route('/api/restaurant/<name>', methods=['GET'])
def api_restaurant_detail(name):
    """
    Get details for a specific restaurant by name
    """
    restaurant = data[data['Name'].str.lower() == name.lower()]
    
    if restaurant.empty:
        return jsonify({'error': 'Restaurant not found'}), 404
    
    result = list(json.loads(restaurant.T.to_json()).values())[0]
    return jsonify({'restaurant': result})

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'restaurants_loaded': len(data),
        'documents_indexed': len(documents)
    })

"""
=============================================================================
API TESTING COMMANDS (run in terminal while Flask is running)
=============================================================================

# Health Check
curl http://localhost:5001/api/health

# Semantic Search
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{"type": "semantic", "query": "vegan gluten free"}'

# Boolean Search (include pizza, exclude meat)
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{"type": "boolean", "query_yes": "pizza", "query_no": "meat"}'

  # TF-IDF Search
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{"type": "tf_idf", "query_yes": "italian pasta", "query_no": ""}'

# List Italian Restaurants (limit 5)
curl "http://localhost:5001/api/restaurants?cuisine=Italian&limit=5"

=============================================================================
"""


@app.route('/test500')
def test_500():
    1 / 0  


if __name__ == "__main__":
    #Change the Flask app to run on a different port than 5000 to prevent port 5000 being used by AirTunes problem
    app.run(debug=False, port=5001) # change debug to false to see the custom 500 error pages instead of Flask's default error pages