from flask import Flask, render_template, request, jsonify
import data_processing as dp
import pandas as pd
import json


app = Flask(__name__)
engine = 'semantic'
sem_show, bool_tfidf_show = True, False

data = pd.DataFrame()
review_dict, menu_dict, embed_review_dict, embed_menu_dict = {}, {}, {}, {}


@app.route('/')
def init():
    global data, review_dict, menu_dict, embed_review_dict, embed_menu_dict
    data, review_dict, menu_dict, embed_review_dict, embed_menu_dict = dp.initialise_index()
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
    else:
        engine = 'tf_idf'
        sem_show = False
        bool_tfidf_show = True
    return render_template('index.html', sem_show=sem_show, bool_tfidf_show=bool_tfidf_show,engine=engine)


@app.route('/search', methods=['POST'])
def search():
    query_yes = request.form.get('query_yes', '')
    query_no = request.form.get('query_no', '')
    query = request.form.get('query', '')
    
    if engine == 'semantic':
        matches = dp.semantic_search(query, documents, doc_embeddings)
    elif engine in ('boolean', 'tf_idf'):
        matches = dp.boolean_search(query_yes, query_no, documents) if engine == 'boolean' else dp.tf_idf_search(query_yes, query_no, documents)
    else:
        return render_template('error.html', error_msg="Wrong search engine name or no search engine provided")
    
    matches_table = data[data.Name.isin(matches)]
    matching_entries = list(json.loads(matches_table.T.to_json()).values())
    print(matching_entries)

    script, div, resources = dp.plot_freq(pd.DataFrame(matches_table), 'Rating (out of 6)')
    
    # Pass the search terms back to template
    return render_template('index.html', 
                         sem_show=sem_show, 
                         bool_tfidf_show=bool_tfidf_show, 
                         matches=matching_entries,
                         engine=engine,
                         chart_script = script,
                         chart_div = div,
                         chart_resources = resources,
                         query_yes=query_yes,
                         query_no=query_no,
                         query=query)




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
  
if __name__ == "__main__":
    #Change the Flask app to run on a different port than 5000 to prevent port 5000 being used by AirTunes problem
    app.run(debug=True, port=5001)