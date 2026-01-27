# friends-with-allergies

Hi, we are *Friends with Allergies* team!
<br>
This is a repository for group assignments in the *Building Natural Language Processing Applications* course (Spring 2026).

Here are the group memebers:
  1. Yi Li
  2. Ricky
  3. Iuliia
  4. Qing


## Dependencies / Setup

This project uses a small Boolean search engine implemented in Python.  
The stemming + quoted exact match logic is implemented using only the Python standard library (`search_stem_exact.py`), but the Boolean indexing uses `scikit-learn` (for `CountVectorizer`) and `numpy`.

### Requirements
- Python 3
- `numpy`
- `scikit-learn`
- `Sentence Transformer`

### Install (recommended in a virtual environment)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn

## Run
python3 boolean_search_stem_exact.py
