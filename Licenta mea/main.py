import requests
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
# Replace 'your_tmdb_api_key_here' with your actual TMDb API key
TMDB_API_KEY = '7fee067b60fd14ed0bd0013b0863045f'

# Flask configuration
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # secure random key

# Global variables
movies_df = None
similarity = None

# Function to create the database tables
def create_database():
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    # Create movies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title TEXT UNIQUE,
            poster_path TEXT,
            overview TEXT,
            release_date TEXT,
            genres TEXT,
            trailer TEXT
        )
    ''')
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            description TEXT,
            address TEXT
        )
    ''')
    # Create ratings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            UNIQUE(user_id, movie_id),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()
algo = None
trainset = None

# Helper function for content-based recommendations
def get_content_based_recommendations(movie_id, top_n=10):
    global similarity, movies_df
    if similarity is None or movies_df is None:
        compute_similarity()
        if similarity is None or movies_df is None:
            return []
    
    # Get the index of the movie that matches the movie_id
    movie_indices = movies_df.index[movies_df['id'] == movie_id].tolist()
    if not movie_indices:
        return []
    movie_index = movie_indices[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    similarity_scores = list(enumerate(similarity[movie_index]))
    
    # Sort the movies based on the similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top_n most similar movies (excluding the first one as it is the same movie)
    top_similar_movies = similarity_scores[1:top_n+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in top_similar_movies]
    
    # Return the top_n most similar movie IDs
    return movies_df.iloc[movie_indices]['id'].tolist()

# Hybrid Recommendation Function
def get_hybrid_recommendations(user_id, movie_id, top_n=7):
    # Content-based recommendations based on the movie ID
    content_based_recommendations = get_content_based_recommendations(movie_id, top_n)
    
    # Collaborative filtering recommendations based on the user ID
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n)
    
    # Merge both recommendation lists, prioritizing unique movies
    hybrid_recommendations = list(set(content_based_recommendations + collaborative_filtering_recommendations))
    
    # If the merged list has more than top_n, trim it
    return hybrid_recommendations[:top_n]


# Collaborative Filtering Model Initialization
def initialize_collaborative_filtering():
    global algo, trainset
    
    # Connect to the database and read the ratings
    conn = sqlite3.connect('tmdb_movies.db')
    ratings_df = pd.read_sql_query('SELECT user_id, movie_id, rating FROM ratings', conn)
    conn.close()

    # Check if ratings data is available
    if ratings_df.empty:
        print("Ratings data is empty. No data available to train the model.")
        return
    else:
        print(f"Ratings data loaded: {ratings_df.shape[0]} records")

    # Create a Surprise dataset
    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        print("Dataset successfully created and trainset built.")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # Train the SVD algorithm
    try:
        algo = SVD()
        algo.fit(trainset)
        print("Collaborative filtering model trained successfully.")
    except Exception as e:
        print(f"Error training the collaborative filtering model: {e}")

# Get Collaborative Filtering Recommendations for a User
def get_collaborative_filtering_recommendations(user_id, top_n=7):
    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(testset)

    # Add debugging output
    if not predictions:
        print(f"No predictions for user {user_id}")
    else:
        predictions.sort(key=lambda x: x.est, reverse=True)
        recommendations = [prediction.iid for prediction in predictions[:top_n]]
        print(f"Recommendations for user {user_id}: {recommendations}")
        return recommendations

    return []

# Decorator to require login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Function to get genre mapping
def get_genre_mapping():
    url = f'https://api.themoviedb.org/3/genre/movie/list' \
          f'?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching genres: {response.status_code}")
        return {}
    data = response.json()
    genres = {genre['id']: genre['name'] for genre in data.get('genres', [])}
    return genres

# Function to get trailer URL
def get_trailer_url(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/videos' \
          f'?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching trailer for movie ID {movie_id}: {response.status_code}")
        return None

    data = response.json()
    videos = data.get('results', [])
    for video in videos:
        if video['site'] == 'YouTube' and video['type'] == 'Trailer':
            return f"https://www.youtube.com/embed/{video['key']}"
    return None

# Function to fetch popular movies from TMDb and save them to the database
def fetch_and_save_movies(page=1, genre_mapping=None):
    if genre_mapping is None:
        genre_mapping = {}
    url = f'https://api.themoviedb.org/3/movie/popular' \
          f'?api_key={TMDB_API_KEY}&language=en-US&page={page}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return

    data = response.json()
    movies = data.get('results', [])

    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    new_movies = 0

    for movie in movies:
        movie_id = movie['id']

        # Get genres from genre IDs
        genre_ids = movie.get('genre_ids', [])
        genres_list = [genre_mapping.get(genre_id, '') for genre_id in genre_ids]
        genres = ', '.join(filter(None, genres_list))

        # Fetch trailer URL
        trailer_url = None
        # print(f"Trailer URL for '{movie['title']}': {trailer_url}")

        try:
            cursor.execute('''
                INSERT INTO movies (id, title, poster_path, overview, release_date, genres, trailer)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                movie_id,
                movie['title'],
                movie['poster_path'],
                movie.get('overview', ''),
                movie.get('release_date', ''),
                genres,
                trailer_url
            ))
            new_movies += 1
        except sqlite3.IntegrityError:
            # Movie already exists in the database
            continue
        except Exception as e:
            print(f"Error inserting movie '{movie['title']}': {e}")

        # Introduce a short pause to respect API rate limits
        time.sleep(0.2)

    conn.commit()
    conn.close()
    print(f"Added {new_movies} new movies from page {page}")

# Function to compute similarity matrix
def compute_similarity():
    global movies_df, similarity

    # Load movies data from the database into a DataFrame
    conn = sqlite3.connect('tmdb_movies.db')
    movies_df = pd.read_sql_query('SELECT id, title, overview, genres, poster_path FROM movies', conn)
    conn.close()

    if movies_df.empty:
        print("No movies found in the database to compute similarity.")
        similarity = None
        return

    # Fill missing values
    movies_df['overview'] = movies_df['overview'].fillna('')
    movies_df['genres'] = movies_df['genres'].fillna('')

    # Create 'tags' by concatenating 'overview' and 'genres'
    movies_df['tags'] = movies_df['overview'] + ' ' + movies_df['genres']

    # Initialize CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words='english')

    # Compute the count matrix
    count_matrix = cv.fit_transform(movies_df['tags'].values.astype('U'))

    # Compute the cosine similarity matrix
    similarity = cosine_similarity(count_matrix)

# Function to update the database with new movies
def update_movies(total_pages=5):
    genre_mapping = get_genre_mapping()
    for page in range(1, total_pages + 1):
        fetch_and_save_movies(page, genre_mapping)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        description = request.form['description'].strip()
        address = request.form['address'].strip()

        # Hash the password
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('tmdb_movies.db')
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            flash('Email already registered. Please log in.', 'warning')
            return redirect(url_for('login'))

        # Insert new user into database
        cursor.execute('''
            INSERT INTO users (name, email, password, description, address)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, hashed_password, description, address))
        conn.commit()
        conn.close()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    else:
        return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password']

        conn = sqlite3.connect('tmdb_movies.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'danger')
            return render_template('login.html')
    else:
        return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        flash('Invalid page number.', 'danger')
        return redirect(url_for('index'))

    per_page = 10
    offset = (page - 1) * per_page

    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM movies')
    total_movies = cursor.fetchone()[0]
    total_pages = (total_movies // per_page) + (1 if total_movies % per_page > 0 else 0)

    if page < 1 or page > total_pages:
        flash(f'Page {page} does not exist. Please choose a number between 1 and {total_pages}.', 'warning')
        conn.close()
        return redirect(url_for('index'))

    cursor.execute('''
        SELECT id, title, poster_path, overview, release_date, genres
        FROM movies
        ORDER BY release_date DESC
        LIMIT ? OFFSET ?
    ''', (per_page, offset))
    movies = cursor.fetchall()

    user_id = session.get('user_id')
    recommended_movies = []
    if user_id and algo is not None:
        recommended_movie_ids = get_collaborative_filtering_recommendations(user_id, top_n=7)
        print(f"Recommended movie IDs for user {user_id}: {recommended_movie_ids}")  # Debug: Printează recomandările

        if recommended_movie_ids:
            cursor.execute(f'''
                SELECT id, title, poster_path
                FROM movies
                WHERE id IN ({','.join('?' * len(recommended_movie_ids))})
            ''', recommended_movie_ids)
            recommended_movies = cursor.fetchall()
            print(f"Recommended movies: {recommended_movies}")  # Debug: Printează filmele recomandate

    conn.close()

    user_name = session.get('user_name')

    return render_template('home.html', movies=movies, page=page, total_pages=total_pages, user_name=user_name,
                           recommended_movies=recommended_movies)


# Search route
@app.route('/search')
@login_required
def search():
    query = request.args.get('query', '').strip()
    if not query:
        flash('Te rog introdu un termen de căutare.', 'warning')
        return redirect(url_for('index'))

    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, title, poster_path, overview, release_date, genres
        FROM movies
        WHERE title LIKE ?
        ORDER BY release_date DESC
    ''', (f'%{query}%',))
    movies = cursor.fetchall()
    conn.close()

    user_name = session.get('user_name')

    return render_template('search_results.html', movies=movies, query=query, user_name=user_name)

@app.route('/movie/<int:movie_id>', methods=['GET', 'POST'])
@login_required
def movie_detail(movie_id):
    global movies_df, similarity

    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()

    if request.method == 'POST':
        # Retrieve the rating from the form
        rating = request.form.get('rating')
        if rating and rating.isdigit():
            rating = int(rating)
            if 1 <= rating <= 5:
                user_id = session['user_id']
                try:
                    # Insert or update rating
                    cursor.execute('''
                        INSERT INTO ratings (user_id, movie_id, rating)
                        VALUES (?, ?, ?)
                        ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating
                    ''', (user_id, movie_id, rating))
                    conn.commit()
                    flash('Rating-ul tău a fost trimis cu succes!', 'success')
                except Exception as e:
                    flash('A apărut o eroare la trimiterea rating-ului.', 'danger')
                    print(f"Error inserting/updating rating: {e}")
            else:
                flash('Valoare invalidă pentru rating. Te rog să alegi un rating între 1 și 5.', 'danger')
        else:
            flash('Input de rating invalid.', 'danger')
        return redirect(url_for('movie_detail', movie_id=movie_id))

    # Handle the GET request
    cursor.execute('''
        SELECT id, title, poster_path, overview, release_date, genres, trailer
        FROM movies
        WHERE id = ?
    ''', (movie_id,))
    movie = cursor.fetchone()

    if not movie:
        flash('Filmul nu a fost găsit.', 'warning')
        conn.close()
        return redirect(url_for('index'))

    user_name = session.get('user_name')
    user_id = session.get('user_id')

    # Ensure the similarity matrix is computed
    if movies_df is None or similarity is None:
        compute_similarity()

    # Verify again after calling `compute_similarity()`
    if movies_df is None or similarity is None:
        flash('An error occurred while computing similarity matrix.', 'danger')
        conn.close()
        return redirect(url_for('index'))
    
    initialize_collaborative_filtering()
    # Get content-based recommendations
    content_based_movie_ids = get_content_based_recommendations(movie_id, top_n=10)
    
    # Get hybrid recommendations
    hybrid_recommended_movie_ids = get_hybrid_recommendations(user_id, movie_id, top_n=10)
    
    # Fetch content-based recommended movie details from the database
    if content_based_movie_ids:
        placeholders = ','.join('?' for _ in content_based_movie_ids)
        cursor.execute(f'''
            SELECT id, title, poster_path
            FROM movies
            WHERE id IN ({placeholders})
        ''', content_based_movie_ids)
        content_based_recommended_movies = cursor.fetchall()
    else:
        content_based_recommended_movies = []

    # Fetch hybrid recommended movie details from the database
    if hybrid_recommended_movie_ids:
        placeholders = ','.join('?' for _ in hybrid_recommended_movie_ids)
        cursor.execute(f'''
            SELECT id, title, poster_path
            FROM movies
            WHERE id IN ({placeholders})
        ''', hybrid_recommended_movie_ids)
        hybrid_recommended_movies = cursor.fetchall()
    else:
        hybrid_recommended_movies = []

    # Calculate average rating
    cursor.execute('''
        SELECT AVG(rating) FROM ratings WHERE movie_id = ?
    ''', (movie_id,))
    avg_rating = cursor.fetchone()[0]
    avg_rating = round(avg_rating, 2) if avg_rating else None
    trailer_url = get_trailer_url(movie_id)

    # Get user's rating for this movie, if available
    cursor.execute('''
        SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ?
    ''', (user_id, movie_id))
    user_rating = cursor.fetchone()
    user_rating = user_rating[0] if user_rating else None

    conn.close()
    
    return render_template('movie_detail.html',
                           movie=movie,
                           user_name=user_name,
                           avg_rating=avg_rating,
                           user_rating=user_rating,
                           trailer_url=trailer_url,
                           content_based_recommended_movies=content_based_recommended_movies,
                           hybrid_recommended_movies=hybrid_recommended_movies)



# Autocomplete route (if needed)
@app.route('/autocomplete', methods=['GET'])
@login_required
def autocomplete():
    term = request.args.get('term', '')
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute('SELECT title FROM movies WHERE title LIKE ?', (f'%{term}%',))
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(results)
#nu face nimic
@app.route('/movie_titles')
def movie_titles():
    term = request.args.get('term', '').strip()
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute('SELECT title FROM movies WHERE title LIKE ?', (f'%{term}%',))
    movies = cursor.fetchall()
    conn.close()
    movie_titles = [title[0] for title in movies]
    return jsonify(movie_titles)

if __name__ == '__main__':
    create_database()
    update_movies()
    compute_similarity()
    initialize_collaborative_filtering()
   # get_collaborative_filtering_recommendations(1, top_n=10)
    app.run(debug=True)
