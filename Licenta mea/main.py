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
            popularity,
            trailer TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INTEGER,
            occupation TEXT,
            sex TEXT CHECK(sex IN ('M', 'F')),
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_selected_movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
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

    # Filter ratings to include only those with a rating of 3 or higher
    ratings_df = ratings_df[ratings_df['rating'] >= 3]

    # Check if filtered ratings data is available
    if ratings_df.empty:
        print("No ratings of 3 or higher are available to train the model.")
        trainset, algo = None, None  # Ensure they are None if initialization fails
        return
    else:
        print(f"Filtered ratings data loaded: {ratings_df.shape[0]} records (rating >= 3)")

    # Create a Surprise dataset with the filtered ratings
    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        print("Dataset successfully created and trainset built with ratings >= 3.")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        trainset, algo = None, None
        return

    # Train the SVD algorithm
    try:
        algo = SVD()
        algo.fit(trainset)
        print("Collaborative filtering model trained successfully.")
    except Exception as e:
        print(f"Error training the collaborative filtering model: {e}")
        trainset, algo = None, None


# Get Collaborative Filtering Recommendations for a User
def get_collaborative_filtering_recommendations(user_id, top_n=10):
    global trainset, algo

    # Check if trainset and algo are initialized
    if trainset is None or algo is None:
        print("Collaborative filtering model is not initialized.")
        return []

    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(testset)

    # Add debugging output
    if not predictions:
        print(f"No predictions for user {user_id}")
    else:
        # Filter and sort predictions based on estimated ratings
        filtered_predictions = [prediction for prediction in predictions if prediction.est >= 3]
        
        # Debugging: Print out filtered predictions with their estimated ratings
        print(f"All predictions for user {user_id}:")
        for prediction in predictions:
            print(f"Movie ID: {prediction.iid}, Estimated Rating: {prediction.est}")
        
        print(f"Filtered predictions (rating >= 3) for user {user_id}:")
        for prediction in filtered_predictions:
            print(f"Movie ID: {prediction.iid}, Estimated Rating: {prediction.est}")

        # Sort filtered predictions by estimated rating in descending order
        filtered_predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top_n recommendations with rating >= 3
        recommendations = [prediction.iid for prediction in filtered_predictions[:top_n]]
        print(f"Final Recommendations for user {user_id} (rating >= 3): {recommendations}")
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
# Function to fetch the actors of a movie from TMDb
def get_actors(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching actors for movie ID {movie_id}: {response.status_code}")
        return []

    data = response.json()
    cast = data.get('cast', [])
    actors = [member['name'] for member in cast if member['known_for_department'] == 'Acting']
    return ', '.join(actors)


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
        popularity = movie.get('popularity', 0)  # Default to 0 if popularity is missing
        # Fetch trailer URL
        trailer_url = None
        # print(f"Trailer URL for '{movie['title']}': {trailer_url}")

        try:
            cursor.execute('''
                INSERT INTO movies (id, title, poster_path, overview, release_date, genres, trailer,popularity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                movie_id,
                movie['title'],
                movie['poster_path'],
                movie.get('overview', ''),
                movie.get('release_date', ''),
                genres,
                popularity,
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

# Define relevant genres with TMDb genre IDs
relevant_genres = {
    "Action": 28,
    "Drama": 18,
    "Adventure": 12,
    "Sci-Fi": 878,
    "Thriller": 53,
    "Comedy": 35,
    "Horror": 27,
    "Romance": 10749
}


def fetch_popular_movies_by_genre(genre_id, limit=5):
    """
    Fetch popular movies for a specific genre from TMDb.

    Args:
    - genre_id (int): The ID of the genre on TMDb.
    - limit (int): Number of top movies to return.

    Returns:
    - list of tuples: Each tuple contains (movie_id, title, poster_path).
    """
    url = f'https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_id}&sort_by=popularity.desc'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching movies for genre ID {genre_id}: {response.status_code}")
        return []

    data = response.json()
    movies = data.get('results', [])
    
    # Extract only the top 'limit' movies with necessary details
    popular_movies = [(movie['id'], movie['title'], movie['poster_path']) for movie in movies[:limit]]
    return popular_movies

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Capture user information from the form
        name = request.form['name'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        age = request.form.get('age')
        occupation = request.form.get('occupation')
        sex = request.form.get('sex')
        address = request.form['address'].strip()
        selected_movies = request.form.getlist('selected_movies')  # Get selected movies from form

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('tmdb_movies.db')
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            flash('Email already registered. Please log in.', 'warning')
            conn.close()
            return redirect(url_for('login'))

        # Insert user into the database
        cursor.execute('''
            INSERT INTO users (name, email, password, age, occupation, sex, address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, email, hashed_password, age, occupation, sex, address))
        user_id = cursor.lastrowid  # Get the newly created user's ID

        # Store selected movies for the user
        for movie_id in selected_movies:
            cursor.execute('''
                INSERT INTO user_selected_movies (user_id, movie_id)
                VALUES (?, ?)
            ''', (user_id, movie_id))

        conn.commit()
        conn.close()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    # Fetch popular movies by genre from TMDb
    genre_movies = {}
    for genre_name, genre_id in relevant_genres.items():
        genre_movies[genre_name] = fetch_popular_movies_by_genre(genre_id, limit=5)

    return render_template('register.html', genre_movies=genre_movies)




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


def get_content_based_recommendations_for_user(user_id, top_n=10):
    global movies_df, similarity

    # Ensure the similarity matrix is computed
    if similarity is None or movies_df is None:
        compute_similarity()
        if similarity is None or movies_df is None:
            return []

    # Connect to the database and get user's selected movies
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute('SELECT movie_id FROM user_selected_movies WHERE user_id = ?', (user_id,))
    selected_movie_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not selected_movie_ids:
        return []

    # Get indices of selected movies
    selected_indices = movies_df[movies_df['id'].isin(selected_movie_ids)].index.tolist()

    if not selected_indices:
        return []

    # Calculate similarity scores
    similarity_scores = similarity[selected_indices].mean(axis=0)

    # Get indices of movies sorted by similarity scores
    sorted_indices = similarity_scores.argsort()[::-1]

    # Exclude movies already selected by the user
    recommended_indices = [i for i in sorted_indices if movies_df.iloc[i]['id'] not in selected_movie_ids]

    # Get top_n recommendations
    top_recommended_indices = recommended_indices[:top_n]

    # Return the movie IDs of the recommended movies
    recommended_movie_ids = movies_df.iloc[top_recommended_indices]['id'].tolist()
    return recommended_movie_ids


@app.route('/')
@login_required
def index():
    try:
        page = int(request.args.get('page', 1))
        sort_by = request.args.get('sort_by', 'release_date')  # Default is 'release_date'
    except ValueError:
        flash('Invalid page number.', 'danger')
        return redirect(url_for('index'))

    per_page = 10
    offset = (page - 1) * per_page

    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    
    # Calculate total pages based on the number of movies
    cursor.execute('SELECT COUNT(*) FROM movies')
    total_movies = cursor.fetchone()[0]
    total_pages = (total_movies // per_page) + (1 if total_movies % per_page > 0 else 0)

    if page < 1 or page > total_pages:
        flash(f'Page {page} does not exist. Please choose a number between 1 and {total_pages}.', 'warning')
        conn.close()
        return redirect(url_for('index', sort_by=sort_by))

    # Choose ordering based on sort_by parameter
    if sort_by == 'popularity':
        cursor.execute('''
            SELECT id, title, poster_path, overview, release_date, genres, popularity
            FROM movies
            ORDER BY popularity DESC
            LIMIT ? OFFSET ?
        ''', (per_page, offset))
    else:  # sort_by == 'release_date'
        cursor.execute('''
            SELECT id, title, poster_path, overview, release_date, genres
            FROM movies
            ORDER BY release_date DESC
            LIMIT ? OFFSET ?
        ''', (per_page, offset))

    movies = cursor.fetchall()

    user_id = session.get('user_id')
    recommended_movies = []
    content_based_recommended_movies = []

    # Collaborative Filtering Recommendations
    if user_id and algo is not None:
        recommended_movie_ids = get_collaborative_filtering_recommendations(user_id, top_n=7)
        print(f"Collaborative Filtering Recommended movie IDs for user {user_id}: {recommended_movie_ids}")  # Debugging

        if recommended_movie_ids:
            cursor.execute(f'''
                SELECT id, title, poster_path
                FROM movies
                WHERE id IN ({','.join('?' * len(recommended_movie_ids))})
            ''', recommended_movie_ids)
            recommended_movies = cursor.fetchall()
            print(f"Collaborative Filtering Recommended movies: {recommended_movies}")  # Debugging

    # Content-Based Recommendations
    if user_id:
        content_based_movie_ids = get_content_based_recommendations_for_user(user_id, top_n=14)
        print(f"Content-Based Recommended movie IDs for user {user_id}: {content_based_movie_ids}")  # Debugging

        if content_based_movie_ids:
            cursor.execute(f'''
                SELECT id, title, poster_path
                FROM movies
                WHERE id IN ({','.join('?' * len(content_based_movie_ids))})
            ''', content_based_movie_ids)
            content_based_recommended_movies = cursor.fetchall()
            print(f"Content-Based Recommended movies: {content_based_recommended_movies}")  # Debugging

    conn.close()

    user_name = session.get('user_name')

    return render_template(
        'home.html',
        movies=movies,
        page=page,
        total_pages=total_pages,
        user_name=user_name,
        recommended_movies=recommended_movies,
        content_based_recommended_movies=content_based_recommended_movies,
        sort_by=sort_by
    )




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
