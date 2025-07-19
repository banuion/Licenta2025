
import math
import requests
import sqlite3
import asyncio
import aiohttp
import tensorflow as tf 
import numpy as np
import speech_recognition as sr
import tempfile
import cv2
import pytesseract
import skimage.transform
from keras.models import model_from_json
from pydub import AudioSegment
from pytesseract import image_to_string
from tensorflow.keras import layers, regularizers, Model
from sklearn.model_selection import KFold
from surprise.model_selection import cross_validate 
from sklearn.decomposition import NMF
from flask_talisman import Talisman
import pandas as pd
from flask import Flask, abort, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import scipy.sparse as sparse 
import matplotlib.pyplot as plt
import implicit
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
#os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk1.8.0_241"  # actualizează cu calea ta către JDK
#os.environ["PYSPARK_PYTHON"] = r"python"  # Asigură-te că e folosită versiunea corectă
from PIL import Image as PILImage, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
# Pentru a evita confuzia, importăm subnumele 'image' din keras ca 'keras_image'
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
# Import din Keras/TensorFlow, îl redenumim ca keras_image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
#import findspark
#os.environ["SPARK_HOME"] = r"C:\Users\BANU\Desktop\lab10\Licenta mea\spark-3.5.5-bin-hadoop3\spark-3.5.5-bin-hadoop3"
#from pyspark.sql import SparkSession 
#spark = SparkSession.builder \
#    .master("local[*]") \
#    .config("spark.network.timeout", "600s") \
#    .config("spark.executor.heartbeatInterval", "60s") \
#    .config("spark.driver.bindAddress", "127.0.0.1") \
#    .getOrCreate()
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#findspark.init()
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
import re
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import datetime as dt
from surprise import SVD, Dataset, Reader,KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from keras.models import load_model
import joblib
from googletrans import Translator
from langdetect import detect
import pyotp
import qrcode
import io
import base64
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
logging.basicConfig(filename="rate_limit_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")
from sklearn.preprocessing import MultiLabelBinarizer

# Replace 'your_tmdb_api_key_here' with your actual TMDb API key
TMDB_API_KEY = '7fee067b60fd14ed0bd0013b0863045f'



# Flask configuration
app = Flask(__name__)
model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")
app.secret_key = 'your_secret_key_here'  # secure random key
Talisman(app, content_security_policy=None)

# Global variables
movies_df = None
similarity = None

def generate_2fa_secret():
    """
    Generează un secret unic pentru 2FA folosind pyotp.
    """
    return pyotp.random_base32()

def generate_qr_code_uri(email, secret):
    """
    Generează URI pentru TOTP care va fi scanat de aplicația de autentificare.
    Formatul recomandat de Google Authenticator: otpauth://totp/[issuer]:[account]?secret=[secret]&issuer=[issuer]
    """
    issuer = "MyFlaskApp"  # numele aplicației tale
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name=issuer)
    return totp_uri


def create_qr_code_base64(totp_uri):
    """
    Generează imaginea QR code pentru totp_uri și o returnează ca string base64,
    astfel încât să poată fi afișată în browser ca <img src='data:image/png;base64,...' />
    """
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str



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
    popularity REAL,
    trailer TEXT
    );
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
    address TEXT,
    twofa_secret TEXT,
    is_2fa_enabled INTEGER DEFAULT 0
    );

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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            review TEXT NOT NULL,
            sentiment TEXT CHECK(sentiment IN ('positive', 'negative')),
            grade INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clicks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS friend_requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id   INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            status      TEXT CHECK(status IN ('pending','accepted','denied')) DEFAULT 'pending',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sender_id, receiver_id),
            FOREIGN KEY(sender_id)   REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(receiver_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS friendships (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            friend_id  INTEGER NOT NULL,
            UNIQUE(user_id, friend_id),
            FOREIGN KEY(user_id)   REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(friend_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id  INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            body       TEXT NOT NULL,
            sent_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(sender_id)   REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(receiver_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()
algo = None
trainset = None

# Configurare Flask-Limiter
limiter = Limiter(
    get_remote_address,  # Detectează adresa IP
    app=app,
    default_limits=["10 per minute"],  # Limită globală implicită
    storage_uri="memory://"  # Depozitare temporară în memorie
)

def preprocess_clicks():
    """
    Preprocess the clicks dataset by adding weights based on recency and interaction frequency.
    """
    # Conectează-te la baza de date și încarcă datele din tabela clicks
    conn = sqlite3.connect('tmdb_movies.db')
    clicks_df = pd.read_sql_query('SELECT user_id, movie_id, timestamp FROM clicks', conn)
    conn.close()

    # Asigură-te că există rânduri în dataset
    if clicks_df.empty:
        print("No click data found in the database.")
        return pd.DataFrame()  # Returnăm un DataFrame gol

    # Convert timestamp to datetime
    try:
        clicks_df['timestamp'] = pd.to_datetime(clicks_df['timestamp'], errors='coerce')
    except Exception as e:
        print(f"Error converting 'timestamp' to datetime: {e}")
        return pd.DataFrame()  # Returnăm un DataFrame gol dacă conversia eșuează

    # Elimină rândurile cu valori NaT (Not a Time) în coloana 'timestamp'
    clicks_df = clicks_df.dropna(subset=['timestamp'])

    # Adaugă greutăți bazate pe recență (ultimele 30 de zile)
    now = dt.datetime.now()
    clicks_df['weight'] = clicks_df['timestamp'].apply(
        lambda x: max(0.1, 1 - (now - x).days / 30) if pd.notnull(x) else 0.1
    )

    # Adaugă frecvența interacțiunilor
    clicks_df['interaction'] = clicks_df.groupby(['user_id', 'movie_id'])['timestamp'].transform('count')

    # Asigură-te că ambele coloane sunt numerice
    clicks_df['weight'] = clicks_df['weight'].astype(float)
    clicks_df['interaction'] = clicks_df['interaction'].astype(int)

    # Calculează greutatea finală ca produs între greutatea temporală și frecvența interacțiunilor
    clicks_df['final_weight'] = clicks_df['weight'] * clicks_df['interaction']

    # Filtrează interacțiunile mai vechi de 30 de zile
    recent_cutoff = now - dt.timedelta(days=30)
    clicks_df = clicks_df[clicks_df['timestamp'] >= recent_cutoff]

    return clicks_df[['user_id', 'movie_id', 'final_weight']]


def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='ro', dest='en')
    return translation.text

def record_click(user_id, movie_id):
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO clicks (user_id, movie_id, timestamp)
        VALUES (?, ?, ?)
    ''', (user_id, movie_id, datetime.now()))
    conn.commit()
    conn.close()
    initialize_collaborative_filtering_clicks()


# Initialize Click-Based Collaborative Filtering
def initialize_collaborative_filtering_clicks():
    global algo_clicks, trainset_clicks

    # Preprocess clicks
    clicks_df = preprocess_clicks()

    if clicks_df.empty:
        print("No recent clicks available for training.")
        algo_clicks, trainset_clicks = None, None
        return

    # Load dataset with Surprise
    reader = Reader(rating_scale=(0, 1))  # Weights are between 0 and 1
    data = Dataset.load_from_df(clicks_df[['user_id', 'movie_id', 'final_weight']], reader)
    trainset_clicks = data.build_full_trainset()

    # Train the model
    algo_clicks = SVD()
    algo_clicks.fit(trainset_clicks)
    print("Enhanced click-based model initialized.")



# Get Recommendations Based on Enhanced Clicks
def get_collaborative_click_recommendations(user_id, top_n=10):
    """
    Generate recommendations based on enhanced click data.
    """
    if not trainset_clicks or not algo_clicks:
        print("Click-based collaborative filtering model is not initialized.")
        return []

    # Build testset for all items not yet interacted with by the user
    testset = trainset_clicks.build_anti_testset()
    user_testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo_clicks.test(user_testset)

    # Sort recommendations by estimated interaction value
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)
    return [rec.iid for rec in recommendations[:top_n]]


def get_hybrid_recommendations_clicks(user_id, movie_id, top_n=10):
    content_recs = get_content_based_recommendations(movie_id, top_n)
    collaborative_recs = get_collaborative_click_recommendations(user_id, top_n)
    hybrid_recs = list(set(content_recs + collaborative_recs))[:top_n]

    # Fetch full details for each movie
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in hybrid_recs)
    cursor.execute(f'''
        SELECT id, title, poster_path
        FROM movies
        WHERE id IN ({placeholders})
    ''', hybrid_recs)
    movies = cursor.fetchall()
    conn.close()
    return movies


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
        print("No ratings available to train the model.")
        trainset, algo = None, None  # Ensure they are None if initialization fails
        return
    else:
        print(f"Ratings data loaded: {ratings_df.shape[0]} records (rating 1-5)")

    # Create a Surprise dataset with all ratings
    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        print("Dataset successfully created and trainset built with all ratings.")
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
        filtered_predictions = predictions  # No filtering based on rating threshold
        
        # Debugging: Print out filtered predictions with their estimated ratings
        print(f"All predictions for user {user_id}:")
        #for prediction in predictions:
            #print(f"Movie ID: {prediction.iid}, Estimated Rating: {prediction.est}")
        
        # Sort predictions by estimated rating in descending order
        filtered_predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top_n recommendations
        recommendations = [prediction.iid for prediction in filtered_predictions[:top_n]]
        # print(f"Final Recommendations for user {user_id}: {recommendations}")
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

tmdb_actor_cache = {}
def get_movie_actors(movie_id):
    """
    Obține lista de actori pentru filmul specificat folosind TMDb.
    Se folosește caching pentru a nu face requesturi repetate.
    Returnează un șir de caractere cu numele primilor 5 actori, separate prin virgulă.
    În caz de eroare, returnează un mesaj corespunzător.
    """
    # Verifică dacă rezultatul este deja în cache
    if movie_id in tmdb_actor_cache:
        return tmdb_actor_cache[movie_id]

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            print(f"Error fetching actors for movie ID {movie_id}: {response.status_code}")
            return "Actors not available"
        data = response.json()
        cast = data.get("cast", [])
        # Extrage numele actorilor din departamentul 'Acting'
        actors = [member.get("name", "") for member in cast if member.get("known_for_department", "").lower() == "acting"]
        # Limităm la primii 5 actori (dacă sunt disponibili)
        actor_str = ", ".join(actors[:5]) if actors else "Actors not available"
        # Stochează rezultatul în cache
        tmdb_actor_cache[movie_id] = actor_str
        return actor_str
    except Exception as e:
        print(f"Exception while fetching actors for movie ID {movie_id}: {e}")
        return "Actors not available"


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
    connn = sqlite3.connect('tmdb_movies_for_nn.db')
    cursorn = connn.cursor()
    
    
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
            cursorn.execute('''
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
    connn.commit()
    connn.close()
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


def fetch_popular_movies_by_genre(genre_id, limit=5, existing_movie_ids=set()):
    """
    Fetch popular movies for a specific genre from TMDb, excluding movies already in existing_movie_ids.

    Args:
    - genre_id (int): The ID of the genre on TMDb.
    - limit (int): Number of top movies to return.
    - existing_movie_ids (set): Set of movie IDs to exclude.

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

    # Filter and add movies that are not in existing_movie_ids
    popular_movies = []
    for movie in movies:
        movie_id = movie['id']
        if movie_id not in existing_movie_ids:
            popular_movies.append((movie_id, movie['title'], movie['poster_path']))
            existing_movie_ids.add(movie_id)  # Add movie_id to the set to avoid duplicates
        if len(popular_movies) >= limit:
            break

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
        conn_for_nn = sqlite3.connect('tmdb_movies_for_nn.db')
        cursor_for_nn = conn_for_nn.cursor()
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
        
        cursor_for_nn.execute('''
            INSERT INTO users (name, email, password, age, occupation, sex, address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, email, hashed_password, age, occupation, sex, address))
        

        # Store selected movies for the user
        for movie_id in selected_movies:
            cursor.execute('''
                INSERT INTO user_selected_movies (user_id, movie_id)
                VALUES (?, ?)
            ''', (user_id, movie_id))

        conn.commit()        
        conn_for_nn.commit()
        conn.close()
        conn_for_nn.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    # Fetch popular movies by genre from TMDb without duplicates
    genre_movies = {}
    existing_movie_ids = set()
    for genre_name, genre_id in relevant_genres.items():
        genre_movies[genre_name] = fetch_popular_movies_by_genre(genre_id, limit=5, existing_movie_ids=existing_movie_ids)

    return render_template('register.html', genre_movies=genre_movies)


# right next to your existing @app.route('/register')
@app.route('/api/genre_movies')
def api_genre_movies():
    # exactly the same code you already have in your register() GET
    genre_movies = {}
    existing_movie_ids = set()
    for genre_name, genre_id in relevant_genres.items():
        genre_movies[genre_name] = fetch_popular_movies_by_genre(
            genre_id,
            limit=5,
            existing_movie_ids=existing_movie_ids
        )
    return jsonify(genre_movies)


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    
    #session.clear()#important !!!

    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password']

        conn = sqlite3.connect('tmdb_movies.db')
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, name, password, is_2fa_enabled FROM users WHERE email = ?',
            (email,)
        )
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            # stocăm email-ul temporar pentru verify_2fa
            session['temp_user_email'] = email
            session['temp_user_id']    = user[0]
            session['temp_user_name']  = user[1]

            if user[3] == 0:
                flash('Trebuie să îți activezi 2FA înainte de a continua.', 'warning')
                return redirect(url_for('enable_2fa'))

            flash('Te rugăm să introduci codul OTP din aplicația ta de autentificare.', 'info')
            return redirect(url_for('verify_2fa'))
        else:
            flash('Email sau parolă invalidă.', 'danger')

    return render_template('login.html')


@app.route('/verify_2fa', methods=['GET', 'POST'])
def verify_2fa():
    if request.method == 'POST':
        otp_code     = request.form.get('otp_code')
        temp_user_id = session.get('temp_user_id')
        temp_email   = session.get('temp_user_email')

        if not temp_user_id:
            flash('Sesiune 2FA invalidă. Te rugăm să te loghezi din nou.', 'warning')
            return redirect(url_for('login'))

        conn = sqlite3.connect('tmdb_movies.db')
        cursor = conn.cursor()
        cursor.execute('SELECT twofa_secret FROM users WHERE id = ?', (temp_user_id,))
        row = cursor.fetchone()
        conn.close()

        if not row or not row[0]:
            flash('Nu ai 2FA activat. Te rugăm să te loghezi din nou.', 'warning')
            return redirect(url_for('login'))

        totp = pyotp.TOTP(row[0])
        if totp.verify(otp_code):
            # autentificare completă
            session['user_id']   = session.pop('temp_user_id')
            session['user_name'] = session.pop('temp_user_name')
            session.pop('temp_user_email', None)

            # e admin?
            if temp_email.lower() == 'admin@admin':
                return redirect(url_for('admin_dashboard'))

            flash('Autentificare 2FA reușită!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Codul OTP introdus este invalid. Încearcă din nou.', 'danger')

    return render_template('verify_2fa.html')

@app.route('/enable_2fa', methods=['GET', 'POST'])
def enable_2fa():
    temp_user_id = session.get('temp_user_id')
    if not temp_user_id:
        flash('Trebuie să te loghezi mai întâi.', 'warning')
        return redirect(url_for('login'))

    # 1) Make sure there is a twofa_secret
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute('SELECT twofa_secret FROM users WHERE id = ?', (temp_user_id,))
    row = cursor.fetchone()
    if row and row[0]:
        twofa_secret = row[0]
    else:
        twofa_secret = generate_2fa_secret()
        cursor.execute(
            'UPDATE users SET twofa_secret = ? WHERE id = ?',
            (twofa_secret, temp_user_id)
        )
        conn.commit()
    conn.close()

    # 2) Always (re)build the QR code so we can show it on GET *and* on any invalid POST
    email = session.get('temp_user_name', 'user@example.com')
    totp_uri = generate_qr_code_uri(email, twofa_secret)
    qr_code_base64 = create_qr_code_base64(totp_uri)

    # 3) On POST, verify the OTP
    if request.method == 'POST':
        otp_code = request.form.get('otp_code', '').strip()
        totp     = pyotp.TOTP(twofa_secret)

        if totp.verify(otp_code):
            # mark 2FA as enabled
            conn = sqlite3.connect('tmdb_movies.db')
            cur  = conn.cursor()
            cur.execute(
                'UPDATE users SET is_2fa_enabled = 1 WHERE id = ?',
                (temp_user_id,)
            )
            conn.commit()
            conn.close()

            # promote temp → full login
            session['user_id']   = temp_user_id
            session['user_name'] = session.pop('temp_user_name')
            session.pop('temp_user_id'  , None)
            session.pop('temp_user_email', None)

            flash('2FA activat cu succes! Ești acum logat.', 'success')
            return redirect(url_for('index'))

        flash('Cod OTP invalid. Încearcă din nou.', 'danger')

    # 4) Render the QR page on GET or after a failed OTP
    return render_template('enable_2fa.html',
                           qr_code_base64=qr_code_base64)



@app.route('/network', endpoint='user_recommendation')
@login_required
def user_network():
    # eventual pregăteşti date suplimentare
    return render_template('user_recommendation.html')

# helpers ─────────────────────────────────────────────────────────
def are_friends(a, b, conn):
    q = "SELECT 1 FROM friendships WHERE (user_id=? AND friend_id=?) OR (user_id=? AND friend_id=?);"
    return conn.execute(q, (a, b, b, a)).fetchone() is not None
# ========= API =========
# ========= API =========

def euclidean(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    return np.sqrt(np.sum((v1 - v2) ** 2))

@app.route("/api/friends/recommended")
@login_required
def api_recommended_users():
    uid = session["user_id"]

    # 1) Încarcă toate rating-urile
    conn = sqlite3.connect('tmdb_movies.db')
    ratings = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()

    # 2) Transformă în matrice user×movie
    pivot = ratings.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating',
        fill_value=0
    )

    # dacă eu nu am notat nimic
    if uid not in pivot.index:
        return jsonify([])

    target_vec = pivot.loc[uid].values.tolist()

    # 3) Preia lista de prieteni ACCEPTAȚI (să îi excludem)
    conn = sqlite3.connect('tmdb_movies.db')
    cur  = conn.cursor()
    cur.execute("SELECT friend_id FROM friendships WHERE user_id = ?", (uid,))
    friends = {row[0] for row in cur.fetchall()}
    conn.close()

    # 4) Calculează distanța față de toți ceilalți (skip self + prieteni)
    distances = []
    for other_uid, row in pivot.iterrows():
        if other_uid == uid or other_uid in friends:
            continue
        d = euclidean(target_vec, row.values.tolist())
        # debug
        print(f"[DEBUG] Distance between user {uid} and user {other_uid}: {d:.4f}")
        distances.append((other_uid, d))

    # 5) Ia top-10 cei mai Mici (deci cei mai asemănători)
    distances.sort(key=lambda x: x[1])
    top10 = distances[:10]

    # 6) Încarcă datele lor de user și returnează JSON
    conn = sqlite3.connect('tmdb_movies.db')
    users = []
    for other_uid, dist in top10:
        r = conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?",
            (other_uid,)
        ).fetchone()
        if r:
            users.append({
                'id':       r[0],
                'name':     r[1],
                'email':    r[2],
                'distance': round(dist, 4)
            })
    conn.close()

    return jsonify(users)


@app.route("/api/messages/inbox")
@login_required
def api_inbox():
    """
    Returnează toate mesajele primite de userul curent, în ordinea
    descrescătoare a datei.
    """
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT 
          m.id,
          m.sender_id,
          u.name   AS sender_name,
          u.email  AS sender_email,
          m.body,
          m.sent_at
        FROM messages m
        JOIN users u ON u.id = m.sender_id
        WHERE m.receiver_id = ?
        ORDER BY m.sent_at DESC
    """, (uid,)).fetchall()
    conn.close()
    # Return as list of dicts
    return jsonify([dict(r) for r in rows])

@app.route("/api/messages/<int:msg_id>/delete", methods=["POST"])
@login_required
def api_delete_message(msg_id):
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')
    cur = conn.cursor()
    # verify that the current user is the receiver
    cur.execute("SELECT 1 FROM messages WHERE id=? AND receiver_id=?", (msg_id, uid))
    if not cur.fetchone():
        conn.close()
        abort(403)
    # delete the message
    cur.execute("DELETE FROM messages WHERE id=?", (msg_id,))
    conn.commit()
    conn.close()
    return jsonify(ok=True)

@app.route("/api/friends/<int:friend_id>/delete", methods=["POST"])
@login_required
def api_delete_friend(friend_id):
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')
    cur  = conn.cursor()
    # Delete both directions
    cur.execute("DELETE FROM friendships WHERE user_id=? AND friend_id=?", (uid, friend_id))
    cur.execute("DELETE FROM friendships WHERE user_id=? AND friend_id=?", (friend_id, uid))
    conn.commit()
    conn.close()
    return jsonify(ok=True)

@app.route("/api/friends/requests", methods=["GET", "POST"])
@login_required
def api_friend_requests():
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')

    if request.method == "POST":
        data = request.get_json() or {}
        target = int(data.get("target_id", 0))
        if not target or target == uid:
            return jsonify(ok=False, msg="id invalid"), 400
        try:
            conn.execute("INSERT INTO friend_requests(sender_id, receiver_id) VALUES (?,?)", (uid, target))
            conn.commit()
            return jsonify(ok=True)
        except sqlite3.IntegrityError:
            return jsonify(ok=False, msg="deja există"), 400

    # GET → toate request‑urile unde eu sunt receiver
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
        fr.id,
        u.id      AS sender_id,
        u.name    AS from_name,
        u.email   AS from_email,
        fr.created_at
        FROM friend_requests fr
        JOIN users u
        ON u.id = fr.sender_id
        WHERE fr.receiver_id=? AND fr.status='pending'
    """, (uid,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/friends/requests/<int:req_id>/answer", methods=["POST"])
@login_required
def api_answer_request(req_id):
    action = (request.get_json() or {}).get("action")
    if action not in ("accept", "deny"):
        return jsonify(ok=False), 400
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')
    row = conn.execute("SELECT sender_id FROM friend_requests WHERE id=? AND receiver_id=? AND status='pending'",
                       (req_id, uid)).fetchone()
    if not row:
        conn.close()
        return jsonify(ok=False), 404

    new_status = "accepted" if action == "accept" else "denied"
    conn.execute("UPDATE friend_requests SET status=? WHERE id=?", (new_status, req_id))
    if action == "accept":
        sender = row[0]
        # adăugăm în friendships (ambele sensuri)
        for a, b in [(uid, sender), (sender, uid)]:
            try:
                conn.execute("INSERT OR IGNORE INTO friendships(user_id, friend_id) VALUES (?,?)", (a, b))
            except sqlite3.IntegrityError:
                pass
    conn.commit()
    conn.close()
    return jsonify(ok=True)


@app.route("/api/friends/list")
@login_required
def api_friend_list():
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT u.id, u.name, u.email
        FROM friendships f
        JOIN users u ON u.id = f.friend_id
        WHERE f.user_id=?
    """, (uid,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/friends/<int:friend_id>/movies")
@login_required
def api_friend_movies(friend_id):
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')
    if not are_friends(uid, friend_id, conn):
        conn.close()
        abort(403)

    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT m.id, m.title, m.poster_path, r.rating
        FROM ratings r
        JOIN movies m ON m.id = r.movie_id
        WHERE r.user_id = ?
        ORDER BY r.rating DESC
        LIMIT 30
    """, (friend_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/messages/send", methods=["POST"])
@login_required
def api_send_msg():
    data   = request.get_json() or {}
    to_id  = int(data.get("to"))
    body   = (data.get("body") or "").strip()
    if not body:
        return jsonify(ok=False), 400
    uid = session["user_id"]
    conn = sqlite3.connect('tmdb_movies.db')
    if not are_friends(uid, to_id, conn):
        conn.close()
        abort(403)

    conn.execute("INSERT INTO messages(sender_id, receiver_id, body) VALUES (?,?,?)", (uid, to_id, body))
    conn.commit()
    conn.close()
    return jsonify(ok=True)


@app.route('/admin')
@login_required
def admin_dashboard():
    if session.get('user_name', '').lower() != 'admin':
        abort(403)

    # Citire useri
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM users")
    rows = cursor.fetchall()
    conn.close()

    # Calcul RMSE
    rmse_list = []
    for user_id, user_name in rows:
        algo_rmse = compute_rmse_for_algorithms(user_id)
        rmse_value = algo_rmse.get('content_cb', float('nan'))
        
        # Verificăm dacă RMSE-ul este valid (nu NaN)
        if not math.isnan(rmse_value):
            rmse_list.append({
                'user_id': user_id,
                'user_name': user_name,
                'rmse': rmse_value
            })

    # Sortare DESCENDING după RMSE și selectare top 10
    rmse_list_sorted = sorted(
        rmse_list, 
        key=lambda x: x['rmse'], 
        reverse=True  # Ordine descrescătoare
    )
    top10_rmse = rmse_list_sorted[:10]  # Primele 10 după sortare
    
     # === Citiți utilizatorii și rating‑urile ===
    conn = sqlite3.connect('tmdb_movies.db')
    ratings = pd.read_sql_query("""
    SELECT r.user_id, r.movie_id, r.rating
    FROM ratings r
    JOIN users u ON r.user_id = u.id
    """, conn)

    users   = pd.read_sql_query("SELECT id, name FROM users", conn)
    movies  = pd.read_sql_query("SELECT id, title FROM movies", conn)

    #conn.close()

    # === Matrice user‑movie & reducere la 2D (PCA) ===
    pivot = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
    pca   = PCA(n_components=2, random_state=42)
    X2D   = pca.fit_transform(pivot.values)

    # aliniază index‑ul cu username‑urile
    id_to_name = dict(users.values)
    names      = [id_to_name[i] for i in pivot.index]

    # === K‑Means cu 3 clustere ===
    km   = KMeans(n_clusters=3, init='random', max_iter=10, n_init=10, random_state=42)
    y    = km.fit_predict(X2D)
    ctrs = km.cluster_centers_

    # === Structuri pentru template ===
    clusters_points = [ [] for _ in range(3) ]
    for (x, y_), c in zip(X2D, y):
        clusters_points[c].append({'x': float(x), 'y': float(y_)})

    centroid_points = [{'x': float(x), 'y': float(y)} for x, y in ctrs]

    #‑‑‑ 3) Cele mai bine / prost evaluate filme
    movie_stats = (ratings.groupby('movie_id')['rating'].mean()
                             .reset_index()
                             .merge(movies, left_on='movie_id', right_on='id'))

    best_movies = (movie_stats.sort_values('rating', ascending=False)
                              .head(10)[['title', 'rating']]
                              .rename(columns={'rating': 'avg'})
                              .to_dict(orient='records'))

    flop_movies = (movie_stats.sort_values('rating')
                              .head(10)[['title', 'rating']]
                              .rename(columns={'rating': 'avg'})
                              .to_dict(orient='records'))

    conn.close()
    print("BEST MOVIE", best_movies)
    print("Bad MOVIE", flop_movies)
    return render_template(
        'admin.html',
        clusters_points = clusters_points,
        centroid_points = centroid_points,
        best_movies     = best_movies,
        flop_movies     = flop_movies,
        top10_rmse=top10_rmse
    )
from email.message import EmailMessage
import smtplib, ssl, mimetypes

SMTP_HOST = 'smtp.mail.yahoo.com'      # sau host-ul tău real
SMTP_PORT = 587
SMTP_USER = 'banuion99@yahoo.com'         # contul configurat în Thunderbird
SMTP_PASS = 'yiwoaufazsjwmuno'

def get_latest_movies(n=5):
    conn = sqlite3.connect('tmdb_movies.db')
    df = pd.read_sql_query(
        """
        SELECT title, release_date, poster_path          -- avem și poster_path
        FROM movies
        ORDER BY date(release_date) DESC
        LIMIT ?
        """,
        conn, params=(n,)
    )
    conn.close()
    return df.to_dict(orient='records')

def build_email(to_list, movies):
    """
    movies = list(dict(title, release_date, poster_path))
    """
    msg = EmailMessage()
    msg['Subject'] = '🎬 Cele mai noi 5 filme din aplicația noastră'
    msg['From']    = SMTP_USER
    msg['To']      = ', '.join(to_list)

    # Text simplu
    text_lines = ["Cele mai noi 5 titluri din baza noastră:\n"]
    for m in movies:
        text_lines.append(f"- {m['title']} ({m['release_date']})")
    msg.set_content("\n".join(text_lines))

    # HTML cu imagini din URL
    html_lines = [
        '<h2>Top 5 filme noi</h2>',
        '<table style="border-collapse:collapse;">'
    ]
    for m in movies:
        html_lines.append(f"""
          <tr style="vertical-align:top;">
            <td style="padding:6px;">
              <img src="{m['poster_path']}" alt="{m['title']}"
                   style="width:120px;border-radius:6px;">
            </td>
            <td style="padding:6px 12px;">
              <strong>{m['title']}</strong><br>
              <em>{m['release_date']}</em>
            </td>
          </tr>
        """)

    html_lines.append('</table>')
    msg.add_alternative("".join(html_lines), subtype='html')
    return msg


@app.route('/send_posters', methods=['POST'])
@login_required
def send_posters():
    if session.get('user_name', '').lower() != 'admin':
        return jsonify({'message': 'Forbidden'}), 403

    data = request.get_json(silent=True) or {}
    single_email = data.get('email')
    send_all     = data.get('send_all')

    # Obținem lista de destinatari
    recipients = []
    if send_all:
        conn = sqlite3.connect('tmdb_movies.db')
        cur  = conn.cursor()
        cur.execute("""
            SELECT DISTINCT email
            FROM users
            WHERE email IS NOT NULL AND email != ''
        """)
        recipients = [row[0] for row in cur.fetchall()]
        conn.close()
    elif single_email:
        recipients = [single_email]

    if not recipients:
        return jsonify({'message': 'Nu ai selectat niciun destinatar!'}), 400

    # Posterele celor mai noi 5 filme
    movies = get_latest_movies(5)
    print('send_all=', send_all, 'single_email=', single_email)
    print('recipients=', recipients)

        # Construim și trimitem email-ul

    context = ssl.create_default_context()
    # deschidem o singură conexiune
    with smtplib.SMTP_SSL(SMTP_HOST, 465, context=context) as server:
        server.login(SMTP_USER, SMTP_PASS)

        # trimitem un email individual fiecărui user
        sent = 0
        for r in recipients:
            try:
                single_msg = build_email([r], movies)
                server.send_message(single_msg)
                sent += 1
            except Exception:
                # loghează în consolă dacă un singur email dă eroare, dar continuă
                print(f"Failed to send to {r}")

    return jsonify({'message': f'Trimis cu succes către {sent} destinatar(i).'})

@app.route('/admin/users', methods=['GET'])
@login_required
def list_users():
    if session.get('user_name', '').lower() != 'admin':
        abort(403)
    conn = sqlite3.connect('tmdb_movies.db')
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, name, email
        FROM users
        WHERE email IS NOT NULL AND email != ''
    """)
    rows = cur.fetchall()
    conn.close()
    users = [{'id':r[0], 'name':r[1], 'email':r[2]} for r in rows]
    return jsonify(users)

@app.route('/admin/users/<int:user_id>', methods=['GET'])
@login_required
def user_detail(user_id):
    if session.get('user_name', '').lower() != 'admin':
        abort(403)

    conn = sqlite3.connect('tmdb_movies.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1) Preluăm detaliile user-ului
    cur.execute("""
        SELECT id, name, email, age, occupation, sex, address, is_2fa_enabled
        FROM users
        WHERE id = ?
    """, (user_id,))
    user_row = cur.fetchone()
    if not user_row:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    user = {key: user_row[key] for key in user_row.keys()}

    # 2) Preluăm rating-urile date de acest user
    cur.execute("""
        SELECT r.movie_id,
               m.title       AS movie_title,
               r.rating      AS user_rating,
               r.id          AS rating_id
        FROM ratings r
        JOIN movies m ON m.id = r.movie_id
        WHERE r.user_id = ?
    """, (user_id,))
    ratings = [
        {
            'rating_id':   row['rating_id'],
            'movie_id':    row['movie_id'],
            'movie_title': row['movie_title'],
            'rating':      row['user_rating']
        }
        for row in cur.fetchall()
    ]

    conn.close()

    # 3) Returnăm împreună
    user['ratings'] = ratings
    return jsonify(user)

@app.route('/admin/users/delete', methods=['POST'])
@login_required
def delete_user():
    if session.get('user_name', '').lower() != 'admin':
        abort(403)
    data = request.get_json() or {}
    uid  = data.get('user_id')
    if not uid:
        return jsonify({'message':'Missing user_id'}), 400
    conn = sqlite3.connect('tmdb_movies.db')
    cur  = conn.cursor()
    cur.execute("DELETE FROM users WHERE id = ?", (uid,))
    conn.commit()
    conn.close()
    return jsonify({'message':f'User {uid} șters cu succes.'})



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



def initialize_user_based_matrices():
    """
    Inițializează r_matrix (matrice user vs. movie) pentru user-based Weighted Mean
    și calculează similaritatea (cosine_sim) dintre utilizatori.
    """
    global r_matrix, cosine_sim  # Folosim variabilele globale

    # 1) Încarcă ratingurile din BD
    conn = sqlite3.connect('tmdb_movies.db')
    ratings_df = pd.read_sql_query('SELECT user_id, movie_id, rating FROM ratings', conn)
    conn.close()

    if ratings_df.empty:
        print("Nu există rating-uri în baza de date.")
        r_matrix, cosine_sim = None, None
        return

    # 2) Construim pivot: rânduri = user_id, coloane = movie_id, valori = rating
    temp_matrix = ratings_df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating'
    )

    # 3) Facem o copie cu NaN => 0 doar pentru calcul similaritate
    # (dar păstrăm r_matrix cu NaN pentru calculele Weighted Mean)
    temp_matrix_filled = temp_matrix.fillna(0)

    # 4) Calculăm similaritatea cosinus între useri
    sim = cosine_similarity(temp_matrix_filled, temp_matrix_filled)
    
    # 5) Convertim la DataFrame, punând index și coloane = user_id
    temp_user_ids = temp_matrix.index
    user_sim_df = pd.DataFrame(sim, index=temp_user_ids, columns=temp_user_ids)

    # 6) Setăm variabilele globale
    r_matrix = temp_matrix
    cosine_sim = user_sim_df

    print("User-based Weighted Mean: r_matrix și cosine_sim create cu succes.")
 #Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
#Define the baseline model to always return 3.
def baseline(user_id, movie_id):
    return 3.0


# Variabilă globală
users_df = None

def load_users():
    global users_df
    conn = sqlite3.connect('tmdb_movies.db')
    # Citim toți userii (coloane: id, sex, occupation, etc.)
    temp_df = pd.read_sql_query("SELECT id, sex, occupation FROM users", conn)
    conn.close()
    # Setăm index = id
    temp_df = temp_df.rename(columns={"id": "user_id"})
    temp_df = temp_df.set_index("user_id")

    users_df = temp_df
    print("Users loaded in 'users_df' with shape:", users_df.shape)


def user_based_weighted_mean_recommendation(user_id, top_n=10):
    """
    Generează top_n recomandări pentru user_id
    pe baza metodei Weighted Mean a rating-urilor (user-based CF),
    într-o singură funcție integrată.
    
    Returnează o listă de movie_id (int).
    """

    

    # Verificăm dacă datele sunt inițializate
    if r_matrix is None or cosine_sim is None:
        print("Matricea de rating (r_matrix) sau matricea de similaritate (cosine_sim) este None.")
        return []

    # Verificăm dacă user_id există în r_matrix și în coloanele matricii de similaritate
    if user_id not in r_matrix.index or user_id not in cosine_sim.columns:
        return []

    # Obținem scorurile de similitudine față de toți ceilalți useri
    user_sim_scores = cosine_sim[user_id]

    # Pentru a evita recomandarea unor filme deja votate de user, le excludem
    user_rated_movies = []
    if user_id in r_matrix.index:
        user_rated_movies = r_matrix.loc[user_id].dropna().index.tolist()

    # Construim lista de candidați – filme încă nevotate de user
    candidate_movies = [m for m in r_matrix.columns if m not in user_rated_movies]

    predicted_ratings = []
    for movie_id in candidate_movies:
        # 1) Verificăm dacă filmul există în coloanele matricei
        if movie_id not in r_matrix.columns:
            # N-avem date, fallback la 3.0
            predicted_ratings.append((movie_id, 3.0))
            continue

        # 2) Luăm rating-urile existente (fără NaN)
        m_ratings = r_matrix[movie_id].dropna()  # Serie Pandas cu (user -> rating)

        # 3) Intersecția userilor care au notat acest film cu cei din user_sim_scores
        common_users = m_ratings.index.intersection(user_sim_scores.index)

        if common_users.empty:
            # Nimeni nu a notat filmul => fallback 3.0
            predicted_ratings.append((movie_id, 3.0))
            continue

        # 4) Selectăm doar similaritățile și rating-urile pentru userii care au notat filmul
        relevant_sims = user_sim_scores.loc[common_users]
        relevant_ratings = m_ratings.loc[common_users]

        # 5) Calculăm media ponderată
        denom = relevant_sims.sum()
        if denom == 0:
            # Dacă sumele de similitudine sunt 0, fallback 3.0
            predicted_ratings.append((movie_id, 3.0))
        else:
            wmean_rating = np.dot(relevant_sims, relevant_ratings) / denom
            predicted_ratings.append((movie_id, wmean_rating))

    # Sortăm descrescător după scorul prezis
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Extragem doar movie_id din top_n
    top_recommended_ids = [x[0] for x in predicted_ratings[:top_n]]
    return top_recommended_ids

# 1. Weighted Mean filtrat după GEN
def cf_user_wmean_gender(user_id, movie_id):
    # fallback dacă user_id lipsește
    if user_id not in users_df.index:
        return 3.0               # fallback, sau np.nan, cum preferi
    
    if user_id not in r_matrix.index:
        return 3.0

    if movie_id not in r_matrix.columns:
        return 3.0
    #global r_matrix, cosine_sim, users_df
    """
    Calculează ratingul prezis pentru (user_id, movie_id)
    folosind Weighted Mean doar cu useri care au același sex.
    """
    # Dacă filmul nu este în coloanele r_matrix, nu avem date => fallback
    if movie_id not in r_matrix.columns:
        return 3.0

    # Sex user curent
    target_gender = users_df.loc[user_id, 'sex']

    # Similarități cu ceilalți useri
    sim_scores = cosine_sim.loc[user_id].copy()  # Series

    # Eliminăm userii cu alt sex
    different_gender_users = users_df[users_df['sex'] != target_gender].index
    sim_scores.drop(different_gender_users, errors='ignore', inplace=True)

    # Rating-urile existente pentru film
    m_ratings = r_matrix[movie_id].dropna()

    # Intersectăm cu userii care au notat filmul
    common_users = sim_scores.index.intersection(m_ratings.index)
    if common_users.empty:
        return 3.0

    # Filtrăm
    sim_scores = sim_scores.loc[common_users]
    relevant_ratings = m_ratings.loc[common_users]

    if sim_scores.sum() == 0:
        return 3.0

    wmean_rating = np.dot(sim_scores, relevant_ratings) / sim_scores.sum()
    return wmean_rating


# 2. Weighted Mean filtrat după GEN + OCCUPATION
def cf_user_wmean_gen_occ(user_id, movie_id):
   
    """
    Calculează ratingul prezis pentru (user_id, movie_id)
    folosind Weighted Mean doar cu useri care au același sex + aceeași ocupație.
    """
    if movie_id not in r_matrix.columns:
        return 3.0
    if user_id not in users_df.index:
        return 3.0
    # Preluăm userul din DataFrame-ul users
    target_user = users_df.loc[user_id]
    target_gender = target_user['sex']
    target_occ = target_user['occupation']

    # Similarități cu ceilalți
    sim_scores = cosine_sim.loc[user_id].copy()

    # Scoatem userii care nu coincid la ambele atribute
    different_users = users_df[
        (users_df['sex'] != target_gender) | (users_df['occupation'] != target_occ)
    ].index
    sim_scores.drop(different_users, errors='ignore', inplace=True)

    # Rating film
    m_ratings = r_matrix[movie_id].dropna()

    common_users = sim_scores.index.intersection(m_ratings.index)
    if common_users.empty:
        return 3.0

    sim_scores = sim_scores.loc[common_users]
    relevant_ratings = m_ratings.loc[common_users]

    if sim_scores.sum() == 0:
        return 3.0

    wmean_rating = np.dot(sim_scores, relevant_ratings) / sim_scores.sum()
    return wmean_rating

def fetch_and_save_movie_by_id(movie_id):
    """
    Verifică dacă 'movie_id' există în DB.
    Dacă nu, face request la TMDb și îl inserează.
    Returnează True dacă filmul a fost găsit (în DB sau la TMDb), False dacă nu.
    """
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()

    # 1) Verifică dacă deja există în movies
    cursor.execute("SELECT id FROM movies WHERE id = ?", (movie_id,))
    row = cursor.fetchone()
    if row:
        # Filmul există deja
        conn.close()
        return True

    # 2) Dacă nu există, facem request la TMDb
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Eroare: filmul cu ID={movie_id} nu a putut fi preluat (status={resp.status_code}).")
        conn.close()
        return False
    
    data = resp.json()
    # Extragem câmpuri relevante
    title = data.get("title", "")
    poster_path = data.get("poster_path", None)
    overview = data.get("overview", "")
    release_date = data.get("release_date", "")
    # Unele filme pot avea un array genres; extragem sub formă de text
    genres_list = data.get("genres", [])
    genres_str = ", ".join(g.get("name", "") for g in genres_list)
    popularity = data.get("popularity", 0.0)
    # opțional: căutăm trailer separat prin /movie/<id>/videos, dar ignorăm pentru scurt exemplu
    trailer_url = None

    # 3) Inserăm filmul în DB
    try:
        cursor.execute("""
            INSERT INTO movies (id, title, poster_path, overview, release_date, genres, popularity, trailer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            movie_id,
            title,
            poster_path,
            overview,
            release_date,
            genres_str,
            popularity,
            trailer_url
        ))
        conn.commit()
        conn.close()
        print(f"Filmul cu ID={movie_id} a fost inserat în baza de date.")
        return True
    except Exception as e:
        print(f"Eroare la inserarea filmului {movie_id}: {e}")
        conn.close()
        return False

def get_top_recommendations(cf_func, user_id, top_n=10):
    if user_id not in r_matrix.index:
        # userul nu apare în pivot -> 0 ratinguri
        return []
    """
    Generează top_n filme recomandate pt user_id,
    folosind funcția de CF primită (cf_func).
    """
    if r_matrix is None or cosine_sim is None:
        print("r_matrix / cosine_sim nu e inițializat.")
        return []

    # Filme candidate = toate coloanele (toate filmele) minus cele deja notate
    user_rated_movies = r_matrix.loc[user_id].dropna().index.tolist() if user_id in r_matrix.index else []
    candidate_movies = [m for m in r_matrix.columns if m not in user_rated_movies]

    predictions = []
    for mid in candidate_movies:
        score = cf_func(user_id, mid)
        predictions.append((mid, score))

    # Sortăm descrescător după rating prezis
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Top N
    top_ids = [x[0] for x in predictions[:top_n]]
    return top_ids
from tensorflow.keras.layers import Embedding, Dense, Dropout

class MovieRecommender(keras.Model):
    def __init__(self, num_users, num_movies, embedding_dim=16, l2_coef=1e-4):
        super().__init__()
        reg = regularizers.l2(l2_coef)

        self.user_emb   = Embedding(num_users,  embedding_dim,
                                    embeddings_regularizer=reg)
        self.movie_emb  = Embedding(num_movies, embedding_dim,
                                    embeddings_regularizer=reg)
        self.drop_emb   = Dropout(0.30)
        self.dense      = Dense(16, activation="relu",
                                kernel_regularizer=reg)
        self.drop_dense = Dropout(0.30)
        self.out        = Dense(1, activation="sigmoid",
                                kernel_regularizer=reg)

    def call(self, inputs):
        u_idx, m_idx = inputs                # întregi [batch]
        u = self.user_emb(u_idx)             # [batch, emb_dim]
        m = self.movie_emb(m_idx)            # [batch, emb_dim]
        x = tf.concat([u, m], axis=-1)       # [batch, 2*emb_dim]
        x = self.drop_emb(x)
        x = self.dense(x)                    # [batch, 16]
        x = self.drop_dense(x)
        return self.out(x)                   # [batch, 1] valori 0-1

def import_missing_movies_from_ratings(db_path="tmdb_movies.db"):
    """
    Pentru fiecare movie_id din tabela ratings, dacă acel ID nu e încă în tabela movies,
    îl aduce de la TMDb și îl inserează (folosind fetch_and_save_movie_by_id).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1) Ia toate ID-urile de film din ratings
    ratings_df = pd.read_sql("SELECT DISTINCT movie_id FROM ratings", conn)
    conn.close()

    # 2) Parcurge fiecare ID și apelează fetch_and_save_movie_by_id
    for mid in ratings_df["movie_id"].unique():
        fetch_and_save_movie_by_id(mid)
        


def train_and_save_model(db_path="tmdb_movies_for_nn.db",
                         save_path="nn_model_tf"):

    # 1) ratinguri din SQLite
    with sqlite3.connect(db_path) as conn:
        ratings = pd.read_sql(
            "SELECT user_id, movie_id, rating FROM ratings", conn)

    if ratings.shape[0] < 20:          # pur orientativ
        print("Prea puţine ratinguri pentru antrenare.")
        return

    # 2) mapări user/movie → index
    user_map  = {u: i for i, u in enumerate(ratings.user_id.unique())}
    movie_map = {m: i for i, m in enumerate(ratings.movie_id.unique())}

    ratings["u_idx"] = ratings.user_id.map(user_map)
    ratings["m_idx"] = ratings.movie_id.map(movie_map)
    ratings["r_norm"] = ratings.rating / 5.0   # 0-1

    # 3) split 75/25
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ratings, test_size=0.25, random_state=42)
    X_train = [train.u_idx.values,  train.m_idx.values]
    y_train =  train.r_norm.values
    X_val   = [test.u_idx.values,   test.m_idx.values]
    y_val   =  test.r_norm.values

    # 4) construim + antrenăm
    model = MovieRecommender(len(user_map), len(movie_map))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    es = EarlyStopping(monitor="val_loss", patience=3,
                       restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=30, batch_size=64,
              validation_data=(X_val, y_val),
              callbacks=[es], verbose=2)

    # 5) salvăm modelul + mapările
    model.save(save_path, save_format="tf")
    np.save("user_map.npy",  user_map)
    np.save("movie_map.npy", movie_map)
    print("Modelul a fost salvat fără a supra-potrivi.")
    # 7. Calculează RMSE-ul pe setul de validare (normalizat 0–1)
    y_pred_norm = model.predict(X_val).flatten()
    rmse_norm = np.sqrt(np.mean((y_pred_norm - y_val) ** 2))
    print(f"RMSE normalizat (0–1): {rmse_norm:.4f}")
    print(f"RMSE original (1–5): {rmse_norm*5:.4f}")

def load_nn_model(model_path="nn_model_tf"):
    # Încarcă modelul din SavedModel format (folderul specificat)
    model = keras.models.load_model(model_path, compile=False)
    # Încarcă mapările
    user_map = np.load("user_map.npy", allow_pickle=True).item()
    movie_map = np.load("movie_map.npy", allow_pickle=True).item()
    return model, user_map, movie_map


def recommend_movies_nn(user_id, model, user_map, movie_map, num_recommendations=5):
    """
    user_id: user_id real, așa cum apare în ratings
    model: rețeaua neuronală încărcată
    user_map, movie_map: dicționare pt a obține user_idx, respectiv movie_idx
    Returnează o listă de movie_id (din DB), ordonată descrescător după rating prezis
    """
    if user_id not in user_map:
        return []
    
    # 1) generăm toate movie_ids, mapate la index
    all_movie_ids = list(movie_map.keys())
    all_movie_idx = [movie_map[m] for m in all_movie_ids]
    
    # 2) repetăm user_idx pentru toate filmele
    uidx = user_map[user_id]
    # Convertește la int64 explicit:
    user_array = np.array([uidx]*len(all_movie_idx), dtype=np.int64)
    movie_array = np.array(all_movie_idx, dtype=np.int64)

    # 3) Prezicere
    predictions = model.predict([user_array, movie_array]).flatten()
    # scalez la 1..5
    ratings_1_5 = predictions * 4.0 + 1.0

    # 3) sortăm descrescător
    # returnăm ID-urile (nu index)
    sort_idx = np.argsort(ratings_1_5)[::-1]
    top_idx = sort_idx[:num_recommendations]
    top_movie_ids = [all_movie_ids[i] for i in top_idx]
    return top_movie_ids

nn_model = None
user_map = None
movie_map = None

def fetch_missing_movies(movie_ids):
    """
    Pentru fiecare ID din lista movie_ids, verifică/inserază filmul în DB,
    folosind fetch_and_save_movie_by_id.
    """
    for mid in movie_ids:
        fetch_and_save_movie_by_id(mid)


def fetch_tmdb_movie_data(movie_id, language="en-US"):
    """
    Face request la TMDb pentru movie_id, cu limba specificată.
    Returnează un tuple (id, title, poster_path) sau None dacă apare o eroare.
    NU inserează nimic în baza de date.
    """
    # Construim URL-ul de interogare
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language={language}"
    
    # Facem request
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[TMDb] Eroare la fetch pentru ID={movie_id}. Cod HTTP={resp.status_code}")
        return None

    data = resp.json()
    
    # TMDb poate întoarce un obiect cu "status_code" != 200 (ex. 34 = not found)
    if 'status_code' in data and data['status_code'] != 200:
        print(f"[TMDb] Eroare la fetch pentru ID={movie_id}: {data['status_message']}")
        return None

    # Extragem câmpurile esențiale
    title = data.get("title")
    poster_path = data.get("poster_path")

    if not title:
        # Poate fi serial sau film inexistent
        print(f"[TMDb] ID={movie_id} nu are titlu (posibil serial sau invalid).")
        return None

    # Returnăm sub formă de tuple, dar poți returna un dict dacă preferi.
    return (movie_id, title, poster_path)


# Variabile globale pentru modelul ALS implicit
als_model_implicit = None
user_to_idx = {}
movie_to_idx = {}
rating_matrix = None

def train_als_model_implicit():
    global als_model_implicit, user_to_idx, movie_to_idx, rating_matrix

    # Conectează-te la baza de date și citește ratingurile
    conn = sqlite3.connect('tmdb_movies.db')
    ratings_df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()

    if ratings_df.empty:
        print("Nu există suficiente date pentru antrenarea modelului ALS implicit.")
        return None

    # Mapăm user_id și movie_id la indici (0-indexați)
    unique_users = ratings_df['user_id'].unique()
    unique_movies = ratings_df['movie_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}

    ratings_df['user_index'] = ratings_df['user_id'].map(user_to_idx)
    ratings_df['movie_index'] = ratings_df['movie_id'].map(movie_to_idx)

    # Construim matricea sparse de ratinguri
    data = ratings_df['rating'].astype(float)
    rows = ratings_df['user_index']
    cols = ratings_df['movie_index']
    rating_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

    # Antrenăm modelul ALS cu implicit
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
    # implicit se așteaptă ca datele de intrare să fie de tip "confidence" – ratingurile pot fi folosite direct
    model.fit(rating_matrix)

    als_model_implicit = model
    print("Modelul ALS implicit a fost antrenat cu succes.")
    return model

def get_als_recommendations_implicit(user_id, top_n=10):
    global als_model_implicit, user_to_idx, movie_to_idx, rating_matrix
    if user_id not in user_to_idx:
        return []
    user_index = user_to_idx[user_id]
    user_items = rating_matrix.getrow(user_index)
    try:
        # Recomandările sunt returnate sub forma unui tuple de două array-uri:
        # (array_de_indexuri, array_de_scori)
        recs = als_model_implicit.recommend(user_index, user_items, N=top_n)
        print("Recomandări:", recs)
        # Extragem array-ul de indexuri
        rec_item_indexes = recs[0]
        # Construim dicționarul invers din movie_to_idx: {index: movie_id}
        index_to_movie = {i: m for m, i in movie_to_idx.items()}
        # Mapăm fiecare index la movie_id, filtrând eventualele indecși care nu se regăsesc
        rec_ids = [index_to_movie.get(int(idx)) for idx in rec_item_indexes if int(idx) in index_to_movie]
        return rec_ids
    except Exception as e:
        print("Error in get_als_recommendations_implicit:", e)
        return []



# La pornirea aplicației, antrenează modelul ALS implicit (o singură dată)
als_model_implicit = train_als_model_implicit()
def DB_to_Features():
    """
    Extrage textul (overview + genres) din tabela movies și returnează:
      - feature_matrice: matricea de features (concatenarea TF-IDF și Count)
      - movie_ids: lista ID-urilor filmelor (în aceeași ordine)
    """
    conn = sqlite3.connect('tmdb_movies.db')
    df = pd.read_sql_query("SELECT id, overview, genres FROM movies", conn)
    conn.close()
    
    # Combină câmpurile overview și genres (poți adăuga și titlul, dacă dorești)
    df['text'] = df['overview'].fillna('') + " " + df['genres'].fillna('')
    posts = df['text'].tolist()
    movie_ids = df['id'].tolist()
    
    # Vectorizare: folosim atât TF-IDF cât și CountVectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500)
    count_vectorizer = CountVectorizer(ngram_range=(1,2), max_features=500)
    tfidf_matrix = tfidf_vectorizer.fit_transform(posts).todense()
    count_matrix = count_vectorizer.fit_transform(posts).todense()
    
    # Dacă numărul de coloane este mai mic de 500, completează cu zerouri (opțional)
    if tfidf_matrix.shape[1] < 500:
        zero_fill = np.zeros((tfidf_matrix.shape[0], 500 - tfidf_matrix.shape[1]))
        tfidf_matrix = np.concatenate((tfidf_matrix, zero_fill), axis=1)
    if count_matrix.shape[1] < 500:
        zero_fill = np.zeros((count_matrix.shape[0], 500 - count_matrix.shape[1]))
        count_matrix = np.concatenate((count_matrix, zero_fill), axis=1)
    
    feature_matrix = np.concatenate((tfidf_matrix, count_matrix), axis=1)
    return feature_matrix, movie_ids

def norm_fro(x):
    return np.linalg.norm(x, ord="fro")

def isNaN(number):
    return number != number

def RegNMF(X, k, lamda, epsilon, print_loss=False):
    # Inițializează W și C cu valori aleatoare non-negative
    W = np.abs(np.random.rand(X.shape[0], k))
    C = np.abs(np.random.rand(k, X.shape[1]))
    i = 0
    if print_loss:
        print("\n" + "_"*100)
        print(" " * 23 + "Reg-NMF K: %d epsilon: %f lamda: %f" % (k, epsilon, lamda))
        print("_"*100)
    start_time = time.time()
    while True:
        W_step = W / (((W @ C) @ C.T) + lamda * W)
        W_new = np.multiply(W_step, (X @ C.T))
        C_step = C / (((W_new.T @ W_new) @ C) + lamda * C)
        C_new = np.multiply(C_step, (W_new.T @ X))
        error = np.abs((norm_fro(X - (W_new @ C_new))**2 - norm_fro(X - (W @ C))**2) / norm_fro(X)**2)
        W, C = W_new, C_new
        i += 1
        if print_loss:
            print("[Iteration %d] Error: %.6f" % (i, error))
        if error < epsilon or isNaN(error):
            total_time = time.time() - start_time
            if print_loss:
                print("Minimun error reached:", error)
                print("Iterations:", i)
                print("Time:", total_time)
            return error, norm_fro(X - (W @ C))**2, norm_fro(W)**2, norm_fro(C)**2, total_time, i, W, C

def get_closest_vectors_order(v1, v2):
    distances = cosine_similarity(v1, v2)
    return (-distances).T.argsort()

def get_recommendations_NMF(movie_id, n_components=7, max_iter=200):
    global movies_df
    # Asigură-te că avem datele filmelor
    if movies_df is None or movies_df.empty:
        # Recalculează dacă este nevoie (folosind compute_similarity sau altă funcție similară)
        compute_similarity()
    # Verificăm dacă movie_id există în baza de date
    if movie_id not in movies_df['id'].values:
        return []
    
    # Construim matricea de text (folosind câmpurile 'overview' și 'genres')
    # Dacă există o coloană 'tags' în movies_df, se poate folosi aceasta
    if 'tags' in movies_df.columns:
        corpus = movies_df['tags'].values.astype('U')
    else:
        corpus = (movies_df['overview'] + " " + movies_df['genres']).values.astype('U')
    
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    count_matrix = vectorizer.fit_transform(corpus)
    
    # Aplicăm NMF pentru a factoriza matricea
    nmf_model = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
    W_nmf = nmf_model.fit_transform(count_matrix)
    
    # Găsim indexul filmului de referință
    movie_index = movies_df.index[movies_df['id'] == movie_id].tolist()[0]
    movie_latent = W_nmf[movie_index]
    
    # Calculăm similaritatea cosinus între vectorul latent al filmului și toți vectorii
    latent_similarities = cosine_similarity([movie_latent], W_nmf)[0]
    #print(f"NMF functie recommended movie IDs for movie : {latent_similarities}")

    # Sortăm indecșii în ordine descrescătoare a similarității
    similar_indices = np.argsort(latent_similarities)[::-1]
    # Eliminăm filmul de referință
    similar_indices = [i for i in similar_indices if i != movie_index]
    # Luăm primele n_components (sau top_n recomandări dorite)
    top_indices = similar_indices[:n_components]
    recommended_ids = movies_df.iloc[top_indices]['id'].tolist()
    #print(f"NMF functie recommended movie IDs for movie : {recommended_ids}")

    return recommended_ids

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit

def build_gmf_mlp_model(num_users: int,
                        num_items: int,
                        latent_dim: int = 8,
                        hidden: tuple = (32, 16),
                        dropout: float = 0.50,
                        reg_emb: float = 5e-5,
                        reg_dense: float = 1e-3,
                        final_activation: str = "sigmoid") -> Model:

    u_in = layers.Input((1,), name="user_in", dtype="int32")
    i_in = layers.Input((1,), name="item_in", dtype="int32")

    # --- GMF
    u_g = layers.Embedding(num_users, latent_dim,
                           embeddings_regularizer=regularizers.l2(reg_emb))(u_in)
    i_g = layers.Embedding(num_items, latent_dim,
                           embeddings_regularizer=regularizers.l2(reg_emb))(i_in)
    gmf = layers.multiply([layers.Flatten()(u_g), layers.Flatten()(i_g)])

    # --- MLP
    u_m = layers.Embedding(num_users, latent_dim,
                           embeddings_regularizer=regularizers.l2(reg_emb))(u_in)
    i_m = layers.Embedding(num_items, latent_dim,
                           embeddings_regularizer=regularizers.l2(reg_emb))(i_in)
    mlp = layers.Concatenate()([layers.Flatten()(u_m), layers.Flatten()(i_m)])

    for k, units in enumerate(hidden, 1):
        mlp = layers.Dense(units, activation="relu",
                           kernel_regularizer=regularizers.l2(reg_dense),
                           name=f"mlp_{k}")(mlp)
        mlp = layers.Dropout(dropout)(mlp)

    x = layers.Concatenate()([gmf, mlp])
    x = layers.Dropout(dropout)(x)                      # extra dropout

    out = layers.Dense(1, activation=final_activation,
                       kernel_regularizer=regularizers.l2(reg_dense))(x)

    model = Model([u_in, i_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss="mse",
                  metrics=["mae"])
    return model


# ------------------------------------------------------------------
# 2. TRAIN cu split pe user
# ------------------------------------------------------------------
def train_and_save_gmf_mlp(db_path="tmdb_movies_for_nn.db",
                           save_path="gmf_mlp_model_tf"):

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)

    if df.empty:
        print("Nu există rating-uri.")
        return

    # remap id-uri
    u_map = {u: idx for idx, u in enumerate(df.user_id.unique())}
    i_map = {m: idx for idx, m in enumerate(df.movie_id.unique())}
    df["u_idx"] = df.user_id.map(u_map)
    df["i_idx"] = df.movie_id.map(i_map)
    df["y"] = (df.rating - 1) / 4.0        # [0,1]

    # split 80/20 pe user (fiecare user exclusiv într-un singur set)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df.u_idx))
    tr, val = df.iloc[train_idx], df.iloc[val_idx]

    X_tr = [tr.u_idx.values,  tr.i_idx.values];   y_tr = tr.y.values
    X_val = [val.u_idx.values, val.i_idx.values]; y_val = val.y.values

    model = build_gmf_mlp_model(num_users=len(u_map),
                                num_items=len(i_map))

    early  = EarlyStopping(monitor="val_loss", patience=1,
                           restore_best_weights=True, verbose=1)
    reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                               patience=1, min_lr=1e-5, verbose=1)

    model.fit(X_tr, y_tr,
              epochs=30,
              batch_size=128,
              validation_data=(X_val, y_val),
              callbacks=[early, reduce],
              verbose=2)

    model.save(save_path, save_format="tf")
    np.save("user_map_gmfmlp.npy", u_map)
    np.save("item_map_gmfmlp.npy", i_map)
    print(f"GMF+MLP salvat în «{save_path}». users={len(u_map)}, items={len(i_map)}")
    y_pred_norm = model.predict(X_val).flatten()
    rmse_norm = np.sqrt(np.mean((y_pred_norm - y_val) ** 2))
    print(f"GMF+MLP RMSE normalizat (0–1): {rmse_norm:.4f}")
    rmse_orig = rmse_norm * 4   # [0,1]→[1,5] prin *4
    print(f"GMF+MLP RMSE pe scara 1–5: {rmse_orig:.4f}")

gmf_mlp_model = None
user_map_gmfmlp = None
item_map_gmfmlp = None

def load_gmf_mlp_model(model_path="gmf_mlp_model_tf"):
    """
    Încarcă modelul GMF+MLP din path + dicționarele user_map, item_map
    """
    global gmf_mlp_model, user_map_gmfmlp, item_map_gmfmlp

    # Încarcă modelul
    gmf_mlp_model = tf.keras.models.load_model(model_path, compile=False)
    
    # Încarcă mapările
    user_map_gmfmlp = np.load("user_map_gmfmlp.npy", allow_pickle=True).item()
    item_map_gmfmlp = np.load("item_map_gmfmlp.npy", allow_pickle=True).item()
    
    print("Modelul GMF+MLP a fost încărcat împreună cu mapările user/item.")

def recommend_movies_gmf_mlp(user_id, top_n=5):
    """
    Returnează o listă de movie_id recomandate userului (user_id).
    Folosește modelul global gmf_mlp_model + mapările user_map_gmfmlp/item_map_gmfmlp.
    """
    if gmf_mlp_model is None:
        print("Eroare: Modelul GMF+MLP nu este încărcat.")
        return []

    if user_id not in user_map_gmfmlp:
        print(f"User {user_id} nu se găsește în user_map_gmfmlp. Fallback.")
        return []

    # 1) Construim listele (X) pentru toți itemii
    all_item_ids = list(item_map_gmfmlp.keys())
    user_idx = user_map_gmfmlp[user_id]

    user_input = []
    item_input = []
    for iid in all_item_ids:
        user_input.append(user_idx)
        item_input.append(item_map_gmfmlp[iid])

    user_input = np.array(user_input)
    item_input = np.array(item_input)

    # 2) Prezicem
    preds = gmf_mlp_model.predict([user_input, item_input], verbose=0).flatten()
    # Reconstruim ratingul 1..5 (dacă dorim), dar în exemplu doar sortăm după preds
    # preds_1_5 = 1.0 + preds * 4.0

    # 3) Sortăm descrescător
    indexed_preds = list(zip(all_item_ids, preds))
    indexed_preds.sort(key=lambda x: x[1], reverse=True)

    # 4) Luăm top_n
    top_recs = [movie_id for (movie_id, score) in indexed_preds[:top_n]]
    return top_recs

def create_ratings_matrix():
    """
    Returns:
      – matrix: a dense numpy array of shape (n_users, n_movies)
      – user_to_idx, movie_to_idx: mappings from raw IDs to row/col indices
    """
    conn = sqlite3.connect('tmdb_movies_for_nn.db')
    ratings_df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()

    # get the *unique* IDs
    unique_users  = ratings_df['user_id'].unique()
    unique_movies = ratings_df['movie_id'].unique()

    user_to_idx  = {u:i for i,u in enumerate(unique_users)}
    movie_to_idx = {m:i for i,m in enumerate(unique_movies)}

    # now we only allocate exactly what we need
    matrix = np.zeros((len(unique_users), len(unique_movies)), dtype=np.float32)

    for _, row in ratings_df.iterrows():
        ui = user_to_idx[row['user_id']]
        mi = movie_to_idx[row['movie_id']]
        matrix[ui, mi] = row['rating']

    return matrix, user_to_idx, movie_to_idx


# Modelul Autoencoder (SAE) – adaptat din exemplul dat
import torch
import torch.nn as nn
import torch.optim as optim

class SAE(nn.Module):
    def __init__(self, nb_movies):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

def train_autoencoder_model(nb_epoch=2, model_path="sae_model.pt", tensor_path="sae_ratings_tensor.pt"):
    ratings_matrix, _, _ = create_ratings_matrix()
    nb_users, nb_movies = ratings_matrix.shape

    # Prepare data and model
    training_tensor = torch.FloatTensor(ratings_matrix)
    sae = SAE(nb_movies)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

    # Training loop
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        for id_user in range(nb_users):
            input_user = training_tensor[id_user].unsqueeze(0)
            target = input_user.clone()
            if torch.sum(target > 0) > 0:
                output = sae(input_user)
                target.requires_grad = False
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_movies / float(torch.sum(target > 0) + 1e-10)
                loss.backward()
                optimizer.step()
                train_loss += np.sqrt(loss.item() * mean_corrector)
                s += 1.
        print(f'Autoencoder epoch: {epoch} loss: {train_loss/s:.6f}')

    # *** Save model and tensor ***
    torch.save({
        'model_state_dict': sae.state_dict(),
    }, model_path)
    torch.save(training_tensor, tensor_path)
    print(f"Saved SAE model to {model_path} and ratings tensor to {tensor_path}")

    return sae, training_tensor

def load_autoencoder_model(model_path="sae_model.pt", tensor_path="sae_ratings_tensor.pt"):
    """
    Loads the SAE model and the ratings tensor from disk.
    Returns (sae, ratings_tensor).
    """
    # Reconstruct the architecture first (we need nb_movies)
    # If you know nb_movies ahead of time, you can hardcode it,
    # or better: save nb_movies alongside the state_dict.
    # Here we load the tensor first to get its shape:
    ratings_tensor = torch.load(tensor_path)
    nb_movies = ratings_tensor.size(1)

    sae = SAE(nb_movies)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()  # set to eval mode if you’re only doing inference

    return sae, ratings_tensor

# Funcţie pentru a genera recomandări folosind modelul SAE
def get_autoencoder_recommendations(user_id, sae, ratings_tensor, top_n=7):
    nb_users, nb_movies = ratings_tensor.shape
    # Ajustăm indexul: presupunem că user_id este 1-indexat
    user_idx = user_id - 1
    #AICI verifici că user_idx e valid în [0, nb_users)
    if user_idx < 0 or user_idx >= nb_users:
        print(f"[DEBUG] user_id={user_id} e în afara intervalului (0..{nb_users-1}). Fallback.")
        return []
    
    input_user = ratings_tensor[user_idx].unsqueeze(0)
    # Prezicem ratingurile
    output = sae(input_user).detach().numpy().flatten()
    # Pentru filmele deja notate, setăm predicţia la -inf pentru a le exclude
    user_ratings = ratings_tensor[user_idx].numpy()
    output[user_ratings > 0] = -np.inf
    # Sortăm și alegem top_n
    recommended_indices = np.argsort(output)[::-1][:top_n]
    # Convertim indexii la movie_id (presupunem că movie_id = index+1)
    recommended_movie_ids = [int(idx)+1 for idx in recommended_indices]
    return recommended_movie_ids

sae_model, ratings_tensor = load_autoencoder_model()


class MovieLensNet(nn.Module):
    def __init__(self, num_movies, num_users, num_genres_encoded, embedding_size, hidden_dim):
        super(MovieLensNet, self).__init__()
        self.num_movies = num_movies
        self.num_users = num_users
        self.num_genres_encoded = num_genres_encoded
        
        # embedding-uri pentru filme și useri
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        
        # fully connected layers
        self.fc1 = nn.Linear(embedding_size * 2 + num_genres_encoded, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, movie_id, user_id, genre_id):
        """
        movie_id: [batch_size]
        user_id: [batch_size]
        genre_id: [batch_size, num_genres_encoded]  (aici punem OneHot sau alt encoding)
        """
        # Embedding pentru movie_id și user_id
        movie_emb = self.movie_embedding(movie_id)  # => [batch_size, embedding_size]
        user_emb = self.user_embedding(user_id)     # => [batch_size, embedding_size]
        
        # Concat: [batch_size, embedding_size + embedding_size + num_genres_encoded]
        x = torch.cat([movie_emb, user_emb, genre_id.float()], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Ultimul layer - iese scorul (ratingul prezis)
        x = self.fc3(x).squeeze(1)  # => [batch_size]
        return x

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, movies_df, user_map, movie_map, genre_map):
        """
        ratings_df: DataFrame cu coloanele [user_id, movie_id, rating] (atenție: user_id / movie_id pot fi 'raw')
        movies_df: DataFrame cu coloanele [movieId, genres]
        user_map, movie_map: dicționare care map[ează ID-ul original la un index 0..num_users-1 / 0..num_movies-1]
        genre_map: ex. un MultiLabelBinarizer sau un CountVectorizer pentru genuri
        """
        self.ratings = ratings_df.reset_index(drop=True)
        self.movies = movies_df
        self.user_map = user_map
        self.movie_map = movie_map
        self.genre_map = genre_map  # ex. MultiLabelBinarizer
        
        # opțional, transformăm direct user_id și movie_id în indici
        self.ratings['user_idx'] = self.ratings['user_id'].map(self.user_map)
        self.ratings['movie_idx'] = self.ratings['movie_id'].map(self.movie_map)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_idx = int(row['user_idx'])
        movie_idx = int(row['movie_idx'])
        rating = float(row['rating'])
        
        # Obține genurile filmului sub formă one-hot
        # Presupunem că genre_map = un dict { movie_id -> torch.tensor([...]) }
        # sau construim direct la cerere:
        genre_vec = self.genre_map[movie_idx]  # [num_genres], de exemplu
        
        return {
            'user_id': torch.tensor(user_idx, dtype=torch.long),
            'movie_id': torch.tensor(movie_idx, dtype=torch.long),
            'genre_id': genre_vec,  # deja e tensori / tens. float
            'rating': torch.tensor(rating, dtype=torch.float)
        }

def train_gnn_model(db_path="tmdb_movies_for_nn.db", model_path="gnn_model.pt", epochs=10):
    # 1) Citește ratingurile din DB
    conn = sqlite3.connect(db_path)
    ratings_df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    
    # 2) Citește tabela movies (pentru genuri)
    movies_df = pd.read_sql_query("SELECT id AS movie_id, genres FROM movies", conn)
    conn.close()
    
    if ratings_df.empty:
        print("Nu există date de antrenare (ratings) în DB.")
        return
    
    # Mapăm user_id și movie_id la range(0..n-1)
    unique_users = ratings_df['user_id'].unique()
    unique_movies = movies_df['movie_id'].unique()
    
    user_map = {u: i for i, u in enumerate(unique_users)}
    movie_map = {m: i for i, m in enumerate(unique_movies)}
    num_users = len(user_map)
    num_movies = len(movie_map)
    
    # Preprocesăm genurile și creăm un "genre_map" => index film -> vector one-hot
    #  - genurile sunt stocate ca un string "Action, Drama"
    #  - le split-uim pe virgula/pipe
    mlb = MultiLabelBinarizer()
    # Transformăm coloana genres într-o listă de liste
    movies_df['genres_list'] = movies_df['genres'].fillna('').apply(lambda x: x.split(','))
    
    mlb.fit(movies_df['genres_list'])
    all_genres = mlb.classes_  # de ex. ["Action", "Drama", ...]
    num_genres = len(all_genres)
    
    # Creăm un dicționar: movie_idx -> tens. float [num_genres]
    genre_map = {}
    for i, row in movies_df.iterrows():
        mid = row['movie_id']
        m_idx = movie_map[mid]  # index 0..(num_movies-1)
        bin_array = mlb.transform([row['genres_list']])[0]  # e un vector numpy, ex. [1,0,1...]
        genre_map[m_idx] = torch.tensor(bin_array, dtype=torch.float)
    
    # 3) Construim Dataset & DataLoader
    dataset = MovieLensDataset(ratings_df, movies_df, user_map, movie_map, genre_map)
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    print("sun t gnn")
    # 4) Instanțiem modelul
    model = MovieLensNet(
        num_movies=num_movies,
        num_users=num_users,
        num_genres_encoded=num_genres,
        embedding_size=16, 
        hidden_dim=64
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5) Antrenare
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            user_id = batch['user_id']
            movie_id = batch['movie_id']
            genre_id = batch['genre_id']  # [batch_size, num_genres]
            rating = batch['rating']
            
            optimizer.zero_grad()
            outputs = model(movie_id, user_id, genre_id)
            loss = criterion(outputs, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}", flush=True)
    
    # 6) Salvează modelul + datele auxiliare (mapările)
    torch.save({
        'model_state_dict': model.state_dict(),
        'user_map': user_map,
        'movie_map': movie_map,
        'genre_map': genre_map,
        'num_users': num_users,
        'num_movies': num_movies,
        'num_genres': num_genres
    }, model_path)
    print(f"Model GNN salvat în {model_path}.")

def load_gnn_model(model_path="gnn_model.pt"):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    num_users = checkpoint['num_users']
    num_movies = checkpoint['num_movies']
    num_genres = checkpoint['num_genres']
    
    model = MovieLensNet(num_movies, num_users, num_genres, embedding_size=16, hidden_dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    user_map = checkpoint['user_map']
    movie_map = checkpoint['movie_map']
    genre_map = checkpoint['genre_map']
    
    return model, user_map, movie_map, genre_map

def get_gnn_recommendations(user_id_real, model, user_map, movie_map, genre_map, top_n=7):
    """
    user_id_real: ID real din DB (cum apare în ratings.user_id)
    """
    # Verificăm dacă userul există în user_map
    if user_id_real not in user_map:
        return []
    user_idx = user_map[user_id_real]
    
    # Construim listă de (movie_idx, scor prezis)
    movie_scores = []
    # Parcurgem toate filmele cunoscute de map
    for m_real, m_idx in movie_map.items():
        # Genre vector
        genre_vec = genre_map[m_idx]  # [num_genres]
        
        # Pregătim tensori
        u_t = torch.tensor([user_idx], dtype=torch.long)
        m_t = torch.tensor([m_idx], dtype=torch.long)
        g_t = genre_vec.unsqueeze(0) # => [1, num_genres]
        
        with torch.no_grad():
            score = model(m_t, u_t, g_t).item()
        movie_scores.append((m_real, score))
    
    # sortăm descrescător după score
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = movie_scores[:top_n]
    # extragem doar ID-urile reale de film
    recommended_movie_ids = [x[0] for x in top_items]
    return recommended_movie_ids


knn_algo = None 
trainset_knn = None 

def initialize_knn_model(db_path="tmdb_movies.db"):
    """
    1) Citește rating-urile din DB
    2) Creează setul de date Surprise
    3) Antrenează modelul KNN
    """
    global knn_algo, trainset_knn

    # 1) Citește rating-urile
    conn = sqlite3.connect(db_path)
    ratings_df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()
    print("TESTARE KNN:")
    print(ratings_df[ratings_df["user_id"] == 1])
    print(ratings_df["user_id"].unique())
    print("TESTARE KNN:")
    if ratings_df.empty:
        print("Nu există rating-uri în baza de date, KNN nu va fi inițializat.")
        knn_algo = None
        trainset_knn = None
        return

    # 2) Construiește dataset Surprise
    #   - rating_scale=(1,5) dacă ratingurile sunt între 1 și 5
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    trainset_knn = data.build_full_trainset()

    # 3) Definim un model KNN (User-based sau Item-based)
    #    - sim_options = {'user_based': True} => user-based
    #    - 'name': 'cosine' sau 'pearson_baseline' etc.
    sim_options = {
        'name': 'pearson',    # sau 'cosine'
        'user_based': True    # sau False pentru item-based
    }
    knn_algo = KNNBasic(sim_options=sim_options)
    knn_algo.fit(trainset_knn)
    print("[KNN] Model KNN antrenat cu succes.")


def get_knn_recommendations(user_id, top_n=10):
    global knn_algo, trainset_knn

    if knn_algo is None or trainset_knn is None:
        print("[KNN] Modelul nu este inițializat sau nu există rating-uri.")
        return []

    # Verificăm direct cu int
    if user_id not in trainset_knn._raw2inner_id_users:
        print(f"[KNN] User {user_id} nu există în setul de train. Fallback => nimic.")
        return []

    testset = trainset_knn.build_anti_testset()
    # Filtrăm direct int
    user_testset = [t for t in testset if t[0] == user_id]

    predictions = knn_algo.test(user_testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommended_ids = [int(x.iid) for x in predictions[:top_n]]
    return recommended_ids


# În app.py, după celelalte importuri și definiții de @app.route

from flask import g

@app.before_request
def load_logged_in_user():
    """Stochează în `g.user` datele curente ale utilizatorului, dacă e logat."""
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        conn = sqlite3.connect('tmdb_movies.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, name, email, age, occupation, sex, address FROM users WHERE id = ?',
            (user_id,)
        )
        g.user = cursor.fetchone()
        conn.close()

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    uid = session.get('user_id')  # your logged‐in user id

    # — POST: attempt password change —
    if request.method == 'POST':
        cur_pwd  = request.form.get('current_password','').strip()
        new_pwd  = request.form.get('new_password','').strip()
        conf_pwd = request.form.get('confirm_password','').strip()
        
        if not (cur_pwd and new_pwd and conf_pwd):
            flash('Completează toate câmpurile.', 'danger')
        elif new_pwd != conf_pwd:
            flash('Parolele nu coincid.', 'danger')
        else:
            con = sqlite3.connect('tmdb_movies.db')
            cur = con.cursor()
            row = cur.execute(
                'SELECT password FROM users WHERE id=?',
                (uid,)
            ).fetchone()
            if not row or not check_password_hash(row[0], cur_pwd):
                flash('Parola curentă este incorectă.', 'danger')
            else:
                new_hash = generate_password_hash(new_pwd)
                cur.execute(
                    'UPDATE users SET password=? WHERE id=?',
                    (new_hash, uid)
                )
                con.commit()
                flash('Parola a fost schimbată cu succes!', 'success')
            con.close()

        # after POST redirect back so GET logic runs
        return redirect(url_for('profile'))

    # — GET: load user info, ratings, selections —
    con = sqlite3.connect('tmdb_movies.db')
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # load full user record
    cur.execute('SELECT * FROM users WHERE id=?', (uid,))
    user = cur.fetchone()  # will be a Row

    # user ratings
    cur.execute('''
      SELECT m.id AS movie_id, m.title, r.rating
      FROM ratings r
      JOIN movies m ON r.movie_id=m.id
      WHERE r.user_id=?
      ORDER BY r.id DESC
    ''', (uid,))
    user_ratings = cur.fetchall()

    # selected movies
    cur.execute('''
      SELECT m.id AS movie_id, m.title
      FROM user_selected_movies u
      JOIN movies m ON u.movie_id=m.id
      WHERE u.user_id=?
    ''', (uid,))
    selected_movies = cur.fetchall()

    con.close()

    return render_template(
      'profile.html',
      user=user,
      user_ratings=user_ratings,
      selected_movies=selected_movies
    )



_neural_home_cache = {
    "nn": None,
    "gmf": None,
    "gnn": None,
    "als": None,
    "nmf": None,
    "autoencoder": None
}

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
    
     # ==== New sections: “Filme Noi” și “Filme Populare” ====
    cursor.execute('''
        SELECT id, title, poster_path
        FROM movies
        ORDER BY release_date DESC
        LIMIT 17
    ''')
    new_movies = cursor.fetchall()

    cursor.execute('''
        SELECT id, title, poster_path
        FROM movies
        ORDER BY popularity DESC
        LIMIT 17
    ''')
    popular_movies = cursor.fetchall()

    # 1) Extragem toate genurile unice din coloana `genres`
    cursor.execute("SELECT genres FROM movies")
    all_genres = set()
    for (genres_str,) in cursor.fetchall():
        if not genres_str:
            continue
        # presupunem că genurile sunt stocate ca "Acțiune,Comedie,Drama"
        for g in genres_str.split(','):
            all_genres.add(g.strip())

    # 2) Pentru fiecare gen, facem query-ul de top filme
    genre_lists = {}
    for genre in sorted(all_genres):
        cursor.execute("""
            SELECT id, title, poster_path
            FROM movies
            -- adăugăm virgule la început și sfârșit pentru a evita potriviri parțiale
            WHERE ',' || genres || ',' LIKE ?
            ORDER BY popularity DESC
            LIMIT 17
        """, (f'%,{genre},%',))
        genre_lists[genre] = cursor.fetchall()

    
    
    # Calculate total pages based on the number of movies
    cursor.execute('SELECT COUNT(*) FROM movies')
    total_movies = cursor.fetchone()[0]
    total_pages = (total_movies // per_page) + (1 if total_movies % per_page > 0 else 0)

    if page < 1 or page > total_pages:
        flash(f'Page {page} does not exist. Please choose a number between 1 and {total_pages}.', 'warning')
        conn.close()
        return redirect(url_for('index', sort_by=sort_by))
    
    
    # Calcularea mediei notelor din recenzii pentru fiecare film și obținerea primelor 10
    cursor.execute('''
    SELECT movies.id, movies.title, movies.poster_path, COALESCE(AVG(reviews.grade), 0) AS avg_grade
    FROM movies
    LEFT JOIN reviews ON movies.id = reviews.movie_id
    GROUP BY movies.id
    ORDER BY avg_grade DESC
    LIMIT 7
    ''')
    top_rated_movies_by_reviews = cursor.fetchall()

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
    click_based_recommended_movies = []

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
   
    # Apelezi funcția "all-in-one":
    weighted_mean_recommended_movie_ids = []
    if user_id:
        weighted_mean_recommended_movie_ids = user_based_weighted_mean_recommendation(user_id, top_n=7)
    
    # Apoi  query în DB ca să obții poster_path, title etc.
    weighted_mean_recommended_movies = []
    if weighted_mean_recommended_movie_ids:
        placeholders = ','.join('?' for _ in weighted_mean_recommended_movie_ids)
        cursor.execute(f'''
            SELECT id, title, poster_path
            FROM movies
            WHERE id IN ({placeholders})
        ''', weighted_mean_recommended_movie_ids)
        weighted_mean_recommended_movies = cursor.fetchall()
       
      # 1) Recomandări Weighted Mean bazate pe gen
    if r_matrix is None:
        # Nu avem date de rating deloc; deci nu putem face user-based
        gender_based_recommended_ids = []
    else:
        gender_based_recommended_ids = get_top_recommendations(cf_user_wmean_gender, user_id, top_n=7)


    # 2) Recomandări Weighted Mean bazate pe gen + ocupație
    if r_matrix is not None:
        gen_occ_based_recommended_ids = get_top_recommendations(cf_user_wmean_gen_occ, user_id, top_n=7)
    else:
        gen_occ_based_recommended_ids = []


    # Citești detaliile din DB
    gender_based_recommended_movies = []
    if gender_based_recommended_ids:
        placeholders = ','.join('?' for _ in gender_based_recommended_ids)
        query = f"SELECT id, title, poster_path FROM movies WHERE id IN ({placeholders})"
        cursor.execute(query, gender_based_recommended_ids)
        gender_based_recommended_movies = cursor.fetchall()

    gen_occ_based_recommended_movies = []
    if gen_occ_based_recommended_ids:
        placeholders = ','.join('?' for _ in gen_occ_based_recommended_ids)
        query = f"SELECT id, title, poster_path FROM movies WHERE id IN ({placeholders})"
        cursor.execute(query, gen_occ_based_recommended_ids)
        gen_occ_based_recommended_movies = cursor.fetchall()
    
    
    # Click-Based Recommendations
    if user_id and algo is not None:
        click_based_movie_ids = get_collaborative_click_recommendations(user_id, top_n=7)
        print(f"Click-Based Recommended movie IDs for user {user_id}: {click_based_movie_ids}")  # Debugging

        if click_based_movie_ids:
            cursor.execute(f'''
                SELECT id, title, poster_path
                FROM movies
                WHERE id IN ({','.join('?' * len(click_based_movie_ids))})
            ''', click_based_movie_ids)
            click_based_recommended_movies = cursor.fetchall()
            print(f"Click-Based Recommended movies: {click_based_recommended_movies}")  # Debugging
        
         # -- AICI modificăm pentru NN recommendations -- #
    if _neural_home_cache["nn"] is None:
        if user_id and nn_model is not None:
            nn_recommended_ids = recommend_movies_nn(
            user_id, nn_model, user_map, movie_map, num_recommendations=7
            )
            print(f"NN model recommended for user {user_id}: {nn_recommended_ids}")

            nn_recommended_movies = []
            for mid in nn_recommended_ids:
                movie_data = fetch_tmdb_movie_data(mid)
                if movie_data:
                    nn_recommended_movies.append(movie_data)
            _neural_home_cache["nn"] = nn_recommended_movies
    nn_recommended_movies=_neural_home_cache["nn"]
    print(f"[DEBUG] NN recommended (fără DB) pentru user {user_id}: {nn_recommended_movies}")

      # Obține recomandări ALS
        #als_recommended_movie_ids = []
        # if user_id:
        # als_recommended_movie_ids = get_als_recommendations(user_id, top_n=7)
        #als_recommended_movies = []
        #if als_recommended_movie_ids:
        #   placeholders = ','.join('?' for _ in als_recommended_movie_ids)
        #cursor.execute(f'''
        #   SELECT id, title, poster_path
        #   FROM movies
        #  WHERE id IN ({placeholders})
        #   ''', als_recommended_movie_ids)
        # als_recommended_movies = cursor.fetchall()

     
        #print(f"ALS {user_id}: {als_recommended_movies}")
            # Obține recomandări ALS din modelul implicit
        # Obține recomandări ALS folosind modelul implicit
    user_id = session.get('user_id')
    # -- AICI modificăm pentru ALS implicit recommendations -- #
    if _neural_home_cache["als"] is None:
        if user_id and als_model_implicit is not None:
            als_recommended_ids = get_als_recommendations_implicit(user_id, top_n=7)
            print(f"ALS implicit model recommended for user {user_id}: {als_recommended_ids}")
        
        als_recommended_movies = []
        for mid in als_recommended_ids:
            movie_data = fetch_tmdb_movie_data(mid)
            if movie_data:
                als_recommended_movies.append(movie_data)
        _neural_home_cache["als"]= als_recommended_movies
    als_recommended_movies=_neural_home_cache["als"]
    print(f"[DEBUG] ALS recommended (fără DB) for user {user_id}: {als_recommended_movies}")

    # --- Recomandări NMF ---
    nmf_recommended_movies = []
    if _neural_home_cache["nmf"] is None:
        if user_id:
            # Folosim ca referință un film din tabela user_selected_movies
            cursor.execute("SELECT movie_id FROM user_selected_movies WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
            row = cursor.fetchone()
            if row:
                current_movie_id = row[0]
                nmf_recommended_ids = get_recommendations_NMF(current_movie_id, n_components=7)
                print(f"NMF recommended movie IDs for movie {current_movie_id}: {nmf_recommended_ids}")
                if nmf_recommended_ids:
                    placeholders = ','.join('?' for _ in nmf_recommended_ids)
                    cursor.execute(f'''            
                        SELECT id, title, poster_path
                        FROM movies
                        WHERE id IN ({placeholders})
                    ''', nmf_recommended_ids)
                    nmf_recommended_movies = cursor.fetchall()
            else:
                # Caz fallback: userul nu are niciun film selectat
                current_movie_id = None
                nmf_recommended_ids = []
                print("[DEBUG] user nu are user_selected_movies => nmf_recommended_ids = []")

    # 1. Apelează funcția de recomandare GMF+MLP
    if _neural_home_cache["gmf"] is None:
        recommended_idss = recommend_movies_gmf_mlp(user_id, top_n=7)
        print(f"NMF_GMF recommended movie IDs for movie {current_movie_id}: {recommended_idss}")

        # 2. Pentru fiecare ID, apelează fetch_tmdb_movie_data,
        #    la fel ca în cazul ALS, și salvează doar cele care au date valide
        recommended_moviesgmf = []
        for mid in recommended_idss:
            movie_data = fetch_tmdb_movie_data(mid)
            if movie_data:
                recommended_moviesgmf.append(movie_data)
        _neural_home_cache["gmf"]=recommended_moviesgmf
    recommended_moviesgmf=_neural_home_cache["gmf"]
    # 3. Log pentru debug
    print(f"[DEBUG] NMF_GMF recommended (fără DB) for user {user_id}: {recommended_moviesgmf}")
     # --- Recomandări Autoencoder ---
    if _neural_home_cache["autoencoder"] is None: 
        auto_recommended_movies = []
        if user_id:
            # Pentru demo, antrenăm modelul autoencoder pe loc.
            auto_rec_ids = get_autoencoder_recommendations(user_id, sae_model, ratings_tensor, top_n=7)
            print(f"Autoencoder recommended movie IDs for user {user_id}: {auto_rec_ids}")

            for mid in auto_rec_ids:
                movie_data = fetch_tmdb_movie_data(mid)
                if movie_data:
                    auto_recommended_movies.append(movie_data)
            _neural_home_cache["autoencoder"]=auto_recommended_movies
    auto_recommended_movies=_neural_home_cache["autoencoder"]
    # Apelare GNN dacă modelul e încărcat
    gnn_recommended_movies = []
    if user_id and gnn_model is not None:
        gnn_rec_ids = get_gnn_recommendations(
        user_id_real=user_id,
        model=gnn_model,
        user_map=gnn_user_map,
        movie_map=gnn_movie_map,
        genre_map=gnn_genre_map,
        top_n=7
    )
    print(f"[DEBUG] GNN recommended for user {user_id}: {gnn_rec_ids}")
    if _neural_home_cache["gnn"] is None:
        if gnn_rec_ids:
            # În loc să query-uim DB, apelăm direct fetch_tmdb_movie_data(mid)
            for mid in gnn_rec_ids:
                movie_data = fetch_tmdb_movie_data(mid)
                if movie_data:
                    gnn_recommended_movies.append(movie_data)
            _neural_home_cache["gnn"]= gnn_recommended_movies
    gnn_recommended_movies= _neural_home_cache["gnn"]
    # Afișare pentru debug
    print(f"[DEBUG] GNN recommended (fără DB) for user {user_id}: {gnn_recommended_movies}")
    
     # Apelare KNN-based
    knn_recommended_movies = []
    if user_id and knn_algo is not None:
        knn_rec_ids = get_knn_recommendations(user_id, top_n=7)
        print(f"[DEBUG] KNN recommended for user {user_id}: {knn_rec_ids}")

        if knn_rec_ids:
            # Ia datele filmelor din DB: id, title, poster_path
            placeholders = ','.join('?' for _ in knn_rec_ids)
            cursor.execute(f'''
                SELECT id, title, poster_path
                FROM movies
                WHERE id IN ({placeholders})
            ''', knn_rec_ids)
            knn_recommended_movies = cursor.fetchall()
            print(f"[DEBUG] KNN recommended movies: {knn_recommended_movies}")
    
    
    conn.close()
    print(f"ALS {user_id}: {als_recommended_movies}")
    user_name = session.get('user_name')

    return render_template(
        'home.html',
        movies=movies,
        new_movies=new_movies,
        popular_movies=popular_movies,
        **{ f"{genre.lower()}_movies": lst for genre, lst in genre_lists.items() },
        top_rated_movies_by_reviews=top_rated_movies_by_reviews,
        recommended_movies=recommended_movies,
        knn_recommended_movies=knn_recommended_movies,
        recommended_moviesgmf=recommended_moviesgmf,
        gnn_recommended_movies=gnn_recommended_movies,
        auto_recommended_movies=auto_recommended_movies,
        nmf_recommended_movies=nmf_recommended_movies,
        als_recommended_movies=als_recommended_movies,
        gender_based_recommended_movies=gender_based_recommended_movies,
        gen_occ_based_recommended_movies=gen_occ_based_recommended_movies,
        nn_recommended_movies=nn_recommended_movies,
        weighted_mean_recommended_movies=weighted_mean_recommended_movies,
        content_based_recommended_movies=content_based_recommended_movies,
        click_based_recommended_movies=click_based_recommended_movies,
        page=page,
        total_pages=total_pages,
        user_name=user_name,
        sort_by=sort_by
    )

@app.route('/session')
def session_info():
    return jsonify(user=session.get('user_name'))


# Filtrul pentru expresii ambigue și structuri de negație
def preprocess_review_text(review_text):
    # Structuri comune pentru negație și ambiguitate în română și engleză
    neutral_patterns = [
        r'\bnot\b.*?\b(bad|terrible|disappointing|the best|impressive)\b',
        r'\bnot\b.*?\b(too|that|really)?\b.*?\b(bad|great|impressive|amazing|perfect)\b',
        r'\bnot\b.*?\b(as|so)?\b.*?\b(bad|good|terrible|excellent|the worst|the best)\b',
        r'\bnu\b.*?\b(prea|chiar|foarte)?\b.*?\b(rău|rau|bun|groaznic|perfect|dezamăgitor)\b',
        r'\bnici\b.*?\b(prea|foarte)?\b.*?\b(bun|rau|rău|grozav|dezamăgitor|groaznic)\b',
        r'\bnu a fost\b.*?\b(prea|chiar|cel mai|atât de|foarte)?\b.*?\b(bun|rau|rău|grozav|perfect|teribil)\b',
        r'\b(a fost)?\b.*?\b(acceptable|decent enough|moderate|just fine|okay|kind of okay)\b',
        r'\b(nu prea\b|nu chiar\b)?\b.*?\b(impressive|disappointing|okay|fine|decent|acceptable)\b'
    ]
    
    
    # Dacă se detectează o expresie neutră, setăm o flag de „ambiguu”
    for pattern in neutral_patterns:
        if re.search(pattern, review_text, re.IGNORECASE):
            return "ambiguous"
    
    # Dacă nu se găsește niciun tipar, returnăm textul inițial
    return review_text


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


def analyze_sentiment_and_grade(review_text):
    # Aplică pre-procesarea pentru expresii de negație
    processed_text = preprocess_review_text(review_text)
    
    # Dacă este un text ambiguu, setăm un sentiment „positive” moderat
    if processed_text == "ambiguous":
        sentiment = "positive"
        grade = 5  # Notează ambiguitatea cu un scor moderat
        return sentiment, grade
    
    # Detectează limba și traduce în engleză dacă e necesar
    if detect(review_text) == 'ro':
        processed_text = translate_to_english(processed_text)

    # Procesăm textul cu modelul pentru scorul real
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)[0][0]

    # Mapare a predicției în notă între 1 și 10
    if prediction >= 0.8:
        sentiment = "positive"
        grade = int(5 + 5 * prediction)  # Distribuie între 6 și 10 pentru note mari
    elif prediction >= 0.5:
        sentiment = "positive"
        grade = int(3 + 5 * prediction)  # Distribuie între 5 și 8
    elif prediction >= 0.2:
        sentiment = "negative"
        grade = int(2 + 4 * prediction)  # Distribuie între 3 și 6
    else:
        sentiment = "negative"
        grade = max(1, int(2 + 3 * prediction))  # Distribuie între 1 și 3
    
    return sentiment, grade

def hybrid_svd(user_id, movie_id, top_n=10):
    """
    1) Ia top N candidate content-based pentru 'movie_id'
    2) Reordonează după rating prezis SVD pentru user_id
    3) Returnează top_n final
    """

    content_candidates = get_content_based_recommendations(movie_id, top_n=25)

    if not user_id:
        # Dacă nu e user logat, returnezi direct content-based
        return content_candidates[:top_n]

    # -- AICI adaugi verificarea dacă algo e None -- 
    if algo is None:
        # Nu există model SVD antrenat – fallback la content_candidates
        return content_candidates[:top_n]

    # Dacă algo nu e None, continuă cu reordonarea după rating SVD
    reweighted = []
    for cand_id in content_candidates:
        pred = algo.predict(user_id, cand_id).est
        reweighted.append((cand_id, pred))

    # 3) Sortezi descrescător după pred
    reweighted.sort(key=lambda x: x[1], reverse=True)

    # 4) Returnezi doar ID-urile top_n
    final_ids = [x[0] for x in reweighted[:top_n]]
    return final_ids



@app.route('/update_nn/<int:movie_id>', methods=['GET'])
@login_required
def update_nn(movie_id):
    # Rulează funcția de antrenare și salvează modelul
   
    train_and_save_model(db_path="tmdb_movies_for_nn.db", save_path="nn_model_tf")
    global nn_model, user_map, sae_model, ratings_tensor,gnn_model, gnn_user_map, gnn_movie_map, gnn_genre_map
    # Încarcă modelul actualizat din folder
    nn_model, user_map, movie_map = load_nn_model("nn_model_tf")
    flash("Modelul neuronal a fost actualizat cu succes!", "success")
    train_and_save_gmf_mlp(db_path="tmdb_movies_for_nn.db", save_path="gmf_mlp_model_tf")
    
    # 2) Încarcă modelul actualizat
    load_gmf_mlp_model("gmf_mlp_model_tf")
    
    # 3) Mesaj către utilizator
    flash("Modelul GMF+MLP a fost actualizat cu succes!", "success")
    
    sae_model, ratings_tensor = train_autoencoder_model(nb_epoch=4)
    train_gnn_model(db_path="tmdb_movies_for_nn.db", model_path="gnn_model.pt", epochs=1)
    gnn_model, gnn_user_map, gnn_movie_map, gnn_genre_map = load_gnn_model("gnn_model.pt")
    
    
    return redirect(url_for('movie_detail', movie_id=movie_id))

# Inițializare globală BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def extract_text_embedding_bert(text):
    """
    Returnează embedding-ul BERT (vector numeric) pentru textul dat.
    Folosim [CLS] ca vector de reprezentare a propoziției/descrierii.
    """
    # Convertim textul în tokeni BERT
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Extragem embedding-ul de pe primul token ([CLS])
    return outputs.last_hidden_state[:, 0, :].numpy()  # shape: (1, 768) în mod normal

bert_embeddings_dict = {}  # { movie_id: np.array([...]) }

BERT_EMB_PATH = 'bert_embeddings.pkl'

def build_bert_embeddings(save_to_disk: bool = True):
    """
    Construiește un embedding BERT pentru fiecare film din BD
    şi, dacă save_to_disk=True, salvează rezultatul în BERT_EMB_PATH.
    """
    global bert_embeddings_dict
    conn = sqlite3.connect('tmdb_movies.db')
    df = pd.read_sql_query("SELECT id, overview, genres FROM movies", conn)
    conn.close()

    bert_embeddings_dict = {}
    for _, row in df.iterrows():
        mid = row['id']
        text = (row['overview'] or "") + " " + (row['genres'] or "")
        emb = extract_text_embedding_bert(text)
        bert_embeddings_dict[mid] = emb

    if save_to_disk:
        with open(BERT_EMB_PATH, 'wb') as f:
            pickle.dump(bert_embeddings_dict, f)
        print(f"[BERT] Salvat embeddings în '{BERT_EMB_PATH}'")

def load_bert_embeddings():
    """
    Încarcă dict-ul de embeddings din fişier. 
    Dacă fişierul nu există, apelează build_bert_embeddings().
    """
    global bert_embeddings_dict
    if os.path.exists(BERT_EMB_PATH):
        with open(BERT_EMB_PATH, 'rb') as f:
            bert_embeddings_dict = pickle.load(f)
        print(f"[BERT] Încărcat {len(bert_embeddings_dict)} embeddings din '{BERT_EMB_PATH}'")
    else:
        print(f"[BERT] Fișier '{BERT_EMB_PATH}' nu există, construiesc embeddings…")
        build_bert_embeddings(save_to_disk=True)



load_bert_embeddings()
def get_bert_recommendations(movie_id, top_n=5):
    """
    Returnează top_n filme (movie_id) similare cu filmul dat, pe baza
    embedding-urilor BERT (similaritate cosinus).
    """
    if movie_id not in bert_embeddings_dict:
        return []

    current_emb = bert_embeddings_dict[movie_id]  # shape (1, 768)
    scores = []
    for mid, emb in bert_embeddings_dict.items():
        if mid == movie_id:
            continue
        # cos_sim este matrice 1x1 dacă shape(emb) = (1, 768)
        sim = cosine_similarity(current_emb, emb)[0][0]
        scores.append((mid, sim))

    # Sortăm descrescător după similaritate
    scores.sort(key=lambda x: x[1], reverse=True)

    # Returnăm doar ID-urile de film
    top_ids = [item[0] for item in scores[:top_n]]
    return top_ids


# def download_posters(csv_path, out_folder="posters"):
#     df = pd.read_csv(csv_path, encoding='latin1')
#     if not os.path.exists(out_folder):
#         os.makedirs(out_folder)

#     for idx, row in df.iterrows():
#         poster_url = row['Poster']  # link la imagine
#         genre = row['Genre']        # genul filmului
#         imdb_id = row['imdbId']     # id

#         # Ia doar primul gen, în caz că sunt mai multe despărțite de '|'
#         if '|' in genre:
#             genre = genre.split('|')[0]

#         # Numele fișierului -> "imdbId_genre.jpg"
#         local_filename = f"{imdb_id}_{genre}.jpg"
#         local_path = os.path.join(out_folder, local_filename)
        
#         # Descarcă imaginea
#         try:
#             r = requests.get(poster_url, timeout=5)
#             if r.status_code == 200:
#                 with open(local_path, 'wb') as f:
#                     f.write(r.content)
#                 print(f"Downloaded {local_filename}")
#             else:
#                 print(f"[!] Eroare la {poster_url} (status={r.status_code})")
#         except Exception as e:
#             print(f"[!] Exceptie la {poster_url}: {e}")



# def create_training_csv(csv_path="MovieGenre.csv", out_path="train.csv", posters_folder="posters"):
#     # Read CSV with encoding and handle missing 'Genre' values
#     df = pd.read_csv(csv_path, encoding='latin1').fillna({'Genre': ''})  # Replace NaN with empty string
    
#     rows = []
#     for idx, row in df.iterrows():
#         imdb_id = row['imdbId']
#         genre = row['Genre']
        
#         # Ensure genre is a string (even if empty)
#         if pd.isna(genre):  # Extra safety check (optional)
#             genre = ''
#         else:
#             genre = str(genre)
        
#         # Split genre if needed
#         if '|' in genre:
#             genre = genre.split('|')[0]
        
#         # Build filename and check if file exists
#         filename = f"{imdb_id}_{genre}.jpg"
#         local_path = os.path.join(posters_folder, filename)
        
#         if os.path.exists(local_path):
#             rows.append({'filename': filename, 'class_label': genre})
    
#     # Save the output CSV
#     df_out = pd.DataFrame(rows)
#     df_out.to_csv(out_path, index=False)
#     print(f"Salvat {out_path} cu {len(df_out)} intrari.")


# def train_poster_model(train_csv="train.csv", poster_folder="posters", model_path="poster_classifier.h5"):
#     df = pd.read_csv(train_csv)
#     df['class_label'] = df['class_label'].astype(str)  # asigurare tip string
    
#     # Toate etichetele distincte, sortate
#     all_classes = sorted(df['class_label'].unique())
#     from sklearn.model_selection import train_test_split
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#     train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#     val_datagen   = ImageDataGenerator(preprocessing_function=preprocess_input)

#     train_generator = train_datagen.flow_from_dataframe(
#         dataframe=train_df,
#         directory=poster_folder,
#         x_col='filename',
#         y_col='class_label',
#         classes=all_classes,            # << adăugăm classes
#         target_size=(224,224),
#         class_mode='categorical',
#         batch_size=32
#     )

#     val_generator = val_datagen.flow_from_dataframe(
#         dataframe=val_df,
#         directory=poster_folder,
#         x_col='filename',
#         y_col='class_label',
#         classes=all_classes,            # << adăugăm classes și la val
#         target_size=(224,224),
#         class_mode='categorical',
#         batch_size=32
#     )

#     # 3) Baza VGG16
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
#     for layer in base_model.layers:
#         layer.trainable = False

#     x = base_model.output
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)

#     # Folosim tot atatea neuroni cat numarul total de clase
#     num_classes = len(all_classes)
#     predictions = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     model.fit(
#         train_generator,
#         epochs=5,
#         validation_data=val_generator
#     )

#     # Salvezi
#     model.save(model_path)
#     print(f"Model salvat in {model_path}")
#     # Salvezi si clasa
#     class_indices = train_generator.class_indices
#     inv_map = {v:k for k,v in class_indices.items()}
#     np.save("class_indices.npy", inv_map)
#     print("Salvat class_indices.npy")


# poster_classifier = tf.keras.models.load_model("poster_classifier.h5")
# class_indices = np.load("class_indices.npy", allow_pickle=True).item()  # dict int->gen



def create_multi_label_csv(
    csv_path='MovieGenre.csv',
    posters_folder='posters',
    out_path='train_multi.csv'
):
    """
    1. Citește CSV-ul original (MovieGenre.csv).
    2. Păstrează DOAR genurile ['Action','Comedy','Drama','Horror','Romance','Sci-Fi'].
    3. Creează un vector 0/1 (lungime 6) pentru fiecare poster (multi-hot).
    4. Salvează un nou CSV (train_multi.csv) și un fișier 'all_genres.npy' cu cele 6 genuri.
    """
    import os
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path, encoding='latin1').fillna({'Genre': ''})

    # Limităm la genurile dorite
    wanted_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]

    def encode_genres(genres_str):
        """
        Construiește vector 0/1 de lungime 6 (ordinea din wanted_genres).
        Dacă filmul nu are deloc un gen din wanted_genres, returnează None (ignorăm filmul).
        """
        splitted = [x.strip() for x in genres_str.split('|') if x.strip()]
        vec = [0]*6
        found_any = False
        for g in splitted:
            if g in wanted_genres:
                idx = wanted_genres.index(g)
                vec[idx] = 1
                found_any = True
        return vec if found_any else None

    rows = []
    for _, row in df.iterrows():
        imdb_id = row.get('imdbId', '')
        genres_str = row.get('Genre', '')

        # Fișier poster, convenție "imdbId.jpg"
        filename = f"{imdb_id}.jpg"
        local_path = os.path.join(posters_folder, filename)

        # Verificăm dacă fișierul există
        if os.path.exists(local_path):
            label_vec = encode_genres(genres_str)
            if label_vec is not None:
                rows.append({
                    'filename': filename,
                    'labels': label_vec
                })

    # Creăm DataFrame cu rândurile rămase
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"Salvat {out_path} cu {len(df_out)} intrări.")

    # Salvăm lista de genuri (fix 6)
    np.save('all_genres.npy', wanted_genres)
    print("Salvat all_genres.npy (lista de genuri).")




def train_multi_label(
    csv_path='train_multi.csv',
    posters_folder='posters',
    model_path='poster_multi.h5',
    epochs=5
):
    """
    Antrenează un model CNN multi-label (VGG16) pe imaginile din 'poster_folder'.
    CSV-ul conține coloane: [filename, labels].
    'labels' este un vector 0/1 sub formă de string ex: "[1, 0, 0, 1, ...]".
    """

    # 1) Încărcăm lista de genuri
    all_genres = np.load('all_genres.npy', allow_pickle=True)
    num_genres = len(all_genres)
    print(f"Număr total de genuri: {num_genres}")

    # 2) Citim fișierul CSV
    df = pd.read_csv(csv_path)

    # Funcție pentru a converti stringul "[0,1,0...]" la array python
    def parse_labels(label_str):
        # Eliminăm parantezele drepte și despărțim
        raw = label_str.strip("[] ")
        if not raw:
            return np.zeros(num_genres, dtype='float32')
        arr = [int(x) for x in raw.split(',')]
        return np.array(arr, dtype='float32')

    df['label_vec'] = df['labels'].apply(parse_labels)

    # 3) Împărțim train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}")

    # 4) Încărcăm imaginile efectiv
    def load_images_and_labels(dataframe):
        images = []
        labels = []
        for _, row in dataframe.iterrows():
            fname = row['filename']
            vec = row['label_vec']
            path = os.path.join(posters_folder, fname)
            if os.path.exists(path):
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert('RGB')
                    pil_img = pil_img.resize((224,224))
                    img_arr = keras_image.img_to_array(pil_img)
                    img_arr = preprocess_input(img_arr)
                images.append(img_arr)
                labels.append(vec)
        return np.array(images), np.array(labels)

    X_train, y_train = load_images_and_labels(train_df)
    X_val,   y_val   = load_images_and_labels(val_df)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)

    # 5) Definim modelul (transfer learning cu VGG16)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Strat final -> 'sigmoid' pentru multi-label
    predictions = Dense(num_genres, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 6) Antrenăm
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=epochs
    )

    # 7) Salvăm modelul și genurile
    model.save(model_path)
    print(f"Modelul multi-label salvat în {model_path}")
    return model, all_genres



def predict_multilabel(model, all_genres, img_path, threshold=0.5):
    pil_img = Image.open(img_path).convert('RGB')
    pil_img = pil_img.resize((224,224))
    img_arr = keras_image.img_to_array(pil_img)
    img_arr = preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)[0]  # vector (num_genres,)

    results = []
    for genre, score in zip(all_genres, preds):
        if score >= threshold:
            results.append((genre, float(score)))
    # Ordonăm descrescător după scor
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# Încarcă modelul deja antrenat și lista de genuri
model_multi = tf.keras.models.load_model('poster_multi.h5')
all_genres = np.load('all_genres.npy', allow_pickle=True)

import urllib.parse


import json
CONTENT_PROFILE_PATH = 'content_profiles.json'
content_profiles = {}

def load_all_content_profiles():
    global content_profiles
    if not os.path.exists(CONTENT_PROFILE_PATH):
        raise RuntimeError(f"{CONTENT_PROFILE_PATH} not found! Run build_content_profiles.py first.")
    with open(CONTENT_PROFILE_PATH, 'r', encoding='utf-8') as f:
        content_profiles = json.load(f)

# load into memory immediately
load_all_content_profiles()

def build_content_profile(movie_id, title=None, genres=None):
    """
    Returns (profile_string, actors_string) from the precomputed JSON.
    If missing, returns two empty strings.
    """
    entry = content_profiles.get(str(movie_id)) or content_profiles.get(movie_id)
    if entry:
        return entry['profile'], entry['actors']
    return "", ""


def get_content_recs_by_profile(current_movie_id, top_n=10):
    # build list of all IDs and their precomputed profiles
    ids = list(content_profiles.keys())
    profiles = [content_profiles[mid]['profile'] for mid in ids]

    # vectorize & similarity
    cv = CountVectorizer(stop_words='english', max_features=5000)
    mat = cv.fit_transform(profiles)
    sim = cosine_similarity(mat)

    try:
        idx = ids.index(str(current_movie_id))
    except ValueError:
        return [], ""

    scores = list(enumerate(sim[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [int(ids[i]) for i, _ in scores[1:top_n+1]]

    # also return the actors string for display
    actors = content_profiles[str(current_movie_id)]['actors']
    return top, actors

@app.route('/voice_recommend', methods=['POST'])
@login_required
def voice_recommend():
    audio = request.files.get('audio_data')
    if not audio:
        return jsonify(transcript='', recommendations=[])

    # 1) Salvează webm într-un temp file
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_webm:
        audio.save(tmp_webm.name)
        webm_path = tmp_webm.name

    # 2) Converteşte în WAV
    wav_path = webm_path.replace('.webm', '.wav')
    try:
        AudioSegment.from_file(webm_path, format='webm')\
                    .export(wav_path, format='wav')
    except Exception as e:
        os.remove(webm_path)
        return jsonify(transcript='', recommendations=[]), 400

    # 3) Foloseşte SpeechRecognition pe WAV
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(
                audio_data, language='ro-RO')
    except (sr.UnknownValueError, sr.RequestError):
        transcript = ''
    finally:
        # curăţenie
        os.remove(webm_path)
        os.remove(wav_path)

    # 4) Recomandări identic cu înainte
    conn = sqlite3.connect('tmdb_movies.db')
    df = pd.read_sql_query(
        "SELECT id,title,poster_path,overview,genres FROM movies", conn)
    conn.close()
    df['tags'] = df['title'].fillna('')+df['title'].fillna('')+df['overview'].fillna('') + ' ' + df['genres'].fillna('')

    corpus = df['tags'].tolist() + [transcript]
    vect = CountVectorizer(stop_words='english', max_features=5000)
    mat = vect.fit_transform(corpus)
    sim = cosine_similarity(mat[-1], mat[:-1])[0]

    top_idx = sim.argsort()[::-1][:10]
    recs = df.iloc[top_idx][['id','title','poster_path']]\
             .to_dict(orient='records')

    return jsonify(transcript=transcript, recommendations=recs)

# Path-uri
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
GPRE_JSON_PATH = os.path.join(MODEL_DIR, 'gpre.json')
GPRE_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'gpre.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'label.json')

# Încarcă o singură dată modelul GPRE
with open(GPRE_JSON_PATH, 'r', encoding='utf-8') as f:
    gpre_model = model_from_json(f.read())
gpre_model.load_weights(GPRE_WEIGHTS_PATH)

# Încarcă maparea id → genre
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    id2genre = json.load(f)['id2genre']

def preprocess_gpre(img: np.ndarray, size=(150,100)) -> np.ndarray:
    """
    Resize + normalize conform scriptului GitHub.
    img: H×W×3, RGB uint8
    retur: 1×150×100×3 float32
    """
    img_resized = skimage.transform.resize(img, size)
    img_resized = img_resized.astype(np.float32)
    return np.expand_dims(img_resized, axis=0)

# 1) Funcție de extracție titlu
def extract_title_region(img_np, expand=40, thresh_ratio=0.2):
    """
    img_np: H×W×3 RGB numpy array
    expand: câte pixeli să extindă sus/jos față de linia de gradient maxim
    thresh_ratio: pragul relativ la gradient_max pentru oprire
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_strength = np.sum(np.abs(sobel), axis=1)
    max_i = int(np.argmax(row_strength))
    max_val = row_strength[max_i]
    # prag absolut
    thresh = max_val * thresh_ratio

    # caută sus
    y1 = max_i
    while y1 > 0 and row_strength[y1] >= thresh:
        y1 -= 1
    # caută jos
    y2 = max_i
    while y2 < len(row_strength)-1 and row_strength[y2] >= thresh:
        y2 += 1

    # extind puțin context
    y1 = max(0, y1 - expand)
    y2 = min(img_np.shape[0], y2 + expand)
    return img_np[y1:y2, :]

# 2) Funcție OCR cu configurare
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # ajustează după instalare
OCR_CONFIG = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

def ocr_title(crop_np):
    text = image_to_string(crop_np, config=OCR_CONFIG)
    # curăță spații multiple și linii goale
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return ' '.join(lines)


@app.route('/poster_recommend', methods=['POST'])
@login_required
def poster_recommend():
    # 1) get file
    file = request.files.get('poster_image')
    if not file or file.filename == '':
        return jsonify(error="no file"), 400

    # 2) OCR + fallback
    from PIL import Image
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    try:
        title_crop = extract_title_region(img_np)
        detected_title = ocr_title(title_crop)
    except:
        detected_title = ''
    # load the DB title if OCR fails
    conn = sqlite3.connect('tmdb_movies.db')
    cur = conn.cursor()
    cur.execute("SELECT title FROM movies WHERE id=?", (movie_id:=request.form.get('movie_id'),))
    row = cur.fetchone()
    conn.close()
    if not detected_title or len(detected_title) < 3:
        detected_title = row[0]

    # 3) GPRE prediction
    batch = preprocess_gpre(img_np)
    preds = gpre_model.predict(batch)[0]
    top5 = preds.argsort()[::-1]
    # filter out Drama
    genres = [ id2genre[i] for i in top5 if id2genre[i] != 'Drama' ][:4]

    # 4) Build the “query” string
    query = detected_title + " " + " ".join(genres)

    # 5) Load all movies, vectorize tags = overview + genres
    conn = sqlite3.connect('tmdb_movies.db')
    import pandas as pd
    df = pd.read_sql_query(
        "SELECT id,title,poster_path,overview,genres FROM movies", conn)
    conn.close()
    df['tags'] =df['overview'].fillna('')+df['title'].fillna('')+df['title'].fillna('')+df['title'].fillna('')+df['title'].fillna('') + " " + df['genres'].fillna('')

    corpus = df['tags'].tolist() + [query]
    vect = CountVectorizer(stop_words='english', max_features=5000)
    mat = vect.fit_transform(corpus)
    sim = cosine_similarity(mat[-1], mat[:-1])[0]

    # top 10
    idxs = sim.argsort()[::-1][:10]
    recs = df.iloc[idxs][['id','title','poster_path']].to_dict(orient='records')

    return jsonify(
      title=detected_title,
      genres=genres,
      recommendations=recs
    )

# ======================  AJAX Rating  ======================
@app.route('/api/rate', methods=['POST'])
@login_required
def api_rate():
    data       = request.get_json(force=True)
    movie_id   = int(data.get('movie_id', 0))
    rating_int = int(data.get('rating', 0))
    user_id    = session.get('user_id')

    if rating_int not in range(1, 6):
        return jsonify(error='Rating invalid'), 400

    conn  = sqlite3.connect('tmdb_movies.db')
    c1    = conn.cursor()
    conn2 = sqlite3.connect('tmdb_movies_for_nn.db')
    c2    = conn2.cursor()

    try:
        c1.execute("""INSERT INTO ratings (user_id, movie_id, rating)
                      VALUES (?,?,?)
                      ON CONFLICT(user_id,movie_id)
                      DO UPDATE SET rating=excluded.rating""",
                   (user_id, movie_id, rating_int))
        conn.commit()

        c2.execute("""INSERT INTO ratings (user_id, movie_id, rating)
                      VALUES (?,?,?)
                      ON CONFLICT(user_id,movie_id)
                      DO UPDATE SET rating=excluded.rating""",
                   (user_id, movie_id, rating_int))
        conn2.commit()

        # rebuild‑uri exact ca înainte
        initialize_collaborative_filtering()
        initialize_collaborative_filtering_clicks()
        initialize_knn_model(db_path="tmdb_movies.db")
        initialize_user_based_matrices()

        return jsonify(msg="OK")
    except Exception as e:
        print("[Rating Error]", e)
        return jsonify(error='DB error'), 500
    finally:
        conn.close()
        conn2.close()

# ======================  AJAX Review  ======================
@app.route('/api/review', methods=['POST'])
@login_required
def api_review():
    print("👉 api_review a fost apelată, JSON primit:", request.get_json(), "user_id:", session.get('user_id'))
    data      = request.get_json(force=True)
    movie_id  = int(data.get('movie_id', 0))
    review_tx = (data.get('review') or '').strip()
    user_id   = session.get('user_id')

    if len(review_tx) < 4:
        return jsonify(error="Review prea scurt"), 400

    sentiment, grade = analyze_sentiment_and_grade(review_tx)

    conn = sqlite3.connect('tmdb_movies.db')
    cur  = conn.cursor()
    try:
        cur.execute("""INSERT INTO reviews
                       (user_id, movie_id, review, sentiment, grade)
                       VALUES (?,?,?,?,?)""",
                    (user_id, movie_id, review_tx, sentiment, grade))
        conn.commit()
        return jsonify(msg="OK",
                       sentiment=sentiment,
                       grade=grade,
                       user=session.get('user_name'))
    except Exception as e:
        print("[Review Error]", e)
        return jsonify(error="DB error"), 500
    finally:
        conn.close()

def _movie_index(movie_id: int) -> int:
    """poziţia filmului în DataFrame / matricele de similaritate"""
    return movies_df.index[movies_df['id'] == movie_id][0]

def _weighted_avg(ratings, sims, eps=1e-8):
    return float(np.dot(ratings, sims) / (sims.sum() + eps))

def user_rated_movies(uid):
    """ (movie_id, rating) doar pentru user-ul uid """
    with sqlite3.connect('tmdb_movies_for_nn.db') as con:
        cur = con.cursor()
        cur.execute("""
            SELECT movie_id, rating
              FROM ratings
             WHERE user_id = ?
        """, (uid,))
        return cur.fetchall()   # listă de (id, rating)

def _content_prediction(uid: int,
                        target_mid: int,
                        sim_matrix: np.ndarray) -> float:
    """scor bazat pe similaritate + ratingurile user‑ului"""
    seen = user_rated_movies(uid)            # [(mid, r), …]
    if not seen:
        return 3.0                           # fallback

    tgt_idx = _movie_index(target_mid)
    idxs    = [ _movie_index(m) for m, _ in seen ]
    sims    = sim_matrix[tgt_idx, idxs]
    ratings = np.array([r for _, r in seen], dtype=float)
    return np.clip(_weighted_avg(ratings, sims), 1, 5)


# ———————————  PREDICTORS  ———————————
def predict_content(uid: int, mid: int)            -> float: return _content_prediction(uid, mid, similarity)


def compute_rmse_for_algorithms(uid: int):
    # ------ Content‑based: DOAR filmele user‑ului ------
    rated = user_rated_movies(uid)          # [(mid,r),…]
    if rated:
        y_true_u = [r for _, r in rated]
        rmse_content = rmse(y_true_u, [predict_content(uid, m)           for m,_ in rated])        
    else:
        rmse_content = rmse_adv = rmse_bert = float('nan')

  

    return {
        "content_cb"  : rmse_content,
   
      
        "mean_rmse"   : np.nanmean([rmse_content,
                                    ])
    }


@app.route('/movie/<int:movie_id>', methods=['GET', 'POST'])
@login_required
def movie_detail(movie_id):
    global movies_df, similarity
    
    conn = sqlite3.connect('tmdb_movies.db')
    cursor = conn.cursor()
    conn_for_nn = sqlite3.connect('tmdb_movies_for_nn.db')
    cursor_for_nn = conn_for_nn.cursor()
    user_id = session.get('user_id')
    user_name = session.get('user_name')
    record_click(user_id, movie_id)
    initialize_user_based_matrices()
    load_users() 
    hybrid_recommendations = []
    if user_id:
        hybrid_recommendations = get_hybrid_recommendations_clicks(user_id, movie_id, top_n=10)
    
    
    # Handle the GET request to display movie details and reviews
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
    
    
    # if request.method == 'POST':
    #     # Handle rating submission
    #        # 1) Verificare upload poster
        
    #     # 2) Verificare rating
    #     if 'rating' in request.form and request.form['rating']:
    #         rating_value = request.form.get('rating')
    #         if rating_value and rating_value.isdigit():
    #             rating_int = int(rating_value)
    #             if 1 <= rating_int <= 5:
    #                 try:
    #                     cursor.execute('''
    #                         INSERT INTO ratings (user_id, movie_id, rating)
    #                         VALUES (?, ?, ?)
    #                         ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating
    #                     ''', (user_id, movie_id, rating_int))
    #                     conn.commit()
    #                     cursor_for_nn.execute('''
    #                         INSERT INTO ratings (user_id, movie_id, rating)
    #                         VALUES (?, ?, ?)
    #                         ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating
    #                     ''', (user_id, movie_id, rating_int))
    #                     conn_for_nn.commit()
    #                     # === rebuild pentru modelele non-neurale ===
    #                     initialize_collaborative_filtering()
    #                     initialize_collaborative_filtering_clicks()
    #                     initialize_knn_model(db_path="tmdb_movies.db") 
    #                     initialize_user_based_matrices()
    #                     # ============================================
    #                     flash('Rating-ul tău a fost trimis cu succes!', 'success')
    #                 except Exception as e:
    #                     flash('A apărut o eroare la trimiterea rating-ului.', 'danger')
    #                     print(f"[Rating Error] {e}")
    #             else:
    #                 flash('Valoare invalidă pentru rating. Trebuie între 1 și 5.', 'danger')

    #     # 3) Verificare review
    #     elif 'review' in request.form and request.form['review']:
    #         review_text = request.form['review']
    #         if review_text.strip():
    #             try:
    #                 sentiment, grade = analyze_sentiment_and_grade(review_text)
    #                 cursor.execute('''
    #                     INSERT INTO reviews (user_id, movie_id, review, sentiment, grade)
    #                     VALUES (?, ?, ?, ?, ?)
    #                 ''', (user_id, movie_id, review_text, sentiment, grade))
    #                 conn.commit()
    #                 flash('Review-ul tău a fost trimis și analizat.', 'success')
    #             except Exception as e:
    #                 flash('A apărut o eroare la trimiterea review-ului.', 'danger')
    #                 print(f"[Review Error] {e}")

    #     # După ce am tratat toate cazurile, dăm un singur redirect
    #     conn.close()
    #     return redirect(url_for('movie_detail', movie_id=movie_id))
        


    

    # Get content-based recommendations
    if movies_df is None or similarity is None:
        compute_similarity()

  
    initialize_collaborative_filtering()
    content_based_movie_ids = get_content_based_recommendations(movie_id, top_n=10)
    hybrid_recommended_movie_ids = get_hybrid_recommendations(user_id, movie_id, top_n=10)

    content_based_recommended_movies = []
    if content_based_movie_ids:
        placeholders = ','.join('?' for _ in content_based_movie_ids)
        cursor.execute(f'''
            SELECT id, title, poster_path
            FROM movies
            WHERE id IN ({placeholders})
        ''', content_based_movie_ids)
        content_based_recommended_movies = cursor.fetchall()

    hybrid_recommended_movies = []
    if hybrid_recommended_movie_ids:
        placeholders = ','.join('?' for _ in hybrid_recommended_movie_ids)
        cursor.execute(f'''
            SELECT id, title, poster_path
            FROM movies
            WHERE id IN ({placeholders})
        ''', hybrid_recommended_movie_ids)
        hybrid_recommended_movies = cursor.fetchall()

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

    if user_id:
    # Aici e noua funcție
        hybrid_svd_ids = hybrid_svd(user_id, movie_id, top_n=10)
    else:
    # fallback daca userul nu e logat
        hybrid_svd_ids = get_content_based_recommendations(movie_id, top_n=10)

    # Apoi faci query in DB pentru postere etc.
    hybrid_svd_recommended_movies = []
    if hybrid_svd_ids:
        placeholders = ','.join('?' for _ in hybrid_svd_ids)
        cursor.execute(f'''
         SELECT id, title, poster_path
         FROM movies
         WHERE id IN ({placeholders})
            ''', hybrid_svd_ids)
        hybrid_svd_recommended_movies = cursor.fetchall()

    
    # Fetch all reviews for this movie, including sentiment and grade
    cursor.execute('''
        SELECT reviews.review, reviews.sentiment, reviews.grade, users.name
        FROM reviews
        JOIN users ON reviews.user_id = users.id
        WHERE reviews.movie_id = ?
    ''', (movie_id,))
    reviews = cursor.fetchall()  # List of reviews with sentiment and grades

        # *** NEW: Advanced Content-Based Recommendations *** #
    advanced_content_ids,actors = get_content_recs_by_profile(movie_id, top_n=7)
    advanced_content_movies = []
    if advanced_content_ids:
        placeholders = ','.join('?' for _ in advanced_content_ids)
        cursor.execute(f'''
            SELECT id, title, poster_path
            FROM movies
            WHERE id IN ({placeholders})
        ''', advanced_content_ids)
        advanced_content_movies = cursor.fetchall()
    

    # print(f"ACTORI: {actors}")
     # Recomandări NMF: folosim filmul curent ca referință
        # Recomandări BERT
    bert_recommended_ids = get_bert_recommendations(movie_id, top_n=5)
    bert_recommended_movies = []
    if bert_recommended_ids:
        placeholders = ','.join('?' for _ in bert_recommended_ids)
        cursor.execute(f'''
            SELECT id, title, poster_path
            FROM movies
            WHERE id IN ({placeholders})
        ''', bert_recommended_ids)
        bert_recommended_movies = cursor.fetchall()


        

    conn.close()
    
     # Înainte de render_template, definește link-urile:
    title = movie[1]  # titlul filmului
    q = urllib.parse.quote_plus(title)
    watch_links = {
        "Netflix":            f"https://www.netflix.com/search?q={q}",
        "Prime Video":        f"https://www.primevideo.com/search/ref=atv_sr_s_single?phrase={q}",
        "HBO Max":            f"https://www.hbomax.com/search?query={q}",
        "Disney Plus":        f"https://www.disneyplus.com/search/{q}",
        "Hulu":               f"https://www.hulu.com/search?q={q}",
        "Google Play Movies": f"https://play.google.com/store/search?q={q}&c=movies",
        # Platforme GRATUITE / public domain
        "YouTube Free":         f"https://www.youtube.com/results?search_query={q}+full+movie",
        "Archive.org":          f"https://archive.org/search.php?query={q}",
        "Tubi TV":              f"https://tubitv.com/search/{q}",
        "Pluto TV":             f"https://pluto.tv/search/{q}",
        "Crackle":              f"https://www.crackle.com/search?query={q}",
        "Popcornflix":          f"https://www.popcornflix.com/search?keyword={q}",
        "Vudu (Free)":          f"https://www.vudu.com/content/movies/search?search={q}&free=true",
        "Peacock (Free Tier)":  f"https://www.peacocktv.com/search?q={q}&contentType=movie",
        "Apple iTunes":       f"https://itunes.apple.com/search?term={q}&media=movie"
    }
        
    

                          
    
    return render_template('movie_detail.html',
                           hybrid_recommendations=hybrid_recommendations,
                           movie=movie,
                           user_name=user_name,                          
                           advanced_content_movies=advanced_content_movies,
                           hybrid_svd_recommended_movies=hybrid_svd_recommended_movies,
                           avg_rating=avg_rating,
                           actors=actors,
                            bert_recommended_movies=bert_recommended_movies,
                           user_rating=user_rating,
                           trailer_url=trailer_url,
                           content_based_recommended_movies=content_based_recommended_movies,
                           hybrid_recommended_movies=hybrid_recommended_movies,
                           watch_links=watch_links,
                           reviews=reviews)  # Pass reviews to the template



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

@app.errorhandler(429)
def ratelimit_error(e):
    # Informații despre atacator
    attacker_ip = request.remote_addr  # IP-ul utilizatorului
    user_agent = request.headers.get('User-Agent')  # Informații despre browser/dispozitiv
    endpoint = request.path  # Endpoint-ul accesat
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Timestamp-ul cererii

    # Logare informații despre atacator
    logging.info(f"Rate Limit Exceeded: IP={attacker_ip}, User-Agent={user_agent}, Endpoint={endpoint}, Time={timestamp}")

    # Răspuns atractiv pentru utilizator
    return jsonify({
        "error": "Ai depășit limita de cereri. Încearcă din nou mai târziu.",
        "details": {
            "IP": attacker_ip,
            "ruta_accesată": endpoint,
            "sfat": "Te rugăm să faci mai puține cereri pentru a evita blocarea completă."
        },
        "time_to_wait": "60 secunde"
    }), 429


# def prepare_poster_model(csv_path='MovieGenre.csv'):
#     """
#     1) Descarcă toate imaginile de poster în folderul 'posters/'.
#     2) Creează 'train.csv' pe baza posterelor existente.
#     3) Antrenează modelul VGG16 + top layers => 'poster_classifier.h5'.
#     """

#     # 1) Descarcă posterele
#     #download_posters(csv_path=csv_path, out_folder="posters")

#     # 2) Creează train.csv
#     create_training_csv(csv_path=csv_path, out_path="train.csv", posters_folder="posters")

#     # 3) Antrenează modelul
#     train_poster_model(train_csv="train.csv", poster_folder="posters", model_path="poster_classifier.h5")

def prepare_multi_label_model(csv_path="MovieGenre.csv"):
    """
    1) Creează train_multi.csv cu vectori multi-label.
    2) Antrenează modelul multi-label => poster_multi.h5.
    """
    create_multi_label_csv(
        csv_path=csv_path,
        posters_folder="posters",
        out_path="train_multi.csv"
    )
    train_multi_label(
        csv_path="train_multi.csv",
        posters_folder="posters",
        model_path="poster_multi.h5",
        epochs=5
    )

# Funcția RMSE standard
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Funcție pentru a obține DataFrame-ul ratingurilor din baza de date
def get_ratings_dataframe():
    conn = sqlite3.connect('tmdb_movies.db')
    df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()
    return df

def get_ratings_dataframe_nn():
    conn = sqlite3.connect('tmdb_movies_for_nn.db')
    df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()
    return df

#########################
# Predictor functions:
#########################

def baseline_predictor(user_id, movie_id):
    return 3.0

def svd_predictor(user_id, movie_id):
    global algo, trainset
    if trainset is None or algo is None:
        return 3.0
    return algo.predict(user_id, movie_id).est

def weighted_mean_predictor(user_id, movie_id):
    # presupunem că r_matrix şi cosine_sim sunt deja iniţializate global
    if r_matrix is None or cosine_sim is None:
        return 3.0
    if user_id not in r_matrix.index:
        return 3.0

    # similaritate user-user
    sim_scores = cosine_sim.loc[user_id]
    # rating-urile existente
    m_ratings = r_matrix[movie_id].dropna() if movie_id in r_matrix.columns else pd.Series()
    common_users = sim_scores.index.intersection(m_ratings.index)
    if common_users.empty or sim_scores.loc[common_users].sum() == 0:
        return 3.0

    relevant_sims    = sim_scores.loc[common_users]
    relevant_ratings = m_ratings.loc[common_users]
    return float((relevant_sims * relevant_ratings).sum() / relevant_sims.sum())


def cf_gender_predictor(user_id, movie_id):
    return cf_user_wmean_gender(user_id, movie_id)

def cf_gen_occ_predictor(user_id, movie_id):
    return cf_user_wmean_gen_occ(user_id, movie_id)

def nn_predictor(user_id, movie_id):
    global nn_model, user_map, movie_map
    if nn_model is None or user_id not in user_map or movie_id not in movie_map:
        return 3.0
    u_idx = user_map[user_id]
    m_idx = movie_map[movie_id]
    # Convertim array-urile la np.int64 pentru a corespunde specificațiilor modelului
    u_array = np.array([u_idx], dtype=np.int64)
    m_array = np.array([m_idx], dtype=np.int64)
    pred = nn_model.predict([u_array, m_array])[0][0]
    rating = pred * 4.0 + 1.0
    return rating


# KNN, folosind knn_algo (antrenat cu KNNBasic)
def knn_predictor(user_id, movie_id):
    global knn_algo, trainset_knn
    if knn_algo is None or trainset_knn is None:
        return 3.0
    try:
        pred = knn_algo.predict(user_id, movie_id).est
        return pred
    except Exception as e:
        print(f"KNN predictor error: {e}")
        return 3.0

def mlp_gmf_predictor(user_id, movie_id):
    global gmf_mlp_model, user_map_gmfmlp, item_map_gmfmlp
    # Verificăm dacă modelul și mapping-urile sunt încărcate și dacă user_id și movie_id există în dicționare
    if gmf_mlp_model is None or user_id not in user_map_gmfmlp or movie_id not in item_map_gmfmlp:
        return 3.0
    # Obținem indicele pentru user și film (item)
    u_idx = user_map_gmfmlp[user_id]
    m_idx = item_map_gmfmlp[movie_id]
    # Convertim la array-uri cu tipul corect (np.int64) pentru a corespunde specificațiilor modelului încărcat
    u_array = np.array([u_idx], dtype=np.int64)
    m_array = np.array([m_idx], dtype=np.int64)
    # Prezicem valoarea folosind modelul GMF+MLP
    pred = gmf_mlp_model.predict([u_array, m_array])[0][0]
    # Presupunând că modelul generează o valoare normalizată în intervalul [0, 1],
    # scalăm predicția la intervalul [1, 5]:
    rating = pred * 4.0 + 1.0
    return rating


# Autoencoder, folosind modelul SAE
def autoencoder_predictor(user_id, movie_id):
    global sae_model, ratings_tensor
    try:
        user_idx = int(user_id) - 1
        movie_idx = int(movie_id) - 1
        if user_idx < 0 or user_idx >= ratings_tensor.shape[0]:
            return 3.0
        input_user = ratings_tensor[user_idx].unsqueeze(0)
        output = sae_model(input_user).detach().numpy().flatten()
        pred = output[movie_idx]
        if np.isnan(pred) or pred == -np.inf:
            return 3.0
        return pred
    except Exception as e:
        print(f"Autoencoder predictor error: {e}")
        return 3.0

# ALS predictor, folosind modelul implicit ALS
def als_predictor(user_id, movie_id):
    global als_model_implicit, user_to_idx, movie_to_idx
    try:
        if user_id not in user_to_idx or movie_id not in movie_to_idx:
            return 3.0
        u_idx = user_to_idx[user_id]
        m_idx = movie_to_idx[movie_id]
        pred = np.dot(als_model_implicit.user_factors[u_idx], als_model_implicit.item_factors[m_idx])
        rating = pred * 4.0 + 1.0
        return rating
    except Exception as e:
        print(f"ALS predictor error: {e}")
        return 3.0

# GNN predictor, folosind modelul GNN
def gnn_predictor(user_id, movie_id):
    global gnn_model, gnn_user_map, gnn_movie_map, gnn_genre_map
    if gnn_model is None or user_id not in gnn_user_map or movie_id not in gnn_movie_map:
        return 3.0
    try:
        u_idx = gnn_user_map[user_id]
        m_idx = gnn_movie_map[movie_id]
        g_vec = gnn_genre_map[m_idx]  # Se presupune că este deja un tensor
        u_t = torch.tensor([u_idx], dtype=torch.long)
        m_t = torch.tensor([m_idx], dtype=torch.long)
        g_t = g_vec.unsqueeze(0)
        with torch.no_grad():
            pred = gnn_model(m_t, u_t, g_t).item()
        return pred
    except Exception as e:
        print(f"GNN predictor error: {e}")
        return 3.0

# NMF predictor, folosind factorizarea NMF
def nmf_predictor(user_id, movie_id):
    """
    Prezice ratingul pentru (user_id, movie_id) folosind NMF.
    Se presupune că train_nmf_model a fost deja apelată, astfel încât W_nmf, H_nmf,
    user_id_to_index și movie_id_to_index sunt definite.
    """
    global W_nmf, H_nmf, user_id_to_index, movie_id_to_index
    try:
        # Dacă user_id sau movie_id nu există în mapping, returnează fallback
        if user_id not in user_id_to_index or movie_id not in movie_id_to_index:
            return 3.0
        u_idx = user_id_to_index[user_id]
        m_idx = movie_id_to_index[movie_id]
        # Calculează ratingul prezis ca produs scalar între factorii corespunzători:
        pred = np.dot(W_nmf[u_idx], H_nmf[:, m_idx])
        return pred
    except Exception as e:
        print("NMF predictor error:", e)
        return 3.0

#########################
# Cross-validation RMSE:
#########################

def compute_rmse_cv_for_predictor(predictor, ratings_df, cv=5):
    """
    Împarte ratingurile folosind KFold cross-validation și calculează RMSE mediu
    pe predicțiile făcute de funcția "predictor".
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmses = []
    for train_idx, test_idx in kf.split(ratings_df):
        test_data = ratings_df.iloc[test_idx]
        y_true = test_data['rating'].values.astype(float)
        y_pred = []
        for _, row in test_data.iterrows():
            pred = predictor(row['user_id'], row['movie_id'])
            y_pred.append(pred)
        fold_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmses.append(fold_rmse)
    return np.mean(rmses)

def compute_all_rmse_cv():
    ratings_df = get_ratings_dataframe()
    if ratings_df.empty:
        print("Nu există rating-uri în baza de date.")
        return {}
    
    missing = set(ratings_df['user_id']) - set(users_df.index)
    if missing:
        print("[WARN] orfani în ratings:", missing)

    results = {
        'baseline_rmse': compute_rmse_cv_for_predictor(baseline_predictor, ratings_df, cv=5),
        'svd_rmse': compute_rmse_cv_for_predictor(svd_predictor, ratings_df, cv=5),
        'cf_gender_rmse': compute_rmse_cv_for_predictor(cf_gender_predictor, ratings_df, cv=5),
        'cf_gen_occ_rmse': compute_rmse_cv_for_predictor(cf_gen_occ_predictor, ratings_df, cv=5),
        'nn_rmse': compute_rmse_cv_for_predictor(nn_predictor, ratings_df, cv=5),
        'knn_rmse': compute_rmse_cv_for_predictor(knn_predictor, ratings_df, cv=5),
        'wmean_rmse': compute_rmse_cv_for_predictor(weighted_mean_predictor,  ratings_df, cv=5),
        'mlp_gmf_rmse': compute_rmse_cv_for_predictor(mlp_gmf_predictor, ratings_df, cv=5),
        'autoencoder_rmse': compute_rmse_cv_for_predictor(autoencoder_predictor, ratings_df, cv=5),
        'als_rmse': compute_rmse_cv_for_predictor(als_predictor, ratings_df, cv=5),
        'gnn_rmse': compute_rmse_cv_for_predictor(gnn_predictor, ratings_df, cv=5),
        #'nmf_rmse': compute_rmse_cv_for_predictor(nmf_predictor, ratings_df, cv=5)
    }
    return results

def compute_rmse_single_split(predictor, ratings_df, test_size=0.25, random_state=42):
    """
    Împarte setul de ratinguri într-un set de test (test_size) și calculează RMSE pentru funcția 'predictor'.
    """
    # Împărțim datele (nu folosim funcțiile din scikit-learn, doar un simplu train_test_split)
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    y_true = test_data['rating'].values.astype(float)
    y_pred = np.array([predictor(row['user_id'], row['movie_id']) for _, row in test_data.iterrows()])
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_all_rmse_single_split():
    ratings_df = get_ratings_dataframe()
    if ratings_df.empty:
        print("Nu există rating-uri în baza de date.")
        return {}
    
    results = {
        'baseline_rmse': compute_rmse_single_split(baseline_predictor, ratings_df),
        'svd_rmse': compute_rmse_single_split(svd_predictor, ratings_df),
        'cf_gender_rmse': compute_rmse_single_split(cf_gender_predictor, ratings_df),
        'cf_gen_occ_rmse': compute_rmse_single_split(cf_gen_occ_predictor, ratings_df),
        'nn_rmse': compute_rmse_single_split(nn_predictor, ratings_df),
        'knn_rmse': compute_rmse_single_split(knn_predictor, ratings_df),
        'mlp_gmf_rmse': compute_rmse_single_split(mlp_gmf_predictor, ratings_df),
        'autoencoder_rmse': compute_rmse_single_split(autoencoder_predictor, ratings_df),
        'als_rmse': compute_rmse_single_split(als_predictor, ratings_df),
        'gnn_rmse': compute_rmse_single_split(gnn_predictor, ratings_df),
        'nmf_rmse': compute_rmse_single_split(nmf_predictor, ratings_df)
    }
    return results

def train_nmf_model(n_components=7, max_iter=200):
    """
    Antrenează un model NMF pe baza ratingurilor filmelor (sau textului asociat filmelor)
    și salvează factorii W și H în variabile globale.
    Este esențial ca movies_df să fie populat înainte (de ex. prin compute_similarity() sau update_movies()).
    """
    global W_nmf, H_nmf, vectorizer_nmf, movies_df, user_id_to_index, movie_id_to_index

    # Asigură-te că movies_df nu este gol:
    if movies_df is None or movies_df.empty:
        compute_similarity()  # sau altă funcție care populează movies_df

    # Construiește corpusul de text: dacă există coloana 'tags', folosește-o, altfel combină 'overview' și 'genres'
    if 'tags' in movies_df.columns:
        corpus = movies_df['tags'].values.astype('U')
    else:
        corpus = (movies_df['overview'] + " " + movies_df['genres']).values.astype('U')

    # Vectorizare
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_nmf = CountVectorizer(max_features=5000, stop_words='english')
    count_matrix = vectorizer_nmf.fit_transform(corpus)

    # Antrenează modelul NMF
    from sklearn.decomposition import NMF
    nmf_model = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
    W_nmf = nmf_model.fit_transform(count_matrix)
    H_nmf = nmf_model.components_  # H are forma (n_components, num_filme)

    print(f"NMF model antrenat: W shape = {W_nmf.shape}, H shape = {H_nmf.shape}")

    # (Opțional) Dacă dorești să utilizezi NMF ca predictor pentru ratinguri, va trebui să construiești
    # mape pentru utilizatori și filme din ratingurile din baza de date:
    ratings_df = get_ratings_dataframe()
    pivot = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    user_ids = list(pivot.index)
    movie_ids = list(pivot.columns)
    user_id_to_index = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_to_index = {mid: i for i, mid in enumerate(movie_ids)}
    print("Mapping-urile pentru NMF au fost create.")

def binarize_rating(rating, threshold=3.1):
    return 1 if rating >= threshold else 0

def compute_precision_recall_f1_cv_for_predictor(predictor, ratings_df, threshold=3.1, cv=5):
    """
    Pentru un anumit predictor care primește (user_id, movie_id) și returnează o predicție numerică,
    calculează Precision, Recall și F1 Score folosind KFold cross-validation.
    Ratingurile sunt binarizate folosind pragul 'threshold'.
    
    Returnează o dicționar cu valorile medii:
      { 'precision': ..., 'recall': ..., 'f1': ... }
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    precisions = []
    recalls = []
    f1_scores = []
    
    for train_idx, test_idx in kf.split(ratings_df):
        test_data = ratings_df.iloc[test_idx]
        y_true = [binarize_rating(r, threshold) for r in test_data['rating'].values.astype(float)]
        y_pred = []
        for _, row in test_data.iterrows():
            pred = predictor(row['user_id'], row['movie_id'])
            y_pred.append(binarize_rating(pred, threshold))
        # Calculează metricile pentru fold
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores)
    }

def compute_all_precision_recall_f1_cv():
    """
    Calculează Precision, Recall și F1 Score (medii, prin cross-validation) pentru
    fiecare predictor de rating din sistem.
    
    Se utilizează setul de ratinguri din baza de date.
    Returnează un dicționar cu rezultate pentru fiecare algoritm.
    """
    ratings_df = get_ratings_dataframe()
    if ratings_df.empty:
        print("Nu există rating-uri în baza de date.")
        return {}
    
    results = {
        'baseline': compute_precision_recall_f1_cv_for_predictor(baseline_predictor, ratings_df),
        'svd': compute_precision_recall_f1_cv_for_predictor(svd_predictor, ratings_df),
        'cf_gender': compute_precision_recall_f1_cv_for_predictor(cf_gender_predictor, ratings_df),
        'cf_gen_occ': compute_precision_recall_f1_cv_for_predictor(cf_gen_occ_predictor, ratings_df),
        #'nn': compute_precision_recall_f1_cv_for_predictor(nn_predictor, ratings_df),
        'knn': compute_precision_recall_f1_cv_for_predictor(knn_predictor, ratings_df),
        'wmean': compute_precision_recall_f1_cv_for_predictor(weighted_mean_predictor, ratings_df),

        #'mlp_gmf': compute_precision_recall_f1_cv_for_predictor(mlp_gmf_predictor, ratings_df),
        #'autoencoder': compute_precision_recall_f1_cv_for_predictor(autoencoder_predictor, ratings_df),
        #'als': compute_precision_recall_f1_cv_for_predictor(als_predictor, ratings_df),
        #'gnn': compute_precision_recall_f1_cv_for_predictor(gnn_predictor, ratings_df),
        #'nmf': compute_precision_recall_f1_cv_for_predictor(nmf_predictor, ratings_df)
    }
    return results

def compute_precision_recall_f1_single_split(predictor, ratings_df, threshold=3.1, test_size=0.25, random_state=42):
    """
    Împarte setul de ratinguri într-un set de test și calculează Precision, Recall și F1 Score
    pentru predictor, folosind un singur train-test split.
    """
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    y_true = [binarize_rating(r, threshold) for r in test_data['rating'].values.astype(float)]
    y_pred = [binarize_rating(predictor(row['user_id'], row['movie_id']), threshold) 
              for _, row in test_data.iterrows()]
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_all_precision_recall_f1_single_split():
    ratings_df = get_ratings_dataframe()
    if ratings_df.empty:
        print("Nu există rating-uri în baza de date.")
        return {}
    
    results = {
        'baseline': compute_precision_recall_f1_single_split(baseline_predictor, ratings_df),
        'svd': compute_precision_recall_f1_single_split(svd_predictor, ratings_df),
        'cf_gender': compute_precision_recall_f1_single_split(cf_gender_predictor, ratings_df),
        'cf_gen_occ': compute_precision_recall_f1_single_split(cf_gen_occ_predictor, ratings_df),
        'nn': compute_precision_recall_f1_single_split(nn_predictor, ratings_df),
        'knn': compute_precision_recall_f1_single_split(knn_predictor, ratings_df),
        'mlp_gmf': compute_precision_recall_f1_single_split(mlp_gmf_predictor, ratings_df),
        'autoencoder': compute_precision_recall_f1_single_split(autoencoder_predictor, ratings_df),
        'als': compute_precision_recall_f1_single_split(als_predictor, ratings_df),
        'gnn': compute_precision_recall_f1_single_split(gnn_predictor, ratings_df),
        'nmf': compute_precision_recall_f1_single_split(nmf_predictor, ratings_df)
    }
    return results

def split_train_test(dataframe, test_size=0.2):
        """
        Împarte DataFrame-ul de ratinguri pe utilizatori în set de antrenament și test.
        Returnează două DataFrame-uri: train_df și test_df.
        Utilizatorii cu 0 sau 1 rating sunt puși integral în train (niciun rating în test).
        """
        train_records = []
        test_records = []
        # Împărțim pentru fiecare utilizator în parte, asigurându-ne că fiecare user are cel puțin un rating în train (dacă e posibil)
        for user_id, user_ratings in dataframe.groupby('user_id'):
            n_ratings = len(user_ratings)
            if n_ratings <= 1:
                # Dacă utilizatorul are 0 sau 1 rating, nu putem face split – îl adăugăm tot în train
                train_records.extend(user_ratings.to_dict('records'))
            else:
                # Calculăm numărul de ratinguri de pus în test (cel puțin 1)
                test_count = max(1, int(n_ratings * test_size))
                if test_count >= n_ratings:
                    test_count = 1  # asigurăm cel puțin un rating rămas în train
                # Selectăm aleator test_count ratinguri pentru test
                import random
                test_indices = random.sample(list(user_ratings.index), test_count)
                for idx, row in user_ratings.iterrows():
                    if idx in test_indices:
                        test_records.append(row.to_dict())
                    else:
                        train_records.append(row.to_dict())
        # Construim DataFrame-urile de train și test din listele de dicționare
        train_df = pd.DataFrame(train_records)
        test_df = pd.DataFrame(test_records)
        return train_df, test_df

TOP_N = 10                  # număr de recomandări luate în considerare la calculele Precision/Recall
RATING_THRESHOLD = 4        # pragul de rating pentru a considera un film "relevant" (ex: >=4 stele)
CV_FOLDS = 5 

# Funcție pentru calculul RMSE pe un singur split train/test
def evaluate_rmse_single(algorithm, ratings_df, test_size=0.2):
    """
    Calculează RMSE pentru un algoritm dat, folosind o singură împărțire train/test.
    """
    # Împărțim datele în train și test
    train_df, test_df = split_train_test(ratings_df, test_size)
    if test_df.empty:
        # Dacă niciun utilizator nu are ratinguri în test (date insuficiente pentru evaluare)
        return 0.0
    # Antrenăm algoritmul pe datele de train, dacă este necesar
    model = None
    if hasattr(algorithm, 'train'):
        model = algorithm.train(train_df)
    elif hasattr(algorithm, 'fit'):
        model = algorithm.fit(train_df)
    # Parcurgem fiecare rating din setul de test și prezicem ratingul folosind algoritmul
    se_sum = 0.0  # suma erorilor pătratice
    count = 0     # numărul de predicții efectuate
    for _, row in test_df.iterrows():
        user = row['user_id']
        # FIX: folosim 'movie_id' în loc de 'item_id'
        movie = row['movie_id']
        actual_rating = row['rating']
        # Obținem predicția de rating a algoritmului pentru (user, movie)
        if model is not None and hasattr(model, 'predict'):
            predicted_rating = model.predict(user, movie)
        elif hasattr(algorithm, 'predict'):
            predicted_rating = algorithm.predict(user, movie, train_df)
        else:
            continue
        if predicted_rating is None:
            continue
        se_sum += (predicted_rating - actual_rating) ** 2
        count += 1
    import math
    rmse_value = math.sqrt(se_sum / count) if count > 0 else 0.0
    return rmse_value


# Funcție pentru calculul Precision, Recall, F1 pe un singur split train/test
def evaluate_prf_single(algorithm, ratings_df, test_size=0.2):
    """
    Calculează metricele Precision, Recall și F1 pentru un algoritm dat, folosind un singur split train/test.
    Returnează un tuplu (precision, recall, f1) calculat ca medie pe toți utilizatorii evaluați.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    train_df, test_df = split_train_test(ratings_df, test_size)
    if test_df.empty:
        return 0.0, 0.0, 0.0
    model = None
    if hasattr(algorithm, 'train'):
        model = algorithm.train(train_df)
    elif hasattr(algorithm, 'fit'):
        model = algorithm.fit(train_df)
    precision_list = []
    recall_list = []
    f1_list = []
    for user_id, user_test in test_df.groupby('user_id'):
        test_relevant_items = set()
        for _, row in user_test.iterrows():
            if row['rating'] >= RATING_THRESHOLD:
                # FIX: folosim 'movie_id' în loc de 'item_id'
                test_relevant_items.add(row['movie_id'])
        if len(test_relevant_items) == 0:
            continue
        recommended_items = []
        if model is not None and hasattr(model, 'recommend'):
            recs = model.recommend(user_id, top_n=TOP_N)
        elif hasattr(algorithm, 'recommend'):
            recs = algorithm.recommend(user_id, train_df, top_n=TOP_N)
        else:
            recs = []
        recommended_set = set()
        if recs:
            if isinstance(recs[0], tuple):
                recommended_set = {item for item, score in recs}
            else:
                recommended_set = set(recs)
        hits = len(recommended_set & test_relevant_items)
        prec = hits / len(recommended_set) if len(recommended_set) > 0 else 0.0
        rec = hits / len(test_relevant_items) if len(test_relevant_items) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
    if len(precision_list) == 0:
        return 0.0, 0.0, 0.0
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1 = sum(f1_list) / len(f1_list)
    return avg_precision, avg_recall, avg_f1


    # Funcție pentru calculul mediu RMSE cu cross-validation (n împărțiri diferite)
def evaluate_rmse_cv(algorithm, ratings_df, n_splits=5):
        """
        Calculează RMSE mediu pe n împărțiri (cross-validation) pentru un algoritm dat.
        """
        total_rmse = 0.0
        # Efectuăm n împărțiri aleatorii și calculăm RMSE pentru fiecare
        for i in range(n_splits):
            rmse_i = evaluate_rmse_single(algorithm, ratings_df)  # folosim split aleator (test_size implicit 0.2)
            total_rmse += rmse_i
        # Media RMSE
        avg_rmse = total_rmse / n_splits if n_splits > 0 else 0.0
        return avg_rmse

 
    
class AlgorithmWrapper:
    """Wrapper pentru algoritmi de recomandare"""
    def __init__(self, predict_func=None, recommend_func=None):
        self.predict_func = predict_func
        self.recommend_func = recommend_func

    def predict(self, user_id, item_id, train_df):
        if self.predict_func:
            return self.predict_func(user_id, item_id, train_df)
        return 3.0  # Valoare implicită

    def recommend(self, user_id, train_df):
        if self.recommend_func:
            return self.recommend_func(user_id, train_df)
        return []  # Listă goală ca fallback



def compute_all_rmse_cv_nn(cv_folds: int = 3):
    ratings_df = get_ratings_dataframe_nn()
    if ratings_df.empty:
        return {}

    results = {}

    # --- 1) SVD with Surprise CV ---
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id','movie_id','rating']], reader)
    svd = SVD()
    cv_svd = cross_validate(svd, data, measures=['RMSE'], cv=cv_folds, n_jobs=-1, verbose=False)
    results['svd_rmse'] = np.mean(cv_svd['test_rmse'])

    # --- 2) single train/test split pentru modele Keras (NN-CF & GMF+MLP) ---
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    y_true_full = test_df['rating'].values

    def rmse_keras(model, u_map, m_map):
        um = test_df['user_id'].map(u_map)
        mm = test_df['movie_id'].map(m_map)
        valid = um.notna() & mm.notna()
        if not valid.any():
            return None
        u_idx = um[valid].astype(np.int64).values
        m_idx = mm[valid].astype(np.int64).values
        y_true = y_true_full[valid.values]
        preds_norm = model.predict([u_idx, m_idx], batch_size=2048).flatten()
        preds = preds_norm * 4.0 + 1.0
        return np.sqrt(mean_squared_error(y_true, preds))

    # --- 2a) NN‑CF ---
    if nn_model is not None:
        results['nn_rmse'] = rmse_keras(nn_model, user_map, movie_map)

    # --- 2b) GMF+MLP ---
    try:
        # train_and_save_gmf_mlp(db_path="tmdb_movies_for_nn.db", save_path="gmf_mlp_model_tf")
        load_gmf_mlp_model("gmf_mlp_model_tf")
        rm = rmse_keras(gmf_mlp_model, user_map_gmfmlp, item_map_gmfmlp)
        if rm is not None:
            results['mlp_gmf_rmse'] = rm
    except Exception:
        results['mlp_gmf_rmse'] = None

    # --- 3) Autoencoder (SAE) single split ---
        # --- 3) Autoencoder (SAE) – fair masking & denormalize --------------------
     # --- 3) Autoencoder (SAE) single split ---
     # --- 3) Autoencoder (SAE) single split ---
    if 'sae_model' in globals() and 'ratings_tensor' in globals():
            # build user/movie → index mappings
            _, ae_u_map, ae_m_map = create_ratings_matrix()

            # map to indices
            um = test_df['user_id'].map(ae_u_map)
            mm = test_df['movie_id'].map(ae_m_map)

            # drop any missings
            mask = um.notna() & mm.notna()
            u_idx = um[mask].astype(int).values
            m_idx = mm[mask].astype(int).values
            y_true = y_true_full[mask.values]

            # now drop any out-of-range indices
            n_users, n_movies = ratings_tensor.shape
            keep = (u_idx >= 0) & (u_idx < n_users) & (m_idx >= 0) & (m_idx < n_movies)
            u_idx = u_idx[keep]
            m_idx = m_idx[keep]
            y_true = y_true[keep]

            if len(u_idx) == 0:
                results['autoencoder_rmse'] = np.nan
            else:
            # only now safe to index into ratings_tensor
                with torch.no_grad():
                    inputs = ratings_tensor[u_idx]             # shape (batch, nb_movies)
                    outputs = sae_model(inputs).cpu().numpy()  # shape (batch, nb_movies)

                # pick out predicted ratings at the right movie-positions
                preds = outputs[np.arange(len(m_idx)), m_idx]

                # replace any NaN or infinite by fallback
                preds = np.where(np.isfinite(preds), preds, 3.0)

                results['autoencoder_rmse'] = np.sqrt(mean_squared_error(y_true, preds))  



    # --- 4) GNN (cross‑validation) ---
    try:
        results['gnn_rmse'] = compute_rmse_cv_for_predictor(gnn_predictor, ratings_df, cv=cv_folds)
    except Exception:
        results['gnn_rmse'] = None

    # --- 5) NMF (cross‑validation) ---
    #try:
    #    results['nmf_rmse'] = compute_rmse_cv_for_predictor(nmf_predictor, ratings_df, cv=cv_folds)
    except Exception:
        results['nmf_rmse'] = None

    return results

from flask_cors import CORS


if __name__ == '__main__':
    CORS(app, supports_credentials=True)
    create_database()
    update_movies()
    print("sunt aici")
    #build_bert_embeddings()
    #load_bert_embeddings()
    # sae_model, ratings_tensor = train_autoencoder_model(nb_epoch=10)
    # sae_model, ratings_tensor = load_autoencoder_model()
    #prepare_multi_label_model(csv_path="MovieGenre.csv")
    # build_bert_embeddings()
    print(">>> Embedding-urile BERT pentru filme au fost construite cu succes!")
    #prepare_poster_model(csv_path="MovieGenre.csv")

    #import_missing_movies_from_ratings("tmdb_movies.db")
    #import_missing_movies_from_ratings("tmdb_movies.db")
    initialize_knn_model(db_path="tmdb_movies.db") 
    #train_gnn_model(db_path="tmdb_movies.db", model_path="gnn_model.pt", epochs=1500)
    gnn_model, gnn_user_map, gnn_movie_map, gnn_genre_map = load_gnn_model("gnn_model.pt")
    #train_and_save_gmf_mlp(db_path="tmdb_movies_for_nn.db", save_path="gmf_mlp_model_tf")

    load_gmf_mlp_model("gmf_mlp_model_tf")
    # model_path = "C:\Users\BANU\Desktop\JAVA\Licenta mea/gmf_mlp_model_tf"
    # load_gmf_mlp_model(model_path)
    #train_and_save_model(db_path="tmdb_movies_for_nn.db", save_path="nn_model_tf")
    # train_nmf_model(n_components=7, max_iter=200)

    nn_model, user_map, movie_map = load_nn_model("nn_model_tf")            # <-- încarci din folder

    print(">>> Neural Model încărcat cu succes!")
    
    compute_similarity()
    initialize_collaborative_filtering()
    initialize_collaborative_filtering_clicks()
    initialize_user_based_matrices()
    load_users()
    print("Calculating RMSE for users 1-37...\n")
    
    # for user_id in range(1, 38):  # 1-37 (inclusiv)
    #     algo_rmse = compute_rmse_for_algorithms(user_id)
    #     print("-" * 30)
    #     print(f"\nUser ID: {user_id}")
    #     print(f"Content-based RMSE:       {algo_rmse['content_cb']:.4f}")
    
    # print("-" * 30)
    # print(f"Mean RMSE:                {algo_rmse['mean_rmse']:.4f}\n")
    # rmse_results = compute_all_rmse_cv()
    # print("RMSE pentru toți algoritmii (cu cross-validation):")
    # for alg, score in rmse_results.items():
    #     print(f"{alg}: {score:.4f}")
        
    
  
    # prf_cv_results = compute_all_precision_recall_f1_cv()
    # print("Precision, Recall, F1 (cross-validation):")
    # for alg, metrics in prf_cv_results.items():
    #     print(f"{alg}: Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}, F1 = {metrics['f1']:.4f}")
    
    
    # rmse_results = compute_all_rmse_cv_nn()
    # print("RMSE pentru toți algoritmii (cu cross-validation):")
    # for alg, score in rmse_results.items():
    #     print(f"{alg}: {score:.4f}")
    
    
    

    # 5. Pornire server Flask
    print("\n" + "="*60)
    print("Evaluare completă. Pornire server Flask...")
    print("="*60)
   # get_collaborative_filtering_recommendations(1, top_n=10)
    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context=('C:\\Users\\BANU\\Desktop\\micro3\\Licenta mea\\cer.pem', 'C:\\Users\\BANU\\Desktop\\micro3\\Licenta mea\\ec.pem'))

