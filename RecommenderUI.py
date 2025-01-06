# RecommenderUI.py
# --------------------------------------------------------------------------------
# This revised code allows users to adjust hyperparameters directly from the
# Streamlit interface for K-Means, k-NN, and Random Forest algorithms. Providing
# hyperparameter controls from the UI empowers users to fine-tune the models,
# a process known as "Hiperparametre Ayarlaması" in machine learning, where
# different parameter values are tested to discover the most suitable ones for
# the model [[4]].

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# Configure the Streamlit page
st.set_page_config(
    page_title="Movie Recommender System with Hyperparameter Tuning",
    page_icon="🎬",
    layout="wide"
)

# Custom Style for a modern look
st.markdown("""
    <meta charset="utf-8">        
    <style>
    .movie-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-5px);
    }
    .movie-rating {
        font-size: 1.2rem;
        color: #ff9800;
    }
    .section-header {
        background: #1E88E5;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """
    Loads the movie dataset from a CSV file (IMDBcleaned.csv).
    Tries UTF-8 encoding first, falls back to ISO-8859-1 if needed.
    """
    try:
        df = pd.read_csv('IMDBcleaned.csv', encoding='utf-8')
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('IMDBcleaned.csv', encoding='ISO-8859-1')
            return df
        except Exception as e:
            st.error(f"Error loading data with ISO-8859-1 encoding: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_feature_matrix(df):
    """
    Creates a TF-IDF based feature matrix by combining genre, plot,
    director, and cast into a single text string for each movie.
    """
    df['combined_features'] = df['Genre'] + ' ' + df['Plot'] + ' ' + df['Director'] + ' ' + df['Cast']
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df['combined_features'])


def get_recommendations(
    feature_matrix,
    movie_idx,
    df,
    algorithm='kmeans',
    n_recommendations=10,
    min_rating=5.0,
    genres=None,
    year_range=None,
    # Hyperparameters for K-Means
    kmeans_clusters=8,
    kmeans_init='k-means++',
    kmeans_max_iter=300,
    # Hyperparameters for K-NN
    knn_neighbors=5,
    knn_metric='cosine',
    # Hyperparameters for Random Forest
    rf_estimators=100,
    rf_max_depth=None
):
    """
    Provides movie recommendations based on the chosen algorithm and its
    hyperparameters. Post-filters recommendations based on min_rating,
    selected genres, and release year range.
    """
    with st.spinner('Finding similar movies...'):
        if algorithm == 'kmeans':
            kmeans = KMeans(
                n_clusters=kmeans_clusters,
                init=kmeans_init,
                max_iter=kmeans_max_iter,
                random_state=42
            )
            cluster_labels = kmeans.fit_predict(feature_matrix)
            movie_cluster = cluster_labels[movie_idx]
            cluster_movies = np.where(cluster_labels == movie_cluster)[0]
            similarities = feature_matrix[movie_idx].dot(
                feature_matrix[cluster_movies].T
            ).toarray().flatten()
            # Sort by similarity (dot product) and pick top matches
            similar_indices = cluster_movies[np.argsort(similarities)[-n_recommendations*2-1:-1]]

        elif algorithm == 'knn':
            knn = NearestNeighbors(
                n_neighbors=knn_neighbors*2+1,
                metric=knn_metric
            )
            knn.fit(feature_matrix)
            distances, indices = knn.kneighbors(
                feature_matrix[movie_idx].reshape(1, -1)
            )
            similar_indices = indices[0][1:]

        else:  # 'rf' -> Random Forest
            le = LabelEncoder()
            df['genre_encoded'] = le.fit_transform(
                df['Genre'].apply(lambda x: x.split(',')[0])
            )
            rf = RandomForestClassifier(
                n_estimators=rf_estimators,
                max_depth=rf_max_depth,
                random_state=42
            )
            rf.fit(feature_matrix.toarray(), df['genre_encoded'])
            predictions = rf.predict_proba(feature_matrix.toarray())
            target_genre_probs = predictions[movie_idx]
            similarities = np.dot(predictions, target_genre_probs)
            similar_indices = np.argsort(similarities)[-n_recommendations*2-1:-1]

        # Apply filtering
        filtered_indices = []
        for idx in similar_indices:
            movie = df.iloc[idx]

            if movie['IMDB Rating'] < min_rating:
                continue
            if genres and not any(genre.strip() in movie['Genre'] for genre in genres):
                continue
            if year_range and not (year_range[0] <= movie['Release Year'] <= year_range[1]):
                continue

            filtered_indices.append(idx)

        if len(filtered_indices) < n_recommendations:
            st.warning(
                f"Only {len(filtered_indices)} movies match your criteria. "
                "Consider relaxing your filters."
            )

        return filtered_indices[:n_recommendations]


def display_movie_card(movie, is_selected=False):
    """
    Displays a movie card with styling that differs if the movie is the selected one.
    """
    background_color = (
        "linear-gradient(105deg, #82bef3, #ffffff)"
        if is_selected else
        "linear-gradient(105deg, #e57e9d, #ffffff)"
    )
    border = "2px solid #1E88E5" if is_selected else "2px solid #e57e9d"

    st.markdown(f"""
        <div class="movie-card" style="background: {background_color}; border: {border};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3>{movie['Movie Name']}</h3>
                <span class="movie-rating">★ {movie['IMDB Rating']}</span>
            </div>
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                <small>📅 {movie['Release Year']} | 🎭 {movie['Genre']} | 👍 {movie['Number of votes']} votes</small>
            </div>
            <p><strong>🎬 Director:</strong> {movie['Director']}</p>
            <p><strong>⭐ Cast:</strong> {movie['Cast']}</p>
            <details>
                <summary>Show Plot</summary>
                <p style="padding: 1rem 0;">{movie['Plot']}</p>
            </details>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.title("🎬 Movie Recommendation System with Hyperparameter Tuning")

    df = load_data()
    if df is None:
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        selected_movie = st.selectbox(
            "Select a movie:",
            df['Movie Name'].tolist()
        )

        # Algorithm selection with descriptive options
        algorithm_name = st.radio(
            "Algorithm:",
            ["K-means Clustering", "K-Nearest Neighbors", "Random Forest"]
        )

        # Map algorithm names to short identifiers
        algorithm_map = {
            "K-means Clustering": "kmeans",
            "K-Nearest Neighbors": "knn",
            "Random Forest": "rf"
        }
        alg_id = algorithm_map.get(algorithm_name, "kmeans")

        # General setting for number of recommendations
        n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)

        # Hyperparameter controls
        st.subheader("Hyperparameters")
        if alg_id == "kmeans":
            kmeans_clusters = st.slider("Number of Clusters (k):", 2, 30, 8)
            kmeans_init = st.selectbox("Initialization Method:", ["k-means++", "random"])
            kmeans_max_iter = st.slider("Max Iterations:", 100, 1000, 300)
            # Default placeholders for other algorithms
            knn_neighbors = 5
            knn_metric = 'cosine'
            rf_estimators = 100
            rf_max_depth = None

        elif alg_id == "knn":
            knn_neighbors = st.slider("Number of Neighbors (k):", 2, 30, 5)
            knn_metric = st.selectbox("Distance Metric:", ["cosine", "euclidean", "manhattan"])
            # Default placeholders for other algorithms
            kmeans_clusters = 8
            kmeans_init = "k-means++"
            kmeans_max_iter = 300
            rf_estimators = 100
            rf_max_depth = None

        else:  # alg_id == "rf"
            rf_estimators = st.slider("Number of Estimators:", 50, 300, 100)
            rf_max_depth = st.selectbox("Max Depth (None = unlimited):", [None, 5, 10, 20, 50])
            # Default placeholders for other algorithms
            kmeans_clusters = 8
            kmeans_init = "k-means++"
            kmeans_max_iter = 300
            knn_neighbors = 5
            knn_metric = 'cosine'

        # Advanced Filters
        with st.expander("Advanced Filters"):
            min_rating = st.slider("Minimum IMDb Rating", 0.0, 10.0, 5.0)
            genres = st.multiselect("Select Genres", df['Genre'].unique())
            year_range = st.slider(
                "Release Year Range",
                int(df['Release Year'].min()),
                int(df['Release Year'].max()),
                (1990, 2024)
            )

        if st.button("Get Recommendations"):
            if not selected_movie:
                st.warning("Please select a movie first!")
                return

            movie_idx = df[df['Movie Name'] == selected_movie].index[0]
            feature_matrix = create_feature_matrix(df)

            similar_indices = get_recommendations(
                feature_matrix=feature_matrix,
                movie_idx=movie_idx,
                df=df,
                algorithm=alg_id,
                n_recommendations=n_recommendations,
                min_rating=min_rating,
                genres=genres,
                year_range=year_range,
                kmeans_clusters=kmeans_clusters,
                kmeans_init=kmeans_init,
                kmeans_max_iter=kmeans_max_iter,
                knn_neighbors=knn_neighbors,
                knn_metric=knn_metric,
                rf_estimators=rf_estimators,
                rf_max_depth=rf_max_depth
            )

            with col2:
                st.subheader("Selected Movie")
                display_movie_card(df.iloc[movie_idx], is_selected=True)

                st.subheader("Recommended Movies")
                for idx in similar_indices:
                    display_movie_card(df.iloc[idx])

            # Debugging info
            print(f"Selected algorithm: {algorithm_name}, alg_id: {alg_id}")


if __name__ == "__main__":
    main()