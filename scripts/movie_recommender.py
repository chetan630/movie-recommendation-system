# -----------------------------
# Movie Recommendation System (Hybrid: Collaborative + Content-Based)
# -----------------------------

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 1: Load Data Safely
# -----------------------------
# Dynamically locate the data folder (so paths work regardless of where you run the script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))   # movieId, title, genres
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv")) # userId, movieId, rating, timestamp
tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))       # userId, movieId, tag, timestamp

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
# One-hot encode genres for content-based similarity
movies_genres = movies['genres'].str.get_dummies(sep='|')

# Merge tags (optional enrichment)
tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
movies = movies.merge(tags_agg, on='movieId', how='left')

# -----------------------------
# Step 3: User-Based Collaborative Filtering
# -----------------------------
# Build user-item rating matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Compute cosine similarity between users
user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

def predict_cf(user_id, movie_id, k=5):
    """Predict rating of user_id for movie_id using top-k similar users."""
    if movie_id not in user_item_matrix.columns:
        return 0  # if movie not rated by anyone
    
    sim_scores = user_sim_df[user_id].drop(user_id)  # exclude self
    rated_users = user_item_matrix[movie_id][sim_scores.index]
    rated_users = rated_users[rated_users > 0]  # keep only users who rated

    if rated_users.empty:
        return user_item_matrix[movie_id].mean()  # fallback to global average
    
    # Take top-k similar users who rated this movie
    top_users = rated_users.index[:k]
    weights = sim_scores[top_users]

    if weights.sum() == 0:
        return rated_users.mean()  # fallback to average of available ratings
    
    return np.dot(user_item_matrix.loc[top_users, movie_id], weights) / weights.sum()

# -----------------------------
# Step 4: Content-Based Filtering
# -----------------------------
# Cosine similarity between movies by genre encoding
genre_sim = cosine_similarity(movies_genres.values)
genre_sim_df = pd.DataFrame(genre_sim, index=movies['movieId'], columns=movies['movieId'])

# -----------------------------
# Step 5: Hybrid Recommendation
# -----------------------------
def hybrid_recommend(user_id, top_n=10, cf_weight=0.7, cb_weight=0.3):
    """Recommend movies by blending CF and CB scores."""
    user_ratings = ratings[ratings['userId'] == user_id]
    liked_movies = user_ratings['movieId'].tolist()
    
    scores = []

    for _, movie in movies.iterrows():
        movie_id = movie['movieId']

        # CF prediction
        cf_score = predict_cf(user_id, movie_id)

        # CB prediction
        if liked_movies:
            sim_scores = [genre_sim_df.loc[movie_id, mid] for mid in liked_movies if mid in genre_sim_df.columns]
            cb_score = np.mean(sim_scores) if sim_scores else 0
        else:
            cb_score = 0

        # Hybrid score
        final_score = cf_weight * cf_score + cb_weight * cb_score
        scores.append((movie['title'], final_score))

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top-N
    return scores[:top_n]

# -----------------------------
# Step 6: Example Usage
# -----------------------------
if __name__ == "__main__":
    user_id = int(input("Enter the user Id: "))
    top_movies = hybrid_recommend(user_id=user_id, top_n=5)

    print(f"\n🎬 Top 5 Recommended Movies for User {user_id}:")
    for i, (title, score) in enumerate(top_movies, 1):
        print(f"{i}. {title}  (score: {score:.2f})")