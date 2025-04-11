import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
# NUM_EPOCHS=100
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

# Perform TF-IDF vectorization
tf = TfidfVectorizer(min_df=1)  # Minimum document frequency of 1 document
tfidf_matrix = tf.fit_transform(movies['genres'])
# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

def genre_recommendations(interested_genres, movie_posters):
    genre_vector = tf.transform([interested_genres])
    sim_scores = linear_kernel(genre_vector, tfidf_matrix).flatten()
    # Get top 20 movies with the highest similarity scores
    sim_scores_idx = sim_scores.argsort()[-20:][::-1]
    movie_info = [(movies['movieId'].iloc[i], movies['title'].iloc[i],
                   movie_posters.get(movies['movieId'].iloc[i], 'https://via.placeholder.com/200x300')) for i in
                  sim_scores_idx]
    return movie_info


def generate_recommendations(user_id):
    # Get all users who rated the same movies as the target user
    similar_users = \
    ratings[(ratings['userId'] != user_id) & ratings['movieId'].isin(ratings[ratings['userId'] == user_id]['movieId'])][
        'userId'].unique()

    # Get average rating for each movie that similar users have rated
    movie_ratings = ratings[ratings['userId'].isin(similar_users)].groupby('movieId')['rating'].mean().reset_index()

    # Sort movies by average rating
    movie_ratings = movie_ratings.sort_values(by='rating', ascending=False)

    # Get top 20 recommended movieIds
    recommended_movies = movie_ratings.head(20)['movieId'].tolist()

    return recommended_movies

train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# Evaluate recommendations
mae_scores = []
user_ids = []
for user_id in test_ratings['userId'].unique():
    user_ratings = test_ratings[test_ratings['userId'] == user_id]
    recommended_movies = generate_recommendations(user_id)  # Replace with your recommendation function
    actual_ratings = user_ratings[user_ratings['movieId'].isin(recommended_movies)]
    if not actual_ratings.empty:
        mae = mean_absolute_error(actual_ratings['rating'], [5] * len(actual_ratings))  # Assuming all recommended movies are rated 5
        mae_scores.append(mae)
        user_ids.append(user_id)

if mae_scores:
    # Plot MAE scores
    plt.figure(figsize=(10, 6))
    plt.bar(user_ids, mae_scores, color='skyblue')
    plt.xlabel('User ID')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error for Recommended Movies by User')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("No recommendations to evaluate.")
# Inside your existing code after plotting the bar plot for MAE scores

# Assuming you have similarity_scores available
similarity_scores = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]  # Example similarity scores

# Plot histogram of similarity scores
plt.figure(figsize=(8, 6))
plt.hist(similarity_scores, bins=10, color='lightblue')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Histogram of Similarity Scores for Recommended Movies')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
