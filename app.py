from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import index  # Import index.py for recommendation generation
import csv
# import matplotlib.pyplot as plt
# import io
# from index import tfidf_matrix, ratings
# from index import train_and_evaluate_model
# import numpy as np
# from threading import Thread

app = Flask(__name__)
# NUM_EPOCHS = 100
# Load data
ratings = pd.read_csv('ratings.csv')
users = pd.read_csv('tags.csv')
movies = pd.read_csv('movies.csv')

# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')
# Load movie poster data
movie_posters = {}
with open('movie_poster.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        movie_posters[int(row[0])] = row[1]

# Train and evaluate model
# train_losses, test_losses, train_maes, test_maes, train_accuracies, test_accuracies = train_and_evaluate_model(tfidf_matrix, ratings)


# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = int(request.form['userId'])
        if user_id > 671:
            return redirect('/interests')
        else:
            # Additional functionality for user_id < 672
            user_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
            genres = movies[movies['movieId'].isin(user_movies)]['genres'].str.split(
                '|').explode().value_counts().index.tolist()
            return redirect(f'/recommendations/{",".join(genres)}')
    return render_template('login.html')


# Route for the interests selection page
@app.route('/interests', methods=['GET', 'POST'])
def interests():
    if request.method == 'POST':
        selected_genres = request.form.getlist('genre')
        return redirect(f'/recommendations/{",".join(selected_genres)}')
    return render_template('interests.html')


# Route for displaying recommendations
@app.route('/recommendations/<genres>', methods=['GET'])
def recommendations(genres):
    # Call index.py functions here
    recommendations = index.genre_recommendations(genres, movie_posters)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
