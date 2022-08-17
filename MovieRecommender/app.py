import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#import streamlit as st

#st.title("Movie Recommender")

# Import csv files and find their lengths
movies = pd.read_csv('resources/movies.csv')
ratings = pd.read_csv('resources/ratings.csv')
#print("This dataset has ", len(movies), " movies and ", len(ratings), " ratings")


# Clean movie titles
## Function to clean movie titles - get rid of non alphanumeric characters
def title_clean(movie_title):
    return re.sub("[^a-zA-Z0-9 ]", "", movie_title)
movies["titleClean"] = movies["title"].apply(title_clean) # creating new column in movies with clean titles

# Create tfidf matrix
vectorizer = TfidfVectorizer(ngram_range=(1,2)) # creating vectorizer with ngrams of 1 or 2 (looks at 1 or 2 words when searching)
tfidf_table = vectorizer.fit_transform(movies["titleClean"]) # creating tfidf table (sets of numbers) with clean titles
#print(tfidf_table)

# Create search function
def searcher(title):
    title = title_clean(title)
    vec_value = vectorizer.transform([title]) # vectorizer value of the given title string
    sim_score = cosine_similarity(vec_value, tfidf_table).flatten() # compares similarity with title vectorizer and all values in tfidf
    ind = np.argpartition(sim_score, -5)[-5:] # finds 5 most similar movie indexes within database
    sim_movies = movies.iloc[ind][::-1] # finds movie titles of the 5 most similar movies based on search
    return sim_movies

#print(searcher("The Godfather"))

# Find users who liked the same movie
movie_id = 106696
sim_users  = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique() # find all users who rated specified movie over a 4
sim_users_movies = ratings[(ratings["userId"].isin(sim_users)) & (ratings["rating"] > 4)]["movieId"] # find all movies that the users who liked specified movie also liked
sim_users_movies = sim_users_movies.value_counts() / len(sim_users) # find percentage of users that recommend each movie
sim_users_movies = sim_users_movies[sim_users_movies > .1] # find movies that more than 10% of users who also liked our movie liked
#print(sim_users_movies)

# Find how much all users like movies
all_users = ratings[(ratings["movieId"].isin(sim_users_movies.index)) & (ratings["rating"] > 4)] # all users that watched recommended movies and gave over a 4 rating
all_users_movies = all_users["movieId"].value_counts() / len(all_users["userId"].unique()) # find percentage of all users that recommend each movie in sim_users_movies

# Create recommendation score
sim_all_percent = pd.concat([sim_users_movies, all_users_movies], axis = 1) # compare similar users recommendation percentage to all users recommendation percentage for recommended movies
sim_all_percent.columns = ["sim", "all"]
sim_all_percent["score"] = sim_all_percent["sim"] / sim_all_percent["all"] # find score - ratio between how much similar users liked a recommended movie and how much all users liked a recommended movie
sim_all_percent = sim_all_percent.sort_values("score", ascending = False)
print(sim_all_percent.head(10).merge(movies, left_index = True, right_on = "movieId"))