import pandas as pd

movies = pd.read_csv('resources/movies.csv')
ratings = pd.read_csv('resources/ratings.csv')
print("This dataset has ", len(movies), " movies and ", len(ratings), " ratings")
