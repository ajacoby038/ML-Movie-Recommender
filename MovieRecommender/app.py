import pandas as pd

movies = pd.read_csv('resources/movies.csv')
ratings = pd.read_csv('resources/ratings.csv')
print("This dataset has ", len(movies), " movies and ", len(ratings), " ratings")

dataset_complete = ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating')
dataset_complete.fillna(0, inplace=True)
print(dataset_complete.head())