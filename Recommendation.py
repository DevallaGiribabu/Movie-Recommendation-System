import pandas as pd
import numpy as np
import ast


# Load datasets

movies = pd.read_csv("tmdb_10000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# Merge datasets

movies = movies.merge(credits, on="title")


# Select features

movies = movies[
[
"movie_id",
"title",
"overview",
"genres",
"keywords",
"cast",
"crew"
]
]


# Remove missing data

movies.dropna(inplace=True)



# Convert genres and keywords

def convert(obj):
    L = []
    parsed = ast.literal_eval(obj)
    
    # Handle both list and dict formats
    if isinstance(parsed, dict):
        # Extract the list from the dictionary (e.g., {'keywords': [...]})
        data = list(parsed.values())[0]
    else:
        data = parsed
    
    for i in data:
        L.append(i['name'])
        
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)



# Convert cast (top 3 actors)

def convert_cast(obj):
    L = []
    count = 0
    
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
            count += 1
            
    return L


movies['cast'] = movies['cast'].apply(convert_cast)



# Extract director

def director(obj):
    L = []

    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            L.append(i['name'])
            break
            
    return L


movies['crew'] = movies['crew'].apply(director)



# Convert overview

movies['overview'] = movies['overview'].apply(lambda x:x.split())



# Create tags column

movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)



# New dataframe

new_movies = movies[
[
"movie_id",
"title",
"tags"
]
].copy()


# Convert list into string

new_movies['tags'] = new_movies['tags'].apply(
    lambda x:" ".join(x)
)


# Convert text into numbers

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    max_features=10000,
    stop_words="english"
)

vectors = cv.fit_transform(
    new_movies['tags']
).toarray()



# Similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)



# Recommendation function

def recommend(movie):

    index = new_movies[
        new_movies['title']==movie
    ].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x:x[1]
    )[1:6]


    for i in movie_list:
        print(new_movies.iloc[i[0]].title)



# Recommendation function

def recommend(movie):

    if movie not in new_movies['title'].values:
        print("Movie not found in database")
        return

    index = new_movies[
        new_movies['title'] == movie
    ].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]


    print("\nRecommended movies for you:\n")

    for i in movie_list:
        print(new_movies.iloc[i[0]].title)



# User input

movie_name = input("Enter movie name: ")
year = input("Enter year of release: ")

recommend(movie_name,year)
