import pandas as pd


def clean_tmdb(tmdb):
    """ TMDB dataset """
    tmdb.drop_duplicates(inplace=True)
    tmdb.dropna(subset=['id', 'title', 'release_date', 'genres'], inplace=True)
    tmdb['id'] = pd.to_numeric(tmdb['id'], errors='coerce')
    tmdb['vote_average'] = pd.to_numeric(tmdb['vote_average'], errors='coerce')
    tmdb['vote_count'] = pd.to_numeric(tmdb['vote_count'], errors='coerce')
    tmdb['revenue'] = pd.to_numeric(tmdb['revenue'], errors='coerce')
    tmdb['runtime'] = pd.to_numeric(tmdb['runtime'], errors='coerce')
    tmdb['release_date'] = pd.to_datetime(tmdb['release_date'], errors='coerce')
    tmdb['genres'] = tmdb['genres'].str.lower().str.strip()
    tmdb['title'] = tmdb['title'].str.strip()
    return tmdb


def clean_imdb(imdb_basics, imdb_ratings, imdb_names):
    """ IMDb datasets """
    imdb_basics.drop_duplicates(inplace=True)
    imdb_basics.dropna(subset=['tconst', 'primaryTitle', 'genres'], inplace=True)
    imdb_basics['startYear'] = pd.to_numeric(imdb_basics['startYear'], errors='coerce')
    imdb_basics['runtimeMinutes'] = pd.to_numeric(imdb_basics['runtimeMinutes'], errors='coerce')
    imdb_basics['genres'] = imdb_basics['genres'].str.lower().str.strip()

    imdb_ratings.drop_duplicates(inplace=True)
    imdb_ratings.dropna(inplace=True)
    imdb_ratings['averageRating'] = pd.to_numeric(imdb_ratings['averageRating'], errors='coerce')
    imdb_ratings['numVotes'] = pd.to_numeric(imdb_ratings['numVotes'], errors='coerce')

    imdb_names.drop_duplicates(inplace=True)
    imdb_names.dropna(subset=['nconst', 'primaryName'], inplace=True)

    return imdb_basics, imdb_ratings, imdb_names


def clean_movielens(ml_links, ml_movies, ml_ratings, ml_tags):
    """ MovieLens datasets """
    ml_movies.drop_duplicates(inplace=True)
    ml_movies.dropna(subset=['movieId', 'title', 'genres'], inplace=True)
    ml_movies['genres'] = ml_movies['genres'].str.lower().str.strip()

    ml_ratings.drop_duplicates(inplace=True)
    ml_ratings.dropna(inplace=True)
    ml_ratings['rating'] = pd.to_numeric(ml_ratings['rating'], errors='coerce')
    ml_ratings['movieId'] = pd.to_numeric(ml_ratings['movieId'], errors='coerce')
    ml_ratings['userId'] = pd.to_numeric(ml_ratings['userId'], errors='coerce')

    ml_tags.drop_duplicates(inplace=True)
    ml_tags.fillna('', inplace=True)
    ml_tags['movieId'] = pd.to_numeric(ml_tags['movieId'], errors='coerce')
    ml_tags['userId'] = pd.to_numeric(ml_tags['userId'], errors='coerce')

    ml_links.drop_duplicates(inplace=True)
    ml_links['movieId'] = pd.to_numeric(ml_links['movieId'], errors='coerce')

    return ml_links, ml_movies, ml_ratings, ml_tags
