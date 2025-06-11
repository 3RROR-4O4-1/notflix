import pandas as pd

def load_tmdb(file_path):
    """Load TMDB dataset from CSV."""
    return pd.read_csv(file_path)

def load_imdb(basics_path, ratings_path, names_path):
    """Load IMDb datasets from TSV files."""
    imdb_basics = pd.read_csv(basics_path, sep='\t', na_values="\\N", low_memory=False)
    imdb_ratings = pd.read_csv(ratings_path, sep='\t', na_values="\\N", low_memory=False)
    imdb_names = pd.read_csv(names_path, sep='\t', na_values="\\N", low_memory=False)
    return imdb_basics, imdb_ratings, imdb_names

def load_movielens(links_path, movies_path, ratings_path, tags_path):
    """Load MovieLens datasets from CSV files."""
    ml_links = pd.read_csv(links_path)
    ml_movies = pd.read_csv(movies_path)
    ml_ratings = pd.read_csv(ratings_path)
    ml_tags = pd.read_csv(tags_path)
    return ml_links, ml_movies, ml_ratings, ml_tags