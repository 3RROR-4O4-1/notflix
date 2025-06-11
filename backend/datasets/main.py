import os
import pandas as pd
from datetime import datetime
import logging
import traceback
from scipy import sparse

from data_loader import load_tmdb, load_imdb, load_movielens
from data_cleaner import clean_tmdb, clean_imdb, clean_movielens
from feature_engineering import (
    scale_numeric_features, fill_numeric_features, encode_genres, preprocess_text,
    vectorize_text, transform_date, build_user_item_matrix_sparse
)
from utils import reduce_dimensionality_nmf
from additional_criteria import (
    extract_popularity_ratings, compute_popularity_score,
    analyze_release_dates_trends, aggregate_user_behavior,
    profile_user_preferences, calculate_diversity_novelty
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.dynamic_user_item_matrix import dynamic_matrix


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Loading datasets...")
        tmdb = load_tmdb('raw/tmdb/tmdb_data.csv')
        imdb_basics, imdb_ratings, imdb_names = load_imdb(
            'raw/imdb/title.basics.tsv', 'raw/imdb/title.ratings.tsv', 'raw/imdb/name.basics.tsv'
        )
        ml_links, ml_movies, ml_ratings, ml_tags = load_movielens(
            'raw/movielens/ml_100k/links.csv', 'raw/movielens/ml_100k/movies.csv', 'raw/movielens/ml_100k/ratings.csv', 'raw/movielens/ml_100k/tags.csv'
        )
        logger.info("Datasets loaded successfully.")
    except Exception as e:
        logger.error("Error loading datasets: " + str(e))
        logger.error(traceback.format_exc())
        raise

    try:
        logger.info("Cleaning datasets...")
        tmdb = clean_tmdb(tmdb)
        imdb_basics, imdb_ratings, imdb_names = clean_imdb(imdb_basics, imdb_ratings, imdb_names)
        ml_links, ml_movies, ml_ratings, ml_tags = clean_movielens(ml_links, ml_movies, ml_ratings, ml_tags)
        logger.info("Datasets cleaned.")
    except Exception as e:
        logger.error("Error cleaning datasets: " + str(e))
        logger.error(traceback.format_exc())
        raise

    try:
        logger.info("Merging datasets...")
        # Using an inner join to retain only movies listed in links.csv.
        ml_merged = ml_movies.merge(ml_links, on='movieId', how='inner')
        # Merging MovieLens and TMDB datasets.
        ml_tmdb = ml_merged.merge(tmdb, left_on='tmdbId', right_on='id', how='left', suffixes=('_ml', '_tmdb'))
        # Merging IMDb datasets (we have to convert tconst to numeric to match with imdbId).
        imdb_basics['tconst'] = imdb_basics['tconst'].astype(str)
        imdb_basics['imdbId_numeric'] = imdb_basics['tconst'].str.replace('tt', '').astype(float)
        # Merging
        ml_tmdb['imdbId'] = pd.to_numeric(ml_tmdb['imdbId'], errors='coerce')
        ml_full = ml_tmdb.merge(imdb_basics, left_on='imdbId', right_on='imdbId_numeric', how='left',
                                suffixes=('', '_imdb'))
        logger.info("Datasets merged successfully.")

        # Note that the resulting matrix has redundant columns, I know it is not optimized

    except Exception as e:
        logger.error("Error merging datasets: " + str(e))
        logger.error(traceback.format_exc())
        raise

    try:
        # adjusting the df
        logger.info("Starting feature engineering...")

        if 'runtime' in ml_full.columns:
            ml_full = fill_numeric_features(ml_full, ['runtime'])
        # this is scaled in case I want to filter according to that later on
        if 'revenue' in ml_full.columns:
            ml_full, _ = scale_numeric_features(ml_full, ['revenue'])
        # same for vote average although negative values might affect the system
        if 'vote_average' in ml_full.columns:
            ml_full, _ = scale_numeric_features(ml_full, ['vote_average'])

        # creates the genre matrix and adds it to the df
        ml_full, _ = encode_genres(ml_full, 'genres_ml')

        if 'overview' in ml_full.columns: # comes from the tmdb dataset
            ml_full['overview_clean'] = ml_full['overview'].apply(preprocess_text)
            tfidf_matrix, tfidf_vectorizer = vectorize_text(ml_full['overview_clean'])
        else:
            tfidf_matrix = None
            tfidf_vectorizer = None

        if 'release_date' in ml_full.columns:
            ml_full = transform_date(ml_full, 'release_date')
        elif 'startYear' in ml_full.columns:
            ml_full['release_year'] = ml_full['startYear']
            current_year = datetime.now().year
            ml_full['recency'] = current_year - ml_full['release_year']

        # Building a sparse user-item matrix from the ratings data.
        sparse_matrix, user_categories, movie_categories = build_user_item_matrix_sparse(ml_ratings)
        # Convertion to CSR format.
        sparse_matrix_csr = sparse_matrix.tocsr()
        # NMF with the sparse matrix.
        reduced_user_item_matrix, nmf_model = reduce_dimensionality_nmf(sparse_matrix_csr, n_components=50)
        logger.info("Feature engineering completed.")
    except Exception as e:
        logger.error("Error during feature engineering: " + str(e))
        logger.error(traceback.format_exc())
        raise

    try:
        logger.info("Extracting additional criteria...")
        popularity_df = extract_popularity_ratings(tmdb)
        popularity_score_df = compute_popularity_score(popularity_df)
        release_trends_df = analyze_release_dates_trends(ml_full)
        user_behavior_df = aggregate_user_behavior(ml_ratings)
        user_preferences_df = profile_user_preferences(ml_tags)
        if tfidf_matrix is not None:
            diversity_df = calculate_diversity_novelty(tfidf_matrix, ml_full['id'])
        else:
            diversity_df = None
        logger.info("Additional criteria extracted.")
    except Exception as e:
        logger.error("Error extracting additional criteria: " + str(e))
        logger.error(traceback.format_exc())
        raise

    extracted_features = {
        'merged_data': ml_full,
        'user_item_matrix': sparse_matrix_csr,
        'reduced_user_item_matrix': reduced_user_item_matrix,
        'tfidf_matrix': tfidf_matrix,
        'tfidf_vectorizer': tfidf_vectorizer,
        'popularity_score': popularity_score_df,
        'release_trends': release_trends_df,
        'user_behavior': user_behavior_df,
        'user_preferences': user_preferences_df,
        'diversity_novelty': diversity_df
    }

    logger.info("Data preparation and feature extraction completed successfully.")

    # Ensure the curated folder exists.
    curated_folder = os.path.join("curated")
    if not os.path.exists(curated_folder):
        os.makedirs(curated_folder)

    # Saves merged data.
    merged_data_path = os.path.join(curated_folder, "merged_movie_data.csv")
    extracted_features['merged_data'].to_csv(merged_data_path, index=False)
    logger.info("Saved merged data to %s", merged_data_path)

    # Saves the sparse user-item matrix as NPZ.
    sparse_matrix_path = os.path.join(curated_folder, "user_item_matrix.npz")
    sparse.save_npz(sparse_matrix_path, sparse_matrix_csr)
    logger.info("Saved sparse user-item matrix to %s", sparse_matrix_path)

    # Saves user and movie ID mappings.
    user_ids_path = os.path.join(curated_folder, "user_ids.csv")
    pd.Series(user_categories).to_csv(user_ids_path, index=False, header=False)
    logger.info("Saved user IDs to %s", user_ids_path)

    movie_ids_path = os.path.join(curated_folder, "movie_ids.csv")
    pd.Series(movie_categories).to_csv(movie_ids_path, index=False, header=False)
    logger.info("Saved movie IDs to %s", movie_ids_path)

    # In case, saves a dense version of the user-item matrix.
    try:
        dense_matrix = pd.DataFrame(sparse_matrix_csr.todense(), index=user_categories, columns=movie_categories)
        dense_matrix_path = os.path.join(curated_folder, "user_item_matrix.csv")
        dense_matrix.to_csv(dense_matrix_path, index=True)
        logger.info("Saved dense user-item matrix to %s", dense_matrix_path)
    except Exception as e:
        logger.error("Error saving dense user-item matrix: %s", str(e))

    # Save the popularity scores.
    popularity_scores_path = os.path.join(curated_folder, "popularity_scores.csv")
    extracted_features['popularity_score'].to_csv(popularity_scores_path, index=False)
    logger.info("Saved popularity scores to %s", popularity_scores_path)

    # Save the reduced user-item matrix (from NMF).
    reduced_matrix_path = os.path.join(curated_folder, "reduced_user_item_matrix.csv")
    reduced_df = pd.DataFrame(extracted_features['reduced_user_item_matrix'], index=user_categories)
    reduced_df.to_csv(reduced_matrix_path, index=True)
    logger.info("Saved reduced user-item matrix to %s", reduced_matrix_path)

    # Save additional criteria.
    release_trends_path = os.path.join(curated_folder, "release_trends.csv")
    extracted_features['release_trends'].to_csv(release_trends_path, index=False)
    logger.info("Saved release trends to %s", release_trends_path)

    user_behavior_path = os.path.join(curated_folder, "user_behavior.csv")
    extracted_features['user_behavior'].to_csv(user_behavior_path, index=False)
    logger.info("Saved user behavior to %s", user_behavior_path)

    user_preferences_path = os.path.join(curated_folder, "user_preferences.csv")
    extracted_features['user_preferences'].to_csv(user_preferences_path, index=False)
    logger.info("Saved user preferences to %s", user_preferences_path)

    if extracted_features['diversity_novelty'] is not None:
        diversity_novelty_path = os.path.join(curated_folder, "diversity_novelty.csv")
        extracted_features['diversity_novelty'].to_csv(diversity_novelty_path, index=False)
        logger.info("Saved diversity/novelty scores to %s", diversity_novelty_path)

    logger.info("Curated data saved in folder: %s", curated_folder)
    print("Data preparation and feature extraction complete.")


if __name__ == "__main__":
    main()
