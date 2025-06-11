import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def extract_popularity_ratings(df):
    """
    Extracts vote_average and vote_count from the dataframe.
    df must contain columns: 'id', 'vote_average', 'vote_count'.
    """
    return df[['id', 'vote_average', 'vote_count']].copy()


def compute_popularity_score(df, rating_weight=0.7, vote_weight=0.3):
    """
    Computes a popularity score as a weighted combination of vote_average and normalized vote_count.
    Normalization scales vote_count to [0, 10] ( Careful ! This is assuming vote_average is on a 10-point scale).
    """
    df = df.copy()
    max_votes = df['vote_count'].max()
    df['normalized_vote_count'] = df['vote_count'] / max_votes if max_votes > 0 else 0
    df['popularity_score'] = rating_weight * df['vote_average'] + vote_weight * df['normalized_vote_count'] * 10
    return df[['id', 'popularity_score']]

def analyze_release_dates_trends(df):
    """
    Analyze release dates to extract trends.
    Computes season (based on month) and a simple buzz score inversely proportional to the movie's age.
    """
    df = df.copy()
    df['release_month'] = df['release_date'].dt.month
    df['season'] = df['release_month'].apply(lambda x: 'Winter' if x in [12, 1, 2]
                                             else ('Spring' if x in [3, 4, 5]
                                                   else ('Summer' if x in [6, 7, 8]
                                                         else 'Fall')))
    current_year = datetime.now().year
    df['buzz_score'] = 1 / (current_year - df['release_year'] + 1)
    return df[['id', 'release_year', 'season', 'buzz_score']]

def aggregate_user_behavior(ratings_df):
    """
    Aggregate historical user behavior from ratings.
    Returns average rating and total count of ratings per user.
    """
    user_behavior = ratings_df.groupby('userId').agg(
        avg_rating=('rating', 'mean'),
        count_ratings=('rating', 'count')
    ).reset_index()
    return user_behavior

def profile_user_preferences(tags_df):
    """
    Aggregate user tags to profile preferences.
    Returns a list of unique tags per user.
    """
    user_tags = tags_df.groupby('userId')['tag'].apply(lambda tags: list(set(tags))).reset_index()
    return user_tags

def calculate_diversity_novelty(tfidf_matrix, movie_ids, similarity_threshold=0.5):
    """
    Calculate a novelty score for each movie based on TF-IDF similarity.
    A lower average similarity to all other movies suggests higher novelty.
    """
    similarity_matrix = cosine_similarity(tfidf_matrix)
    novelty_scores = []
    num_movies = similarity_matrix.shape[0]
    for i in range(num_movies):
        similar_count = (similarity_matrix[i] < similarity_threshold).sum()
        novelty_scores.append(similar_count / (num_movies - 1))
    diversity_df = pd.DataFrame({'id': movie_ids, 'novelty_score': novelty_scores})
    return diversity_df
