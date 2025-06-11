import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self, metadata_features, text_embeddings=None):
        """
        metadata_features: DataFrame indexed by movie ID containing numeric features (e.g. genres, popularity, etc.)
        text_embeddings: (Optional) Additional text features.
        """
        self.metadata_features = metadata_features
        self.text_embeddings = text_embeddings
        # (We do not precompute the similarity matrix since we'll compute on the fly for a user profile)

    def recommend(self, user_id, top_n=10):
        """
        Generates recommendations for a user based on a content profile computed as the weighted average
        of the metadata features for the movies the user has rated.

        This method accesses the dynamic user-item matrix to retrieve the user's ratings.
        It then computes a user profile and returns the top_n movies (that the user hasn't already rated)
        sorted by cosine similarity between the user profile and each movie's features.
        """
        # Import dynamic_matrix from our module
        from model.dynamic_user_item_matrix import dynamic_matrix
        
        try:
            # Get the user's ratings from the dynamic matrix (dense DataFrame)
            user_ratings = dynamic_matrix.get_matrix().loc[user_id]
        except KeyError:
            # If the user doesn't exist in the matrix, return empty recommendations
            return []
        
        # Filter to movies with positive ratings
        rated_movies = user_ratings[user_ratings > 0]
        if rated_movies.empty:
            return []
        
        rated_movie_ids = rated_movies.index.tolist()
        
        # Get metadata features for the movies the user has rated.
        try:
            user_movie_features = self.metadata_features.loc[rated_movie_ids]
        except KeyError:
            return []
        
        # Align the ratings with the features
        ratings = rated_movies.loc[user_movie_features.index]
        
        # Compute the user profile as a weighted average of the features
        user_profile = (user_movie_features.multiply(ratings, axis=0)).sum() / ratings.sum()
        
        # Compute cosine similarity between the user profile and all movie features
        user_profile_vector = user_profile.values.reshape(1, -1)
        similarity_scores = cosine_similarity(user_profile_vector, self.metadata_features)[0]
        sim_series = pd.Series(similarity_scores, index=self.metadata_features.index)
        
        # Exclude movies already rated by the user
        sim_series = sim_series.drop(labels=rated_movie_ids, errors='ignore')
        sim_series = sim_series.sort_values(ascending=False)
        recommendations = list(sim_series.head(top_n).items())
        return recommendations
