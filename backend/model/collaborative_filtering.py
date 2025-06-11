import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, user_item_matrix, user_ids=None, movie_ids=None):
        """
        Initializes user-based collaborative filtering with a sparse user-item matrix.

        Parameters:
          user_item_matrix: A sparse matrix (preferably in CSR format) with shape (num_users, num_movies).
          user_ids: List of user IDs corresponding to the rows.
          movie_ids: List of movie IDs corresponding to the columns.
        """
        self.user_item_matrix = user_item_matrix
        self.user_ids = list(user_ids) if user_ids is not None else list(range(user_item_matrix.shape[0]))
        self.movie_ids = list(movie_ids) if movie_ids is not None else list(range(user_item_matrix.shape[1]))

        # Compute cosine similarity for users.
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=self.user_ids,
            columns=self.user_ids
        )

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_ids:
            return None
        user_index = self.user_ids.index(user_id)
        user_similarities = self.similarity_df.loc[user_id]
        # Exclude self-similarity and take top 10 similar users.
        similar_users = user_similarities.drop(user_id).sort_values(ascending=False).head(10)
        recommendations = {}
        # Build a mapping from movie ID to column index.
        movie_to_index = {mid: i for i, mid in enumerate(self.movie_ids)}

        for similar_user, sim_score in similar_users.iteritems():
            similar_user_index = self.user_ids.index(similar_user)
            # Get the ratings vector for the similar user.
            row = self.user_item_matrix.getrow(similar_user_index).toarray().ravel()
            for idx, rating in enumerate(row):
                if rating > 0:
                    movie_id = self.movie_ids[idx]
                    recommendations[movie_id] = recommendations.get(movie_id, 0) + sim_score * rating
        # Remove items already rated by the target user.
        target_row = self.user_item_matrix.getrow(user_index).toarray().ravel()
        recommendations = {
            movie: score
            for movie, score in recommendations.items()
            if target_row[movie_to_index[movie]] == 0
        }
        recommended_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return list(recommended_items)


class ItemBasedCF:
    def __init__(self, user_item_matrix, user_ids=None, movie_ids=None):
        """
        Initializes item-based collaborative filtering with a sparse user-item matrix.

        Parameters:
          user_item_matrix: A sparse matrix (CSR format) with shape (num_users, num_movies).
          user_ids: List of user IDs corresponding to the rows.
          movie_ids: List of movie IDs corresponding to the columns.
        """
        self.user_item_matrix = user_item_matrix
        self.user_ids = list(user_ids) if user_ids is not None else list(range(user_item_matrix.shape[0]))
        self.movie_ids = list(movie_ids) if movie_ids is not None else list(range(user_item_matrix.shape[1]))

        # Compute cosine similarity between items using the transpose.
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.movie_ids,
            columns=self.movie_ids
        )

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_ids:
            return None
        user_index = self.user_ids.index(user_id)
        # Get the user's ratings vector.
        user_ratings = self.user_item_matrix.getrow(user_index).toarray().ravel()
        recommendations = {}
        movie_to_index = {mid: i for i, mid in enumerate(self.movie_ids)}
        for idx, rating in enumerate(user_ratings):
            if rating > 0:
                movie_id = self.movie_ids[idx]
                similar_items = self.item_similarity_df[movie_id].sort_values(ascending=False).head(10)
                for sim_movie, sim_score in similar_items.iteritems():
                    if user_ratings[movie_to_index[sim_movie]] == 0:
                        recommendations[sim_movie] = recommendations.get(sim_movie, 0) + sim_score * rating
        recommended_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return recommended_items
