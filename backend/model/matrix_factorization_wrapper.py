from model.matrix_factorization import MatrixFactorization


class MatrixFactorizationWrapper:
    """
    A wrapper for the matrix factorization recommender that provides a consistent interface:
    recommend(user_id, top_n=10). This wrapper uses the underlying MatrixFactorization model (e.g., NMF or SVD)
    to predict scores for items that the user has not yet rated.

    It expects a user-item DataFrame with user IDs as the index and item IDs as the columns.
    """

    def __init__(self, user_item_df, method='nmf', n_components=50):
        self.user_item_df = user_item_df
        # Fill missing values with zeros to ensure a complete matrix.
        self.model = MatrixFactorization(method=method, n_components=n_components)
        self.user_factors, self.item_factors = self.model.fit_transform(user_item_df.fillna(0))
        self.user_ids = user_item_df.index.tolist()
        self.movie_ids = user_item_df.columns.tolist()

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_ids:
            return []
        user_index = self.user_ids.index(user_id)
        scores = {}
        # Iterate over all items and score those that are unrated by the user (assumed to be 0).
        for idx, movie in enumerate(self.movie_ids):
            if self.user_item_df.loc[user_id, movie] == 0:
                pred = self.user_factors[user_index].dot(self.item_factors[idx])
                scores[movie] = pred
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]
