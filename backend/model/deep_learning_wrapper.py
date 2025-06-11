from model.deep_learning import AutoEncoderRecommender


class DeepLearningWrapper:
    """
    A wrapper for the deep learning recommender (AutoEncoderRecommender) that provides
    a consistent interface with a recommend(user_id, top_n=10) method.

    Expects a user-item DataFrame with user IDs as the index and item IDs as the columns.
    """

    def __init__(self, model: AutoEncoderRecommender, user_item_df):
        self.model = model
        self.user_item_df = user_item_df

    def recommend(self, user_id, top_n=10):
        # Return an empty list if the user is not present in the DataFrame.
        if user_id not in self.user_item_df.index:
            return []
        # Retrieve the user's rating vector.
        user_vector = self.user_item_df.loc[user_id].values
        # Generate recommendations using the deep learning model.
        recs = self.model.recommend(user_vector, top_n=top_n)
        return recs
from model.deep_learning import AutoEncoderRecommender

class DeepLearningWrapper:
    """
    A wrapper for the deep learning recommender (AutoEncoderRecommender) that provides
    a consistent interface with a recommend(user_id, top_n=10) method.

    Expects a user-item DataFrame with user IDs as the index and item IDs as the columns.
    """
    def __init__(self, model: AutoEncoderRecommender, user_item_df):
        self.model = model
        self.user_item_df = user_item_df

    def recommend(self, user_id, top_n=10):
        # Return an empty list if the user is not present in the DataFrame.
        if user_id not in self.user_item_df.index:
            return []
        # Retrieve the user's rating vector.
        user_vector = self.user_item_df.loc[user_id].values
        # Generate recommendations using the deep learning model.
        recs = self.model.recommend(user_vector, top_n=top_n)
        return recs
