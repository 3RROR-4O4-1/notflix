import pandas as pd


class ColdStartRecommender:
    def __init__(self, popularity_score_df):
        """
        popularity_score_df: DataFrame with columns 'id' and 'popularity_score'
        """
        self.popularity_score = popularity_score_df.set_index('id')

    def recommend(self, top_n=10):
        # Recommend the top-n most popular items
        recs = self.popularity_score.sort_values('popularity_score', ascending=False).head(top_n)
        return list(recs.index)
