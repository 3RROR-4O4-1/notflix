import logging
import pandas as pd

class EnsembleHybridRecommender:
    def __init__(self, recommenders, weights=None, cold_start_recommender=None):
        """
        Initializes the Ensemble/Hybrid Recommender.

        Parameters:
        - recommenders: Dictionary mapping recommender names (strings) to recommender objects.
          Each recommender must implement a recommend(user_id, top_n) method that returns a list of (item_id, score) pairs or a pandas Series.
        - weights: Optional dictionary mapping recommender names to weight values (floats).
          If None, equal weights (1.0) are applied to all recommenders.
        - cold_start_recommender: An optional recommender for fallback when no base recommendations are available.
          This recommender should implement a recommend(top_n) method that returns a list of item IDs.
        """
        self.recommenders = recommenders
        if weights is None:
            self.weights = {name: 1.0 for name in recommenders.keys()}
        else:
            self.weights = weights
        self.cold_start_recommender = cold_start_recommender
        self.logger = logging.getLogger(__name__)

    def recommend(self, user_id, top_n=10):
        """
        Aggregates recommendations from all base recommenders.

        For each recommender, retrieves a list of (item_id, score) pairs, multiplies each score by its assigned weight,
        and sums the scores for each item across all recommenders. The final recommendations are sorted in descending order by score.
        """
        combined_scores = {}

        for name, recommender in self.recommenders.items():
            weight = self.weights.get(name, 1.0)
            try:
                recs = recommender.recommend(user_id, top_n=top_n)
                if recs is None:
                    self.logger.warning("Recommender '%s' returned None for user %s", name, user_id)
                    continue
                # If recs is a pandas Series, convert it to list of (key, value) pairs using .items()
                if isinstance(recs, pd.Series):
                    recs = list(recs.items())
                for item, score in recs:
                    combined_scores[item] = combined_scores.get(item, 0) + weight * score
            except Exception as e:
                self.logger.error("Error from recommender '%s' for user %s: %s", name, user_id, str(e))
        
        # Fallback to cold start recommendations if no base recommendations are produced.
        if not combined_scores and self.cold_start_recommender is not None:
            self.logger.info("Falling back to cold start recommendations for user %s", user_id)
            try:
                cold_start_items = self.cold_start_recommender.recommend(top_n=top_n)
                combined_scores = {item: 1.0 for item in cold_start_items}
            except Exception as e:
                self.logger.error("Error obtaining cold start recommendations for user %s: %s", user_id, str(e))
        
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]

    def update_weights(self, new_weights):
        self.weights.update(new_weights)
