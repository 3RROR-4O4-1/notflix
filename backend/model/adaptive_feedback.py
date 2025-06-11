from collections import defaultdict


class AdaptiveFeedbackModule:
    def __init__(self):
        """
        Initializes the module to store feedback adjustments.
        feedback_dict is a dictionary mapping user_id to another dictionary
        that maps item_id to the accumulated feedback adjustment.
        """
        self.feedback_dict = defaultdict(dict)

    def update(self, user_id, feedback):
        """
        Update the stored feedback for a given user.

        Parameters:
        - user_id: The user identifier.
        - feedback: A dictionary of {item_id: rating_adjustment}.
                    The adjustment can be positive (indicating preference) or negative.
        """
        for item_id, adjustment in feedback.items():
            # If feedback for the item already exists, sum the adjustments.
            if item_id in self.feedback_dict[user_id]:
                self.feedback_dict[user_id][item_id] += adjustment
            else:
                self.feedback_dict[user_id][item_id] = adjustment

    def recommend(self, user_id, base_recommendations=None, top_n=10):
        """
        Generate updated recommendations for a user by adjusting base recommendation scores
        using the stored feedback.

        Parameters:
        - user_id: The user for whom recommendations are generated.
        - base_recommendations: Optional list of tuples (item_id, base_score) from another recommender.
                                If provided, the stored feedback adjustment is added to the base score.
                                If not provided, the method returns items with positive feedback sorted by feedback.
        - top_n: Number of recommendations to return.

        Returns:
        - A list of tuples (item_id, adjusted_score) sorted by descending adjusted_score.
        """
        user_feedback = self.feedback_dict.get(user_id, {})
        if base_recommendations is not None:
            # Adjust each base recommendation by adding the user's feedback (if any)
            adjusted_recs = []
            for item_id, base_score in base_recommendations:
                adjustment = user_feedback.get(item_id, 0)
                adjusted_score = base_score + adjustment
                adjusted_recs.append((item_id, adjusted_score))
            # Sort recommendations by adjusted score (highest first)
            adjusted_recs.sort(key=lambda x: x[1], reverse=True)
            return adjusted_recs[:top_n]
        else:
            # If no base recommendations are provided, return items with positive feedback sorted by the feedback score.
            positive_feedback = [(item_id, score) for item_id, score in user_feedback.items() if score > 0]
            positive_feedback.sort(key=lambda x: x[1], reverse=True)
            return positive_feedback[:top_n]
