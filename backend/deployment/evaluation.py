import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


def compute_rmse(y_true, y_pred):
    """Compute the root mean squared error."""
    return sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true, y_pred):
    """Compute the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def precision_recall_at_k(recommendations, test_set, k=10):
    """
    Calculate precision and recall at K.

    Parameters:
    - recommendations: dict mapping user_id -> list of recommended item_ids.
    - test_set: dict mapping user_id -> set of relevant (ground truth) item_ids.
    - k: the number of top recommendations to consider.

    Returns:
    - avg_precision, avg_recall: average precision and recall over users.
    """
    precisions = []
    recalls = []
    for user, recs in recommendations.items():
        recs_k = set(recs[:k])
        true_items = test_set.get(user, set())
        if not true_items:
            continue
        num_relevant = len(recs_k.intersection(true_items))
        precision = num_relevant / k
        recall = num_relevant / len(true_items)
        precisions.append(precision)
        recalls.append(recall)
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    return avg_precision, avg_recall


def evaluate_model(recommender, test_data, rating_threshold=3.5, top_k=10):
    """
    Evaluates a recommender model on a test dataset.

    Parameters:
    - recommender: A recommender model that implements:
         - predict(users, items): returns predicted ratings.
         - recommend(user, top_n): returns list of tuples (item_id, score).
    - test_data: A DataFrame with columns ['userId', 'movieId', 'rating'].
    - rating_threshold: Threshold above which an item is considered relevant.
    - top_k: Number of top recommendations to consider for ranking metrics.

    Returns:
    - A dictionary with evaluation metrics: RMSE, MAE, Precision@K, Recall@K.
    """
    # Extract arrays from test_data
    users = test_data['userId'].values
    items = test_data['movieId'].values
    true_ratings = test_data['rating'].values

    # Predict ratings (assuming recommender.predict exists)
    try:
        predicted_ratings = recommender.predict(users, items)
    except Exception as e:
        print("Error: The recommender does not implement predict().", e)
        predicted_ratings = np.zeros_like(true_ratings)

    rmse = compute_rmse(true_ratings, predicted_ratings)
    mae = compute_mae(true_ratings, predicted_ratings)

    # Build ground truth for ranking metrics: for each user, items with rating >= threshold.
    test_ground_truth = {}
    for _, row in test_data.iterrows():
        user = row['userId']
        item = row['movieId']
        rating = row['rating']
        if rating >= rating_threshold:
            if user in test_ground_truth:
                test_ground_truth[user].add(item)
            else:
                test_ground_truth[user] = {item}

    # Generate recommendations for each user in test_ground_truth
    recommendations = {}
    for user in test_ground_truth.keys():
        try:
            recs = recommender.recommend(user, top_n=top_k)
            # Assume recs is a list of tuples (item_id, score)
            recommendations[user] = [item for item, score in recs]
        except Exception as e:
            recommendations[user] = []

    precision, recall = precision_recall_at_k(recommendations, test_ground_truth, k=top_k)

    return {"RMSE": rmse, "MAE": mae, "Precision@K": precision, "Recall@K": recall}


if __name__ == "__main__":
    # Example usage: load a test dataset and a dummy recommender model for evaluation.

    # For illustration, assume we have a test CSV file in the curated folder:
    try:
        test_data = pd.read_csv("../dataset/curated/test_ratings.csv")
    except Exception as e:
        print("Test dataset not found, creating dummy test data.")
        # Create dummy test data: 5 users, 10 items, random ratings between 1 and 5.
        dummy_data = {
            "userId": np.random.randint(1, 6, 50),
            "movieId": np.random.randint(1, 11, 50),
            "rating": np.random.uniform(1, 5, 50)
        }
        test_data = pd.DataFrame(dummy_data)


    # Dummy recommender: returns constant predictions and dummy recommendations.
    class DummyRecommender:
        def predict(self, users, items):
            return np.full(len(users), 3.5)

        def recommend(self, user, top_n=10):
            # Return dummy recommendations: items 1 to top_n with score 3.5
            return [(i, 3.5) for i in range(1, top_n + 1)]


    dummy_model = DummyRecommender()
    metrics = evaluate_model(dummy_model, test_data)
    print("Evaluation Metrics:", metrics)
