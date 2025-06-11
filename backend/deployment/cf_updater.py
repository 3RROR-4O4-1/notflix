# cf_updater.py

import logging
from scipy.sparse import csr_matrix
from model.dynamic_user_item_matrix import DynamicUserItemMatrix

# Assume you have a global dynamic matrix instance (initialized in feedback_loop.py, for example)
# If you haven't imported it already, do so here.
from model.dynamic_user_item_matrix import dynamic_matrix

def get_dynamic_user_item_matrix():
    """
    Returns the current dynamic user-item matrix as a sparse matrix, along with the user and item IDs.
    """
    current_matrix_df = dynamic_matrix.get_matrix()
    if current_matrix_df.empty:
        raise ValueError("Dynamic user-item matrix is empty. No feedback has been recorded yet.")
    sparse_matrix = csr_matrix(current_matrix_df.fillna(0).values)
    user_ids = list(current_matrix_df.index)
    movie_ids = list(current_matrix_df.columns)
    return sparse_matrix, user_ids, movie_ids

def reinitialize_cf_models():
    """
    Reinitializes the CF models (User-Based and Item-Based) using the current dynamic matrix.
    Returns new instances of the CF models.
    """
    try:
        sparse_matrix, user_ids, movie_ids = get_dynamic_user_item_matrix()
        # Import CF models.
        from model.collaborative_filtering import UserBasedCF, ItemBasedCF
        updated_user_cf = UserBasedCF(sparse_matrix, user_ids=user_ids, movie_ids=movie_ids)
        updated_item_cf = ItemBasedCF(sparse_matrix, user_ids=user_ids, movie_ids=movie_ids)
        logging.info("Successfully reinitialized CF models with dynamic matrix (%d users, %d items).", 
                     len(user_ids), len(movie_ids))
        return updated_user_cf, updated_item_cf
    except Exception as e:
        logging.error("Error reinitializing CF models: %s", str(e))
        raise e

# Optionally, if you're using an ensemble recommender, you can update its CF components:
def update_ensemble_cf_models(ensemble_recommender):
    """
    Updates the ensemble recommender's 'user_cf' and 'item_cf' components with reinitialized models.
    """
    updated_user_cf, updated_item_cf = reinitialize_cf_models()
    ensemble_recommender.recommenders['user_cf'] = updated_user_cf
    ensemble_recommender.recommenders['item_cf'] = updated_item_cf
    logging.info("Ensemble recommender's CF models have been updated.")
