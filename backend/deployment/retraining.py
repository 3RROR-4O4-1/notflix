import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
ARTIFACTS_DIR = os.path.join("..", "models", "trained")
MF_COMPONENTS = 50
DL_ENCODING_DIM = 50
DL_HIDDEN_LAYERS = [200, 100]
DL_EPOCHS = 10 # Reduced for faster retraining cycles in example
DL_BATCH_SIZE = 128
GCN_IN_CHANNELS = 16
GCN_HIDDEN_CHANNELS = 32
GCN_OUT_CHANNELS = 16
GCN_EPOCHS = 20 # Reduced for faster retraining cycles in example
GCN_LR = 0.01

# Ensure artifact directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def retrain_models(user_item_matrix_df):
    """
    Retrains models using the provided user-item matrix DataFrame and saves their artifacts.

    Args:
        user_item_matrix_df (pd.DataFrame): The user-item interaction matrix (dense).
    """
    logger.info(f"Starting retraining pipeline with matrix shape: {user_item_matrix_df.shape}")

    if user_item_matrix_df.empty:
        logger.error("Input user-item matrix is empty. Aborting retraining.")
        return

    # Ensure IDs are consistent (using strings as in the original api_server logic)
    user_ids = user_item_matrix_df.index.astype(str).tolist()
    item_ids = user_item_matrix_df.columns.astype(str).tolist()
    matrix_values = user_item_matrix_df.fillna(0).values
    sparse_matrix = csr_matrix(matrix_values)

    # --- 0. Save IDs ---
    try:
        pd.Series(user_ids).to_csv(os.path.join(ARTIFACTS_DIR, "user_ids.csv"), index=False, header=False)
        pd.Series(item_ids).to_csv(os.path.join(ARTIFACTS_DIR, "item_ids.csv"), index=False, header=False)
        logger.info("Saved user and item IDs.")
    except Exception as e:
        logger.error(f"Error saving user/item IDs: {e}")
        # Decide if we should abort or continue

    # --- 1. Retrain Matrix Factorization (NMF) ---
    logger.info("Retraining Matrix Factorization (NMF)...")
    try:
        from model.matrix_factorization import MatrixFactorization
        mf_model = MatrixFactorization(method='nmf', n_components=MF_COMPONENTS)
        user_factors, item_factors = mf_model.fit_transform(user_item_matrix_df.fillna(0)) # NMF needs dense

        np.save(os.path.join(ARTIFACTS_DIR, "mf_user_factors.npy"), user_factors)
        np.save(os.path.join(ARTIFACTS_DIR, "mf_item_factors.npy"), item_factors)
        logger.info(f"NMF retraining complete. Saved factors.")
    except Exception as e:
        logger.error(f"Error retraining NMF: {e}")

    # --- 2. Retrain Deep Learning (Autoencoder) ---
    logger.info("Retraining Deep Learning (Autoencoder)...")
    try:
        from model.deep_learning import AutoEncoderRecommender
        import tensorflow as tf # Import TF here to avoid loading it if DL fails

        num_items = len(item_ids)
        deep_model_instance = AutoEncoderRecommender(
            num_items,
            encoding_dim=DL_ENCODING_DIM,
            hidden_layers=DL_HIDDEN_LAYERS,
            dropout_rate=0.5
        )
        # Use matrix_values (numpy array) for training
        deep_model_instance.train(
            train_data=matrix_values,
            epochs=DL_EPOCHS,
            batch_size=DL_BATCH_SIZE
        )
        # Save the trained model
        model_path = os.path.join(ARTIFACTS_DIR, "autoencoder_model.keras")
        deep_model_instance.model.save(model_path)
        logger.info(f"Autoencoder retraining complete. Saved model to {model_path}")
        # Clean up GPU memory if needed
        tf.keras.backend.clear_session()
    except ImportError:
        logger.warning("TensorFlow/Keras not found. Skipping Deep Learning retraining.")
    except Exception as e:
        logger.error(f"Error retraining Autoencoder: {e}")

    # --- 3. Retrain Graph-Based (GCN) ---
    logger.info("Retraining Graph-Based (GCN)...")
    try:
        from model.graph_based import GraphBasedGCNRecommender
        import torch # Import torch here

        # GCN expects DataFrame input in its current implementation
        graph_rec = GraphBasedGCNRecommender(
            user_item_matrix_df, # Pass the original DF
            in_channels=GCN_IN_CHANNELS,
            hidden_channels=GCN_HIDDEN_CHANNELS,
            out_channels=GCN_OUT_CHANNELS,
            device='cpu' # Use CPU for potentially broader compatibility, change if GPU available
        )
        graph_rec.train(epochs=GCN_EPOCHS, lr=GCN_LR)

        # Save user and item embeddings
        np.save(os.path.join(ARTIFACTS_DIR, "gcn_user_embeddings.npy"), graph_rec.user_embeddings)
        np.save(os.path.join(ARTIFACTS_DIR, "gcn_item_embeddings.npy"), graph_rec.item_embeddings)
        logger.info("GCN retraining complete. Saved embeddings.")
        del graph_rec # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        logger.warning("PyTorch or PyTorch Geometric not found. Skipping GCN retraining.")
    except Exception as e:
        logger.error(f"Error retraining GCN: {e}")

    # --- 4. Precompute Collaborative Filtering Similarities ---
    logger.info("Precomputing CF similarities...")
    try:
        # User-based similarity
        user_similarity_matrix = cosine_similarity(sparse_matrix)
        np.save(os.path.join(ARTIFACTS_DIR, "cf_user_similarity.npy"), user_similarity_matrix)
        logger.info("Computed and saved user-user similarity matrix.")

        # Item-based similarity
        item_similarity_matrix = cosine_similarity(sparse_matrix.T)
        np.save(os.path.join(ARTIFACTS_DIR, "cf_item_similarity.npy"), item_similarity_matrix)
        logger.info("Computed and saved item-item similarity matrix.")
    except Exception as e:
        logger.error(f"Error computing CF similarities: {e}")

    logger.info("Retraining pipeline completed.")


if __name__ == "__main__":
    logger.info(f"Starting manual retraining script at {datetime.now().isoformat()}")

    # Load the latest data source for manual retraining
    # Option 1: Use the dynamic matrix's persisted state (if saved)
    # Option 2: Load from the curated sparse matrix (might be slightly stale)
    curated_folder = os.path.join("..", "datasets", "curated")
    sparse_matrix_path = os.path.join(curated_folder, "user_item_matrix.npz")
    user_ids_path = os.path.join(curated_folder, "user_ids.csv")
    movie_ids_path = os.path.join(curated_folder, "movie_ids.csv")

    matrix_to_retrain = None
    if os.path.exists(sparse_matrix_path) and os.path.exists(user_ids_path) and os.path.exists(movie_ids_path):
        try:
            logger.info(f"Loading data from {sparse_matrix_path}")
            from scipy.sparse import load_npz
            sparse_matrix_csr = load_npz(sparse_matrix_path)
            user_categories = pd.read_csv(user_ids_path, header=None).squeeze().tolist()
            movie_categories = pd.read_csv(movie_ids_path, header=None).squeeze().tolist()
            # Convert to dense DataFrame for compatibility with current model inputs
            matrix_to_retrain = pd.DataFrame.sparse.from_spmatrix(
                sparse_matrix_csr, index=user_categories, columns=movie_categories
            ).sparse.to_dense().copy()
            logger.info("Loaded data successfully.")
        except Exception as e:
            logger.error(f"Failed to load data from NPZ/CSV: {e}")
    else:
        logger.warning("Could not find preprocessed matrix/IDs. Manual retraining requires data.")

    if matrix_to_retrain is not None:
        # Ensure string IDs for consistency with API server logic during retraining trigger
        matrix_to_retrain.index = matrix_to_retrain.index.astype(str)
        matrix_to_retrain.columns = matrix_to_retrain.columns.astype(str)
        retrain_models(matrix_to_retrain)
    else:
        logger.error("No data matrix available for retraining.")

    logger.info(f"Manual retraining script finished at {datetime.now().isoformat()}")