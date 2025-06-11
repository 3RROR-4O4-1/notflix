import pandas as pd
# Remove the iteritems patch if all uses are updated to .items()
# if not hasattr(pd.Series, "iteritems"):
#     pd.Series.iteritems = pd.Series.items

import os
import logging
import numpy as np # Needed for loading .npy files
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
# Keep sparse matrix loading for initial dynamic matrix setup if needed
from scipy.sparse import csr_matrix, load_npz
import tensorflow as tf # Import TF for loading model

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from deployment.retraining import retrain_models, ARTIFACTS_DIR # Import ARTIFACTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Recommendation API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Instances ---
ensemble_recommender = None
feedback_loop = None
metadata_df = None
# Import the dynamic matrix instance (keeps track of latest ratings)
from model.dynamic_user_item_matrix import dynamic_matrix

# --- Artifact Loading Status ---
# Use a simple dict to track if artifacts for each model were loaded successfully
model_load_status = {
    'user_ids': False, 'item_ids': False, 'mf': False, 'dl': False,
    'gcn': False, 'cf_user': False, 'cf_item': False, 'metadata': False
}

# --- Load Metadata ---
def load_metadata():
    global metadata_df
    curated_folder = os.path.join("..", "datasets", "curated")
    metadata_file_path = os.path.join(curated_folder, "merged_movie_data.csv")
    if os.path.exists(metadata_file_path):
        try:
            metadata_df = pd.read_csv(metadata_file_path, index_col='movieId') # Assuming movieId is the primary ID now
            logger.info("Loaded metadata from merged_movie_data.csv")
            model_load_status['metadata'] = True
            # Ensure index is string if item IDs from artifacts are strings
            if model_load_status['item_ids']: # Check if item_ids were loaded first
                 loaded_item_ids = pd.read_csv(os.path.join(ARTIFACTS_DIR, "item_ids.csv"), header=None).squeeze().astype(str).tolist()
                 if loaded_item_ids:
                     metadata_df.index = metadata_df.index.astype(str) # Align metadata index type
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            metadata_df = pd.DataFrame() # Ensure it's an empty DF on error
    else:
        logger.warning("Metadata file not found. Recommendations might lack detail.")
        metadata_df = pd.DataFrame()


# --- Load Processed Matrix for Dynamic Initialization ---
def initialize_dynamic_matrix():
    """
    Initializes the dynamic user-item matrix, prioritizing saved artifacts if available,
    then falling back to pre-processed NPZ, finally to empty.
    """
    global dynamic_matrix
    curated_folder = os.path.join("..", "datasets", "curated")

    # Check if artifacts exist (means retraining likely ran)
    user_ids_path = os.path.join(ARTIFACTS_DIR, "user_ids.csv")
    item_ids_path = os.path.join(ARTIFACTS_DIR, "item_ids.csv")
    # Check for one of the factor files as a proxy for artifact presence
    mf_user_factors_path = os.path.join(ARTIFACTS_DIR, "mf_user_factors.npy")

    if os.path.exists(user_ids_path) and os.path.exists(item_ids_path) and os.path.exists(mf_user_factors_path):
         logger.info("Found trained model artifacts. Initializing dynamic matrix as empty (will be populated by feedback).")
         # Initialize empty, assuming feedback will populate it, or potentially load latest snapshot if needed
         dynamic_matrix.matrix = {}
         # Load IDs primarily from artifacts now
         try:
             user_ids = pd.read_csv(user_ids_path, header=None).squeeze().astype(str).tolist()
             item_ids = pd.read_csv(item_ids_path, header=None).squeeze().astype(str).tolist()
             model_load_status['user_ids'] = True
             model_load_status['item_ids'] = True
             # Store these IDs somewhere globally accessible if needed, or pass to initializers
             # Dynamic matrix doesn't strictly need them upfront if built from feedback
         except Exception as e:
             logger.error(f"Failed to load IDs from artifacts: {e}. Dynamic matrix may be inconsistent.")

    elif os.path.exists(os.path.join(curated_folder, "user_item_matrix.npz")):
        logger.info(f"Loading pre-processed user-item matrix from NPZ for initial dynamic state.")
        try:
            sparse_matrix_path = os.path.join(curated_folder, "user_item_matrix.npz")
            sparse_matrix_csr = load_npz(sparse_matrix_path)
            orig_user_ids_path = os.path.join(curated_folder, "user_ids.csv")
            orig_movie_ids_path = os.path.join(curated_folder, "movie_ids.csv")
            user_categories = pd.read_csv(orig_user_ids_path, header=None).squeeze().astype(str).tolist() # Use str
            movie_categories = pd.read_csv(orig_movie_ids_path, header=None).squeeze().astype(str).tolist() # Use str

            dense_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix_csr, index=user_categories, columns=movie_categories)
            dynamic_matrix.matrix = dense_df.sparse.to_dense().copy().to_dict(orient='index')
            logger.info("Dynamic user-item matrix initialized successfully from NPZ.")
            # Set status based on IDs loaded from this source
            model_load_status['user_ids'] = True
            model_load_status['item_ids'] = True
        except Exception as e:
             logger.error(f"Failed to load initial state from NPZ: {e}. Initializing empty.")
             dynamic_matrix.matrix = {}
    else:
        logger.warning(f"No artifacts or pre-processed matrix found. Initializing with an empty matrix.")
        dynamic_matrix.matrix = {}

# --- Pydantic model for incoming feedback ---
class FeedbackPayload(BaseModel):
    user_id: str # Assuming string IDs now
    item_id: str # Assuming string IDs now
    feedback: float

# --- Recommendation Loading and Initialization ---
def load_and_initialize_recommenders():
    """
    Loads pre-trained model artifacts and initializes recommender components.
    Does NOT perform training.
    """
    global ensemble_recommender, model_load_status
    logger.info("Loading artifacts and initializing recommenders...")
    recommenders = {}

    # --- Load Common Artifacts: User/Item IDs ---
    user_ids, item_ids = None, None
    try:
        user_ids_path = os.path.join(ARTIFACTS_DIR, "user_ids.csv")
        item_ids_path = os.path.join(ARTIFACTS_DIR, "item_ids.csv")
        if os.path.exists(user_ids_path) and os.path.exists(item_ids_path):
            user_ids = pd.read_csv(user_ids_path, header=None).squeeze().astype(str).tolist()
            item_ids = pd.read_csv(item_ids_path, header=None).squeeze().astype(str).tolist()
            if not user_ids or not item_ids:
                 raise ValueError("Loaded ID lists are empty.")
            model_load_status['user_ids'] = True
            model_load_status['item_ids'] = True
            logger.info(f"Loaded {len(user_ids)} user IDs and {len(item_ids)} item IDs.")
        else:
            raise FileNotFoundError("User or Item ID files not found in artifacts directory.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load user/item IDs: {e}. Many recommenders will fail.")
        # Optionally, try loading from curated folder as fallback if dynamic init used it
        # ... (add fallback logic if needed) ...
        if not model_load_status['user_ids'] or not model_load_status['item_ids']:
            logger.error("Aborting recommender initialization due to missing IDs.")
            return None # Cannot proceed without IDs

    # Reload metadata to ensure index type consistency after IDs are loaded
    load_metadata()

    # --- 1. Matrix Factorization ---
    if model_load_status['user_ids'] and model_load_status['item_ids']:
        try:
            logger.info("Loading NMF artifacts...")
            user_factors_path = os.path.join(ARTIFACTS_DIR, "mf_user_factors.npy")
            item_factors_path = os.path.join(ARTIFACTS_DIR, "mf_item_factors.npy")
            if not os.path.exists(user_factors_path) or not os.path.exists(item_factors_path):
                 raise FileNotFoundError("NMF factor files not found.")

            user_factors = np.load(user_factors_path)
            item_factors = np.load(item_factors_path)

            if user_factors.shape[0] != len(user_ids) or item_factors.shape[0] != len(item_ids):
                 raise ValueError("NMF factor dimensions do not match loaded ID counts.")

            from model.matrix_factorization_wrapper import MatrixFactorizationWrapper
            # Pass dynamic_matrix for checking rated items in recommend()
            mf_rec = MatrixFactorizationWrapper(
                user_factors=user_factors,
                item_factors=item_factors,
                user_ids=user_ids,
                item_ids=item_ids,
                dynamic_matrix_instance=dynamic_matrix # Pass instance
            )
            recommenders['mf'] = mf_rec
            model_load_status['mf'] = True
            logger.info("NMF recommender initialized.")
        except FileNotFoundError as e:
            logger.warning(f"NMF artifacts not found: {e}. MF recommender disabled.")
        except ValueError as e:
            logger.error(f"NMF dimension mismatch: {e}. MF recommender disabled.")
        except Exception as e:
            logger.error(f"Error loading NMF recommender: {e}")

    # --- 2. Deep Learning (Autoencoder) ---
    if model_load_status['user_ids'] and model_load_status['item_ids']:
        try:
            logger.info("Loading Autoencoder artifacts...")
            model_path = os.path.join(ARTIFACTS_DIR, "autoencoder_model.keras")
            if not os.path.exists(model_path):
                 raise FileNotFoundError("Autoencoder model file (.keras) not found.")

            loaded_model = tf.keras.models.load_model(model_path)

            # Verify input shape
            expected_shape = (None, len(item_ids))
            if loaded_model.input_shape != expected_shape:
                raise ValueError(f"Model input shape {loaded_model.input_shape} incompatible with item count {len(item_ids)}")

            from model.deep_learning_wrapper import DeepLearningWrapper
            # Pass dynamic_matrix for getting user vectors in recommend()
            deep_rec = DeepLearningWrapper(
                model=loaded_model,
                user_ids=user_ids,
                item_ids=item_ids,
                dynamic_matrix_instance=dynamic_matrix # Pass instance
            )
            recommenders['deep'] = deep_rec
            model_load_status['dl'] = True
            logger.info("Autoencoder recommender initialized.")
        except FileNotFoundError as e:
            logger.warning(f"Autoencoder artifacts not found: {e}. DL recommender disabled.")
        except ValueError as e:
             logger.error(f"DL model incompatibility: {e}. DL recommender disabled.")
        except ImportError:
             logger.warning("TensorFlow not found. Skipping DL recommender.")
        except Exception as e:
            logger.error(f"Error loading Autoencoder recommender: {e}")

    # --- 3. Graph-Based (GCN) ---
    if model_load_status['user_ids'] and model_load_status['item_ids']:
        try:
            logger.info("Loading GCN artifacts...")
            user_emb_path = os.path.join(ARTIFACTS_DIR, "gcn_user_embeddings.npy")
            item_emb_path = os.path.join(ARTIFACTS_DIR, "gcn_item_embeddings.npy")
            if not os.path.exists(user_emb_path) or not os.path.exists(item_emb_path):
                 raise FileNotFoundError("GCN embedding files not found.")

            user_embeddings = np.load(user_emb_path)
            item_embeddings = np.load(item_emb_path)

            if user_embeddings.shape[0] != len(user_ids) or item_embeddings.shape[0] != len(item_ids):
                 raise ValueError("GCN embedding dimensions do not match loaded ID counts.")

            from model.graph_based import GraphBasedGCNRecommender
            # Modify GraphBasedGCNRecommender to accept embeddings in __init__
            graph_rec = GraphBasedGCNRecommender(
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
                user_ids=user_ids,
                item_ids=item_ids,
                # No need for matrix, channels, etc. if only recommending from embeddings
            )
            recommenders['graph'] = graph_rec
            model_load_status['gcn'] = True
            logger.info("GCN recommender initialized.")
        except FileNotFoundError as e:
            logger.warning(f"GCN artifacts not found: {e}. GCN recommender disabled.")
        except ValueError as e:
             logger.error(f"GCN dimension mismatch: {e}. GCN recommender disabled.")
        except ImportError:
             logger.warning("PyTorch/PyG not found. Skipping GCN recommender.")
        except Exception as e:
            logger.error(f"Error loading GCN recommender: {e}")

    # --- 4. Collaborative Filtering ---
    if model_load_status['user_ids'] and model_load_status['item_ids']:
        try:
            logger.info("Loading CF artifacts...")
            user_sim_path = os.path.join(ARTIFACTS_DIR, "cf_user_similarity.npy")
            item_sim_path = os.path.join(ARTIFACTS_DIR, "cf_item_similarity.npy")
            if not os.path.exists(user_sim_path) or not os.path.exists(item_sim_path):
                raise FileNotFoundError("CF similarity matrix files not found.")

            user_similarity = np.load(user_sim_path)
            item_similarity = np.load(item_sim_path)

            if user_similarity.shape != (len(user_ids), len(user_ids)) or \
               item_similarity.shape != (len(item_ids), len(item_ids)):
                 raise ValueError("CF similarity matrix dimensions do not match loaded ID counts.")

            from model.collaborative_filtering import UserBasedCF, ItemBasedCF
            # Pass dynamic_matrix for getting ratings in recommend()
            user_cf = UserBasedCF(
                similarity_matrix=user_similarity,
                user_ids=user_ids,
                item_ids=item_ids,
                dynamic_matrix_instance=dynamic_matrix
            )
            item_cf = ItemBasedCF(
                similarity_matrix=item_similarity,
                user_ids=user_ids,
                item_ids=item_ids,
                dynamic_matrix_instance=dynamic_matrix
            )
            # No need to wrap recommend if CF methods are updated to return list/handle Series internally

            recommenders['user_cf'] = user_cf
            recommenders['item_cf'] = item_cf
            model_load_status['cf_user'] = True
            model_load_status['cf_item'] = True
            logger.info("CF recommenders initialized.")
        except FileNotFoundError as e:
            logger.warning(f"CF artifacts not found: {e}. CF recommenders disabled.")
        except ValueError as e:
             logger.error(f"CF dimension mismatch: {e}. CF recommenders disabled.")
        except Exception as e:
            logger.error(f"Error loading CF recommenders: {e}")

    # --- 5. Content-Based ---
    if model_load_status['metadata'] and model_load_status['item_ids']:
        try:
            from model.content_based import ContentBasedFiltering
            if not metadata_df.empty:
                # Select numeric features from metadata, ensure index matches item_ids type (string)
                metadata_numeric = metadata_df.select_dtypes(include=['number']).fillna(0)
                metadata_numeric.index = metadata_numeric.index.astype(str)
                # Check if metadata index contains all item_ids
                missing_meta = [item for item in item_ids if item not in metadata_numeric.index]
                if missing_meta:
                    logger.warning(f"{len(missing_meta)} item IDs missing from metadata. Content-based might be incomplete.")
                    # Optional: Add placeholder rows for missing items if needed
                    # placeholder_df = pd.DataFrame(0, index=missing_meta, columns=metadata_numeric.columns)
                    # metadata_numeric = pd.concat([metadata_numeric, placeholder_df])

                # Ensure metadata is aligned with item_ids order if necessary, though lookup by ID is typical
                metadata_aligned = metadata_numeric.reindex(item_ids).fillna(0)


                content_rec = ContentBasedFiltering(
                    metadata_features=metadata_aligned, # Pass aligned metadata
                    item_ids=item_ids, # Pass item IDs
                    dynamic_matrix_instance=dynamic_matrix # Pass for user ratings
                )
                recommenders['content'] = content_rec
                logger.info("Content-based recommender initialized.")
            else:
                 logger.warning("Metadata is empty. Content-based recommender disabled.")
        except Exception as e:
            logger.error(f"Error loading Content-based recommender: {e}")

    # --- 6. Cold Start ---
    cold_start_recommender = None
    try:
        curated_folder = os.path.join("..", "datasets", "curated")
        popularity_scores_path = os.path.join(curated_folder, "popularity_scores.csv")
        if os.path.exists(popularity_scores_path):
            # Assuming 'id' column needs to be string to match item_ids
            popularity_df = pd.read_csv(popularity_scores_path, index_col=0).reset_index()
            # Determine the correct ID column name ('index', 'movieId', 'id'?)
            id_col_name = 'index' # Adjust if needed based on actual CSV
            if id_col_name not in popularity_df.columns:
                 # Try other common names
                 if 'movieId' in popularity_df.columns: id_col_name = 'movieId'
                 elif 'id' in popularity_df.columns: id_col_name = 'id'
                 else: raise ValueError("Cannot find ID column in popularity_scores.csv")

            popularity_df = popularity_df.rename(columns={id_col_name: 'id'})
            popularity_df['id'] = popularity_df['id'].astype(str) # Ensure string IDs

            from model.cold_start import ColdStartRecommender
            cold_start_recommender = ColdStartRecommender(popularity_df)
            logger.info("Cold start recommender initialized.")
        else:
            logger.warning("Popularity scores file not found. Cold start fallback disabled.")
    except Exception as e:
        logger.error(f"Error initializing Cold start recommender: {e}")


    # --- Build Ensemble ---
    if not recommenders:
         logger.error("CRITICAL: No recommenders were successfully initialized. API will not function.")
         return None

    from model.ensemble_hybrid import EnsembleHybridRecommender
    ensemble = EnsembleHybridRecommender(
        recommenders,
        cold_start_recommender=cold_start_recommender
    )
    logger.info("Ensemble recommender initialized with models: %s", list(recommenders.keys()))
    return ensemble

# --- Initialize Feedback Loop ---
def initialize_feedback_loop():
    """ Initializes the feedback loop. """
    try:
        from deployment.feedback_loop import FeedbackLoop
        # Path relative to api_server.py location
        feedback_file = os.path.join("..", "datasets", "curated", "feedback.csv")
        fb_loop = FeedbackLoop(feedback_file_path=feedback_file)
        logger.info("Feedback loop initialized with file: %s", feedback_file)
        return fb_loop
    except Exception as e:
        logger.error("Error initializing feedback loop: %s", str(e))
        # Decide if this is critical. Maybe allow server to run without feedback?
        return None

# --- Initial Load on Startup ---
initialize_dynamic_matrix() # Initialize dynamic matrix first (loads IDs if available)
load_metadata() # Load metadata
ensemble_recommender = load_and_initialize_recommenders() # Load models using artifacts
feedback_loop = initialize_feedback_loop()

# --- API Endpoints ---

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackPayload):
    """ Records user feedback, updates dynamic matrix, and triggers retraining. """
    if feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback system not available.")
    if not model_load_status['metadata'] or metadata_df.empty:
         logger.warning(f"Received feedback for item {feedback.item_id} but metadata is missing.")
         # Allow feedback, but context might be lost
    elif feedback.item_id not in metadata_df.index:
         # Log warning but still accept feedback if item exists in dynamic matrix or IDs
         logger.warning(f"Feedback received for item {feedback.item_id} which is not in metadata.")
         # Ensure item_id is valid based on loaded item_ids if available
         if model_load_status['item_ids']:
             loaded_item_ids = pd.read_csv(os.path.join(ARTIFACTS_DIR, "item_ids.csv"), header=None).squeeze().astype(str).tolist()
             if feedback.item_id not in loaded_item_ids:
                  raise HTTPException(status_code=400, detail=f"Item ID {feedback.item_id} is not known.")
    try:
        # Record feedback and update in-memory matrix
        feedback_loop.record_feedback(feedback.user_id, feedback.item_id, feedback.feedback)
        dynamic_matrix.update(feedback.user_id, feedback.item_id, feedback.feedback)

        # --- Trigger Retraining and Reload ---
        # This remains synchronous for simplicity, but consider async/background task
        logger.info("Feedback recorded. Triggering model retraining and reloading.")
        status = await run_retraining_and_reload()
        return status

    except HTTPException as he:
        raise he # Re-raise validation errors
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing feedback: {e}")

async def run_retraining_and_reload():
     """Synchronously runs retraining and reloads models."""
     global ensemble_recommender
     try:
        logger.info("Getting current dynamic matrix for retraining...")
        current_matrix_df = dynamic_matrix.get_matrix()
        # Ensure IDs are strings before passing to retraining
        current_matrix_df.index = current_matrix_df.index.astype(str)
        current_matrix_df.columns = current_matrix_df.columns.astype(str)

        if current_matrix_df.empty:
            # This might happen if feedback is for new users/items only before matrix is populated
            logger.warning("Dynamic user-item matrix is currently empty. Skipping retraining.")
            return {"status": "warning", "message": "Feedback recorded, but matrix empty. Retraining skipped."}

        logger.info("Starting retraining pipeline...")
        retrain_models(current_matrix_df) # Pass the current matrix

        logger.info("Retraining finished. Reloading models...")
        new_ensemble = load_and_initialize_recommenders()
        if new_ensemble:
            ensemble_recommender = new_ensemble # Atomically update the global instance
            logger.info("Models reloaded successfully.")
            return {"status": "success", "message": "Feedback recorded and models updated."}
        else:
            logger.error("Failed to reload models after retraining. Keeping old models.")
            # The server continues with the potentially stale model
            return {"status": "error", "message": "Feedback recorded, but failed to update models."}

     except Exception as e:
        logger.error(f"Error during retraining/reload: {e}", exc_info=True)
        # Don't raise HTTPException here if the primary goal (recording feedback) succeeded.
        # Return an error status instead.
        return {"status": "error", "message": f"Feedback recorded, but error during model update: {e}"}


@app.get("/recommendations/{user_id}", response_model=dict)
def get_recommendations(user_id: str, top_n: int = 5): # User ID as string
    """ Returns top N recommendations for a given user. """
    if ensemble_recommender is None:
        raise HTTPException(status_code=503, detail="Recommender system not initialized or failed to load.")
    # Check if user exists in the system (based on loaded IDs)
    if model_load_status['user_ids']:
        loaded_user_ids = pd.read_csv(os.path.join(ARTIFACTS_DIR, "user_ids.csv"), header=None).squeeze().astype(str).tolist()
        if user_id not in loaded_user_ids and user_id not in dynamic_matrix.matrix:
             # User is unknown both in original data and new feedback
             logger.warning(f"User ID {user_id} not found in known users. Attempting cold start/default.")
             # Ensemble should handle this via cold_start_recommender if configured

    try:
        recs = ensemble_recommender.recommend(user_id, top_n=top_n) # Pass string ID

        # Filter recommendations to ensure they have metadata if possible
        filtered_recs = []
        if not metadata_df.empty and model_load_status['metadata']:
            valid_metadata_ids = set(metadata_df.index.astype(str)) # Ensure string comparison
            for item_id, score in recs:
                 item_id_str = str(item_id) # Ensure item_id is string
                 if item_id_str in valid_metadata_ids:
                    # Convert item_id back to int ONLY for the final JSON response if required by client
                    try:
                        response_item_id = int(item_id_str)
                    except ValueError:
                        response_item_id = item_id_str # Keep as string if not convertible
                    filtered_recs.append({'item_id': response_item_id, 'score': float(score)})
                 else:
                     logger.debug(f"Recommendation {item_id_str} excluded due to missing metadata.")
            # If filtering removed all recommendations, fall back to returning unfiltered ones? Or stick to filtered?
            # Let's return only those with metadata for consistency.
            if not filtered_recs and recs: # If filtering left nothing, but we had recs
                logger.warning(f"No recommendations for user {user_id} had available metadata.")
                # Optionally return unfiltered recs here:
                # filtered_recs = [{'item_id': str(item_id), 'score': float(score)} for item_id, score in recs]

        elif recs: # If metadata is not available, return raw recommendations
             logger.warning("Metadata not available, returning recommendations without filtering.")
             filtered_recs = [{'item_id': str(item_id), 'score': float(score)} for item_id, score in recs]


        if not filtered_recs:
            # Use 404 if no recommendations could be generated *at all*, maybe 204 (No Content) if just filtered out?
            # Let's stick to 404 for simplicity meaning "no suitable recommendations found".
            raise HTTPException(status_code=404, detail="No recommendations available for this user.")

        return {'user_id': user_id, 'recommendations': filtered_recs}

    except Exception as ex:
        logger.error(f"Error generating recommendations for user {user_id}: {ex}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error generating recommendations: {ex}")


if __name__ == "__main__":
    logger.info("Starting Recommendation API Server...")
    # Check if essential models loaded
    if ensemble_recommender is None:
         logger.critical("Ensemble recommender failed to initialize. API might not serve recommendations.")
    # Add more checks as needed
    uvicorn.run(app, host="0.0.0.0", port=8000)