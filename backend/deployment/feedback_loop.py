import os
import pandas as pd
import logging
import datetime
from model.adaptive_feedback import AdaptiveFeedbackModule
from model.dynamic_user_item_matrix import DynamicUserItemMatrix

# --- Load Movie Metadata for Genre Information ---

# Corrected file path: include 'datasets'
metadata_file_path = os.path.join("..", "datasets", "curated", "merged_movie_data.csv")
if os.path.exists(metadata_file_path):
    metadata_df = pd.read_csv(metadata_file_path)
    logging.info("Loaded merged movie metadata from %s", metadata_file_path)
    # Use 'movieId' as the index if it exists; otherwise fallback to 'id'
    if 'movieId' in metadata_df.columns:
        metadata_df.set_index('movieId', inplace=True)
    elif 'id' in metadata_df.columns:
        metadata_df.set_index('id', inplace=True)
    else:
        logging.warning("No movieId or id column found in metadata. Genre-based recommendations will not work properly.")
    # Determine which genre column to use.
    if 'genres' in metadata_df.columns:
        genre_column = 'genres'
    elif 'genres_ml' in metadata_df.columns:
        genre_column = 'genres_ml'
    else:
        logging.warning("No genre column found in metadata. Genre-based recommendations will not work properly.")
        genre_column = None
else:
    logging.warning("Merged movie metadata file not found at %s. Genre-based recommendations will be disabled.", metadata_file_path)
    metadata_df = pd.DataFrame()
    genre_column = None

def get_movie_genres(item_id):
    """
    Returns a list of genres for a given movie ID.
    It splits genres into separate strings, ensuring that genres are processed correctly.
    """
    try:
        if metadata_df.empty or genre_column is None:
            return []
        if item_id in metadata_df.index:
            genres_str = metadata_df.loc[item_id, genre_column]
            if pd.isna(genres_str) or not genres_str:
                return []
            # Split genres by either '|' or ',' to handle different formats.
            genres = []
            if "|" in genres_str:
                genres = [g.strip() for g in genres_str.split("|") if g.strip()]
            elif "," in genres_str:
                genres = [g.strip() for g in genres_str.split(",") if g.strip()]
            return genres
        else:
            return []
    except Exception as e:
        logging.error("Error retrieving genres for item_id %s: %s", item_id, e)
        return []

# --- Initialize the Dynamic User-Item Matrix ---
# This in-memory matrix will be updated every time new feedback is recorded.
dynamic_matrix = DynamicUserItemMatrix()


# --- Feedback Loop with Genre Profile, Adaptive Feedback, and Dynamic Matrix Updates ---

class FeedbackLoop:
    def __init__(self, feedback_file_path="feedback.csv"):
        """
        Initializes the feedback loop.

        Parameters:
        - feedback_file_path: Path to store the feedback CSV file.
          In production, this could be a database or more robust storage.
        """
        self.feedback_file_path = feedback_file_path
        # Load existing feedback if available, else create an empty DataFrame.
        if os.path.exists(self.feedback_file_path):
            self.feedback_df = pd.read_csv(self.feedback_file_path)
        else:
            self.feedback_df = pd.DataFrame(columns=["user_id", "item_id", "feedback", "timestamp"])
        # Initialize the adaptive feedback module.
        self.adaptive_module = AdaptiveFeedbackModule()
        # Initialize a per-user genre profile dictionary.
        self.user_genre_profile = {}
        logging.info("FeedbackLoop initialized with file: %s", self.feedback_file_path)

    def record_feedback(self, user_id, item_id, feedback):
        """
        Record feedback from a user on an item.

        Parameters:
        - user_id: Identifier for the user.
        - item_id: Identifier for the item.
        - feedback: A numerical value indicating feedback adjustment
                    (e.g., +1 for positive, -1 for negative, or a rating value).
        """
        timestamp = datetime.datetime.now().isoformat()
        new_entry = pd.DataFrame({
            "user_id": [user_id],
            "item_id": [item_id],
            "feedback": [feedback],
            "timestamp": [timestamp]
        })
        self.feedback_df = pd.concat([self.feedback_df, new_entry], ignore_index=True)
        # Persist feedback to file.
        self.feedback_df.to_csv(self.feedback_file_path, index=False)
        # Update the adaptive feedback module.
        self.adaptive_module.update(user_id, {item_id: feedback})
        # Update the user's genre profile.
        movie_genres = get_movie_genres(item_id)
        if user_id not in self.user_genre_profile:
            self.user_genre_profile[user_id] = {}
        for genre in movie_genres:
            # Update the genre score by adding the feedback.
            self.user_genre_profile[user_id][genre] = self.user_genre_profile[user_id].get(genre, 0) + feedback

        # --- New: Update the dynamic user-item matrix ---
        # This call dynamically adds or updates the user's rating in the in-memory matrix.
        dynamic_matrix.update(user_id, item_id, feedback)

        logging.info("Feedback recorded: user %s, item %s, feedback %s", user_id, item_id, feedback)
        logging.info("Updated genre profile for user %s: %s", user_id, self.user_genre_profile.get(user_id, {}))

    def process_feedback(self):
        """
        Processes the collected feedback. This function can be scheduled to run periodically
        to perform operations such as:
          - Aggregating feedback statistics.
          - Updating ensemble weights.
          - Triggering retraining based on significant feedback changes.

        For demonstration, it prints a summary of average feedback per user-item pair.
        """
        if self.feedback_df.empty:
            logging.info("No feedback to process.")
            return
        summary = self.feedback_df.groupby(["user_id", "item_id"])["feedback"].mean().reset_index()
        logging.info("Feedback summary:\n%s", summary.to_string(index=False))
        # Additional processing (e.g., triggering model retraining) can be added here.

    def get_feedback(self):
        """
        Returns the current feedback DataFrame.
        """
        return self.feedback_df

    def get_user_genre_profile(self, user_id):
        """
        Returns the stored genre profile for a given user.
        """
        return self.user_genre_profile.get(user_id, {})


# --- For Testing Purposes ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Define a folder for curated feedback if needed.
    curated_folder = os.path.join("..", "curated")
    if not os.path.exists(curated_folder):
        os.makedirs(curated_folder)
    feedback_file = os.path.join(curated_folder, "feedback.csv")

    feedback_loop = FeedbackLoop(feedback_file_path=feedback_file)

    # Record some dummy feedback entries.
    feedback_loop.record_feedback(user_id=1, item_id=101, feedback=1.0)
    feedback_loop.record_feedback(user_id=2, item_id=102, feedback=-0.5)
    feedback_loop.record_feedback(user_id=1, item_id=105, feedback=0.5)

    # Process and log the feedback summary.
    feedback_loop.process_feedback()

    # For demonstration, print out the current dynamic matrix.
    current_matrix = dynamic_matrix.get_matrix()
    logging.info("Current Dynamic User-Item Matrix:\n%s", current_matrix)
