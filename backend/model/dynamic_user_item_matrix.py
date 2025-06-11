import os
import pandas as pd
from scipy.sparse import csr_matrix

class DynamicUserItemMatrix:
    def __init__(self, initial_matrix=None):
        """
        Initializes the dynamic user-item matrix as a dictionary-of-dictionaries.
        If an initial matrix (a pandas DataFrame) is provided, it is converted to a dict-of-dicts;
        otherwise, an empty dictionary is created.
        """
        if initial_matrix is not None:
            # Convert the provided dense DataFrame to a dictionary-of-dictionaries.
            self.matrix = initial_matrix.to_dict(orient='index')
        else:
            self.matrix = {}

    def update(self, user_id, item_id, rating):
        """
        Updates the matrix with a new rating.
        If the user does not exist, creates a new entry.
        """
        if user_id not in self.matrix:
            self.matrix[user_id] = {}
        self.matrix[user_id][item_id] = rating
        return self.matrix

    def get_matrix(self):
        """
        Returns the current dynamic matrix as a dense pandas DataFrame.
        Missing ratings are filled with 0.
        """
        df = pd.DataFrame.from_dict(self.matrix, orient='index')
        return df.fillna(0)

    def get_sparse_matrix(self):
        """
        Returns a CSR sparse matrix representation of the dynamic matrix.
        """
        dense_df = self.get_matrix()
        return csr_matrix(dense_df.values)

# Global instance of the dynamic user-item matrix.
static_matrix_path = os.path.join("..", "datasets", "curated", "user_item_matrix.csv")
if os.path.exists(static_matrix_path):
    initial_matrix = pd.read_csv(static_matrix_path, index_col=0)
else:
    initial_matrix = None

dynamic_matrix = DynamicUserItemMatrix(initial_matrix=initial_matrix)
