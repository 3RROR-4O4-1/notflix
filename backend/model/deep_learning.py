import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class AutoEncoderRecommender:
    def __init__(self, num_items, encoding_dim=50, hidden_layers=[200, 100], dropout_rate=0.5):
        """
        Autoencoder Recommender using a dense autoencoder architecture.

        Parameters:
        - num_items: The number of items (i.e., input dimension equals the total number of items).
        - encoding_dim: Dimension of the latent space (bottleneck).
        - hidden_layers: List containing the number of units for each hidden layer in the encoder.
        - dropout_rate: Dropout rate used for regularization.
        """
        self.num_items = num_items
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        # Input layer: each user is represented by a vector of item ratings
        input_layer = Input(shape=(self.num_items,), name="input")

        # Encoder: create hidden layers based on hidden_layers list
        x = input_layer
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        encoded = Dense(self.encoding_dim, activation='relu', name="encoded")(x)

        # Decoder: mirror the encoder layers (you could also have a symmetric architecture)
        x = encoded
        for units in reversed(self.hidden_layers):
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        output_layer = Dense(self.num_items, activation='linear', name="reconstructed")(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train(self, train_data, epochs=50, batch_size=128, validation_data=None):
        """
        Trains the autoencoder.

        Parameters:
        - train_data: Numpy array of shape (num_users, num_items) representing user ratings.
        - epochs: Number of training epochs.
        - batch_size: Batch size for training.
        - validation_data: Optional tuple (val_data) for validation.
        """
        self.model.fit(train_data, train_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(validation_data, validation_data) if validation_data is not None else None,
                       verbose=1)

    def get_encoded(self, data):
        """
        Returns the latent representation (encoded vector) of the input data.
        """
        encoder = Model(self.model.input, self.model.get_layer("encoded").output)
        return encoder.predict(data)

    def predict(self, data):
        """
        Reconstructs the input data using the autoencoder.
        """
        return self.model.predict(data)

    def recommend(self, user_vector, top_n=10):
        """
        Generates recommendations for a single user based on their ratings.

        Parameters:
        - user_vector: 1D numpy array of shape (num_items,) representing the user's ratings.
                       Unrated items should be 0.
        - top_n: Number of recommendations to return.

        Returns:
        - List of tuples (item_index, predicted_rating) for the top_n items not already rated.
        """
        reconstructed = self.predict(np.array([user_vector]))[0]
        # Identify items the user has not rated (assume 0 means unrated)
        unrated_indices = np.where(user_vector == 0)[0]
        recommendations = [(i, reconstructed[i]) for i in unrated_indices]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
