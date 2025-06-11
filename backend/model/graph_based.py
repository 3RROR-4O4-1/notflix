import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class BipartiteGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        """
        A simple two-layer GCN model.

        Parameters:
        - num_nodes: Total number of nodes (users + items).
        - in_channels: Input feature dimension.
        - hidden_channels: Hidden layer dimension.
        - out_channels: Output embedding dimension.
        """
        super(BipartiteGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphBasedGCNRecommender:
    def __init__(self, user_item_matrix, in_channels=16, hidden_channels=32, out_channels=16, device='cpu'):
        """
        Initializes the GCN-based recommender.

        Parameters:
        - user_item_matrix: A pandas DataFrame with users as index and items as columns.
        - in_channels: Input feature dimension (will be randomly initialized).
        - hidden_channels: Dimension of the hidden GCN layer.
        - out_channels: Dimension of the output embeddings.
        - device: 'cpu' or 'cuda' for computation.
        """
        self.device = device
        self.user_item_matrix = user_item_matrix.fillna(0)
        # Create mappings for user and item indices in a unified node list
        self.user_ids = list(self.user_item_matrix.index)
        self.item_ids = list(self.user_item_matrix.columns)
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.num_nodes = self.num_users + self.num_items

        # Build bipartite graph edge index and initial node features
        self.edge_index = self._build_edge_index()  # shape [2, num_edges]
        # For simplicity, initialize node features randomly
        self.x = torch.randn((self.num_nodes, in_channels), device=self.device)

        # Build GCN model
        self.model = BipartiteGCN(num_nodes=self.num_nodes, in_channels=in_channels,
                                  hidden_channels=hidden_channels, out_channels=out_channels).to(self.device)

    def _build_edge_index(self):
        """
        Converts the user-item matrix into an edge index for PyTorch Geometric.
        Users are indexed [0, num_users-1] and items are indexed [num_users, num_nodes-1].
        An edge (u, i) is added if there is a non-zero interaction.
        """
        edge_list = []
        # Iterate over each user (row)
        for u_idx, user in enumerate(self.user_ids):
            # For each item (column) with a non-zero interaction
            for item in self.item_ids:
                rating = self.user_item_matrix.loc[user, item]
                if rating > 0:
                    # Map item index: shift by num_users
                    i_idx = self.num_users + self.item_ids.index(item)
                    edge_list.append((u_idx, i_idx))
                    # Since graph is undirected, add reverse edge as well
                    edge_list.append((i_idx, u_idx))
        # Convert edge list to tensor
        if len(edge_list) == 0:
            raise ValueError("No edges found in the user-item matrix.")
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index.to(self.device)

    def train(self, epochs=100, lr=0.01):
        """
        Trains the GCN model to learn node embeddings.
        In this simplified example, we use an unsupervised approach:
        We aim to reconstruct node features using a simple mean squared error loss.
        (In practice, use a loss function suited for link prediction or recommendation.)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.x, self.edge_index)
            # For simplicity, use MSE loss to reconstruct the initial features (self-supervised)
            loss = F.mse_loss(out, self.x)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        self.embeddings = out.detach().cpu().numpy()
        # Split embeddings into user and item parts
        self.user_embeddings = self.embeddings[:self.num_users]
        self.item_embeddings = self.embeddings[self.num_users:]

    def recommend(self, user_id, top_n=10):
        """
        Generates recommendations for a given user using learned embeddings.
        Computes cosine similarity between the user's embedding and all item embeddings.

        Parameters:
        - user_id: The user identifier (as in the original user_item_matrix index).
        - top_n: Number of recommendations to return.

        Returns:
        - A list of tuples (item_id, similarity_score) sorted by descending similarity.
        """
        if not hasattr(self, 'embeddings'):
            raise RuntimeError("Model has not been trained. Call train() before recommending.")

        if user_id not in self.user_ids:
            raise ValueError("User ID not found in the user-item matrix.")

        # Get index of user in our unified node list
        user_index = self.user_ids.index(user_id)
        user_vec = self.user_embeddings[user_index].reshape(1, -1)
        # Compute cosine similarity between user vector and all item embeddings
        sim_scores = cosine_similarity(user_vec, self.item_embeddings)[0]
        # Create list of (item_id, score) and sort
        recs = list(zip(self.item_ids, sim_scores))
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:top_n]
