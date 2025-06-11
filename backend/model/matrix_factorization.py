from sklearn.decomposition import TruncatedSVD, NMF


class MatrixFactorization:
    def __init__(self, method='nmf', n_components=50, random_state=42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        if method == 'nmf':
            self.model = NMF(n_components=n_components, init='random', random_state=random_state, max_iter=500)
        elif method == 'svd':
            self.model = TruncatedSVD(n_components=n_components, random_state=random_state)
        else:
            raise ValueError("Method must be either 'nmf' or 'svd'")

    def fit_transform(self, user_item_matrix):
        # Fill missing values with 0 (MF requires non-negative entries)
        matrix = user_item_matrix.fillna(0)
        self.user_factors = self.model.fit_transform(matrix)
        self.item_factors = self.model.components_.T
        return self.user_factors, self.item_factors

    def predict(self, user_index, item_index):
        # Dot product of latent factors
        return self.user_factors[user_index].dot(self.item_factors[item_index])
