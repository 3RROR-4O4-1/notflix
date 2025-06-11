from sklearn.decomposition import NMF

def reduce_dimensionality_nmf(matrix, n_components=50):
    """
    Reduce dimensionality using Non-negative Matrix Factorization (NMF).
    The input matrix must be non-negative (e.g., a user-item rating matrix with missing values filled as 0).
    """
    nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
    reduced_matrix = nmf_model.fit_transform(matrix)
    return reduced_matrix, nmf_model
