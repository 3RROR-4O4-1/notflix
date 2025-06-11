import pandas as pd
import re
import string
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
import nltk


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def scale_numeric_features(df, numeric_cols):
    """ Standardizes numeric features using StandardScaler (to N(0,1)) """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
    return df, scaler

def fill_numeric_features(df, numeric_cols):
    """Pads missing values in specified numeric columns with zeros."""
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

def encode_genres(df, genre_col):
    """Convert genre strings into lists and then one-hot encode them."""
    def process_genres(genres_str):
        if isinstance(genres_str, str):
            return [g.strip() for g in re.split(r'[|,]', genres_str) if g.strip()]
        return []
    df[genre_col + '_list'] = df[genre_col].apply(process_genres)

    # One-hot encoding of genres
    mlb = MultiLabelBinarizer() # Initializes MultiLabelBinarizer object (used to transform a list of labels intoa binary format)
    genre_encoded = pd.DataFrame(mlb.fit_transform(df[genre_col + '_list']),
                                 columns=["genre_" + g for g in mlb.classes_],
                                 index=df.index)
    df = pd.concat([df, genre_encoded], axis=1)
    return df, mlb

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def vectorize_text(corpus, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def transform_date(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['release_year'] = df[date_col].dt.year
    current_year = datetime.now().year
    df['recency'] = current_year - df['release_year']
    return df

def build_user_item_matrix(ratings_df):
    user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
    return user_item_matrix


def build_user_item_matrix_sparse(ratings_df):
    """
    Build a sparse user-item matrix from the ratings DataFrame.
    This function maps userId and movieId to categorical codes and constructs
    a sparse matrix that only stores the actual ratings.

    Parameters:
      ratings_df (DataFrame): DataFrame containing at least 'userId', 'movieId', and 'rating' columns.

    Returns:
      sparse_matrix (scipy.sparse.coo_matrix): Sparse user-item matrix.
      user_categories (Index): The unique user IDs corresponding to the rows.
      movie_categories (Index): The unique movie IDs corresponding to the columns.
    """
    # Convert userId and movieId to categorical types to map them to integer codes
    user_ids = ratings_df['userId'].astype('category')
    movie_ids = ratings_df['movieId'].astype('category')

    # Get the categorical codes which will serve as indices in the sparse matrix
    user_idx = user_ids.cat.codes
    movie_idx = movie_ids.cat.codes

    # Create the sparse matrix using COO format
    sparse_matrix = coo_matrix(
        (ratings_df['rating'], (user_idx, movie_idx)),
        shape=(user_ids.nunique(), movie_ids.nunique())
    )

    # Return the sparse matrix along with the mapping of categorical codes to original IDs
    return sparse_matrix, user_ids.cat.categories, movie_ids.cat.categories
