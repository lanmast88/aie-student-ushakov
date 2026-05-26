import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def build_feature_matrices(df: pd.DataFrame):
    """Строит struct_matrix и tfidf_matrix из датафрейма активных туров.

    df должен быть pre-filtered (только active, reset_index выполнен).

    Returns:
        struct_matrix: np.ndarray (n, 19)  — one-hot + нормированные числовые
        tfidf_matrix:  np.ndarray (n, 200) — TF-IDF по описаниям
        scaler:        fitted MinMaxScaler
        tfidf:         fitted TfidfVectorizer
    """
    cat_features = pd.get_dummies(df[["city_id", "category", "meal_type"]], dtype=float)

    scaler = MinMaxScaler()
    num_features = pd.DataFrame(
        scaler.fit_transform(
            df[["hotel_stars", "duration_nights"]].assign(log_price=np.log1p(df["price"]))
        ),
        columns=["hotel_stars", "duration_nights", "log_price"],
    )

    struct_matrix = pd.concat([cat_features, num_features], axis=1).values

    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["description"]).toarray()

    return struct_matrix, tfidf_matrix, scaler, tfidf
