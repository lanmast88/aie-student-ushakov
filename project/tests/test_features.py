import numpy as np
import pandas as pd
import pytest

from src.models.features import build_feature_matrices


def test_struct_matrix_shape(mini_df):
    struct_m, _, _, _ = build_feature_matrices(mini_df)
    n = len(mini_df)
    n_onehot = (
        mini_df["city_id"].nunique()
        + mini_df["category"].nunique()
        + mini_df["meal_type"].nunique()
    )
    assert struct_m.shape == (n, n_onehot + 3)  # +3: hotel_stars, duration_nights, log_price


def test_tfidf_matrix_row_count(mini_df):
    _, tfidf_m, _, _ = build_feature_matrices(mini_df)
    assert tfidf_m.shape[0] == len(mini_df)


def test_numerical_cols_in_unit_range(mini_df):
    # MinMaxScaler нормирует числовые признаки в [0, 1]
    struct_m, _, _, _ = build_feature_matrices(mini_df)
    num_part = struct_m[:, -3:]
    assert num_part.min() >= 0.0
    assert num_part.max() <= 1.0


def test_tfidf_rows_l2_normalized(mini_df):
    # TfidfVectorizer с norm='l2' по умолчанию
    _, tfidf_m, _, _ = build_feature_matrices(mini_df)
    norms = np.linalg.norm(tfidf_m, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_fitted_objects_can_transform(mini_df):
    row = mini_df.iloc[[0]]
    _, _, scaler, tfidf_vec = build_feature_matrices(mini_df)
    # Должны трансформировать без ошибок
    scaler.transform(row[["hotel_stars", "duration_nights"]].assign(
        log_price=np.log1p(row["price"])
    ))
    tfidf_vec.transform(row["description"])


def test_single_row_no_crash():
    # MinMaxScaler на одной строке не должен падать (хотя нормирует в 0)
    df = pd.DataFrame([{
        "city_id": "paris", "category": "luxury", "meal_type": "breakfast",
        "hotel_stars": 5, "duration_nights": 7, "price": 5000.0,
        "description": "Роскошный отель в Париже с видом на Эйфелеву башню",
    }])
    struct_m, tfidf_m, _, _ = build_feature_matrices(df)
    assert struct_m.shape[0] == 1
    assert tfidf_m.shape[0] == 1
