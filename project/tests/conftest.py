import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from src.models.features import build_feature_matrices


@pytest.fixture(scope="session")
def mini_df():
    """9 туров: 3 категории × 3 города — минимальная база для unit-тестов."""
    rows = []
    for cat, stars, price in [("luxury", 5, 5000), ("comfort", 4, 1500), ("budget", 3, 600)]:
        for city, desc in [
            ("paris",  "Роскошный отель Париж Эйфелева башня гастрономия haute cuisine"),
            ("dubai",  "Дубай пляж небоскрёб Бурдж-Халифа VIP люкс"),
            ("prague", "Прага исторический центр замок Карлов мост пиво"),
        ]:
            rows.append({
                "id":             f"{cat[:3]}-{city[:3]}",
                "name":           f"{cat.title()} {city.title()}",
                "city_id":        city,
                "city_name":      city.title(),
                "category":       cat,
                "hotel_stars":    stars,
                "meal_type":      "breakfast",
                "price":          float(price),
                "duration_nights": 7,
                "description":    desc,
            })
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture(scope="session")
def mini_sim(mini_df):
    struct_m, tfidf_m, _, _ = build_feature_matrices(mini_df)
    sim_s = cosine_similarity(struct_m)
    sim_t = cosine_similarity(tfidf_m)
    return 0.7 * sim_s + 0.3 * sim_t
