import numpy as np
import pandas as pd


def get_popularity_recs(df: pd.DataFrame, idx: int, top_k: int = 5) -> list:
    """Индексы top_k туров того же города, отсортированных по hotel_stars DESC."""
    city = df.iloc[idx]["city_id"]
    return (
        df[(df["city_id"] == city) & (df.index != idx)]
        .sort_values("hotel_stars", ascending=False)
        .index[:top_k]
        .tolist()
    )


def get_similar_tours(
    df: pd.DataFrame,
    tour_id: str,
    sim_matrix: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """Возвращает top_k туров, похожих на tour_id, по заданной матрице схожести."""
    idx     = df.index[df["id"] == tour_id][0]
    scores  = sim_matrix[idx]
    top_idx = np.argsort(scores)[::-1][1:top_k + 1]
    result  = df.iloc[top_idx][["name", "city_name", "category", "hotel_stars", "meal_type", "price"]].copy()
    result["similarity"] = scores[top_idx].round(3)
    return result.reset_index(drop=True)
