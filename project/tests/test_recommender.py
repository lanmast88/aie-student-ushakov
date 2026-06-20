import pytest

from src.models.recommender import get_popularity_recs, get_similar_tours


class TestGetSimilarTours:
    def test_returns_exactly_top_k(self, mini_df, mini_sim):
        result = get_similar_tours(mini_df, "lux-par", mini_sim, top_k=3)
        assert len(result) == 3

    def test_query_not_in_recommendations(self, mini_df, mini_sim):
        result = get_similar_tours(mini_df, "com-par", mini_sim, top_k=5)
        assert "com-par" not in result["id"].values

    def test_similarity_descending(self, mini_df, mini_sim):
        result = get_similar_tours(mini_df, "lux-par", mini_sim, top_k=5)
        sims = result["similarity"].tolist()
        assert sims == sorted(sims, reverse=True)

    def test_similarity_in_valid_range(self, mini_df, mini_sim):
        result = get_similar_tours(mini_df, "bud-pra", mini_sim, top_k=5)
        assert (result["similarity"] >= 0.0).all()
        assert (result["similarity"] <= 1.0).all()

    def test_required_columns_present(self, mini_df, mini_sim):
        result = get_similar_tours(mini_df, "lux-par", mini_sim, top_k=3)
        required = {"id", "name", "city_name", "category", "hotel_stars", "meal_type", "price", "similarity"}
        assert required.issubset(set(result.columns))

    def test_top_k_one(self, mini_df, mini_sim):
        result = get_similar_tours(mini_df, "com-dub", mini_sim, top_k=1)
        assert len(result) == 1

    def test_top_k_larger_than_catalog_returns_n_minus_1(self, mini_df, mini_sim):
        # При top_k > n-1 возвращает все туры кроме самого запроса
        result = get_similar_tours(mini_df, "lux-par", mini_sim, top_k=999)
        assert len(result) == len(mini_df) - 1

    def test_unknown_tour_id_raises(self, mini_df, mini_sim):
        with pytest.raises(IndexError):
            get_similar_tours(mini_df, "nonexistent-id", mini_sim, top_k=3)


class TestGetPopularityRecs:
    def test_all_recommendations_same_city(self, mini_df):
        idx = mini_df.index[mini_df["id"] == "lux-par"].item()
        cands = get_popularity_recs(mini_df, idx, top_k=3)
        cities = mini_df.iloc[cands]["city_id"].tolist()
        assert all(c == "paris" for c in cities)

    def test_query_not_in_results(self, mini_df):
        idx = mini_df.index[mini_df["id"] == "com-par"].item()
        cands = get_popularity_recs(mini_df, idx, top_k=5)
        assert idx not in cands

    def test_sorted_by_stars_descending(self, mini_df):
        idx = mini_df.index[mini_df["id"] == "bud-par"].item()
        cands = get_popularity_recs(mini_df, idx, top_k=3)
        stars = mini_df.iloc[cands]["hotel_stars"].tolist()
        assert stars == sorted(stars, reverse=True)

    def test_respects_top_k(self, mini_df):
        idx = mini_df.index[mini_df["id"] == "lux-dub"].item()
        cands = get_popularity_recs(mini_df, idx, top_k=1)
        assert len(cands) <= 1
