import numpy as np
import pandas as pd
import pytest

from src.models.metrics import (
    catalog_coverage,
    category_hit_rate,
    intra_list_diversity,
    mean_similarity,
    personalization,
)


@pytest.fixture
def block_df():
    """12 туров: 3 категории по 4 — для тестов с предсказуемым ответом."""
    rows = []
    for cat in ["luxury", "comfort", "budget"]:
        for i in range(4):
            rows.append({"id": f"{cat}-{i}", "category": cat})
    return pd.DataFrame(rows)


@pytest.fixture
def anti_block_sim():
    """Анти-блочная матрица: каждый тур похож только на туры ДРУГОЙ категории.
    luxury(0-3)→comfort(4-7), comfort→budget(8-11), budget→luxury.
    Используется для проверки CHR=0.0."""
    n = 12
    sim = np.zeros((n, n))
    np.fill_diagonal(sim, 1.0)
    sim[0:4, 4:8] = 0.99    # luxury → comfort
    sim[4:8, 8:12] = 0.99   # comfort → budget
    sim[8:12, 0:4] = 0.99   # budget → luxury
    return sim


@pytest.fixture
def block_sim(block_df):
    """Блочная матрица: self=1.0, внутри блока=0.99, между блоками=0.0.
    Self выше всех соседей → argsort детерминирован, self всегда на позиции 0."""
    n = len(block_df)
    sim = np.zeros((n, n))
    np.fill_diagonal(sim, 1.0)
    for start in range(0, n, 4):
        block = slice(start, start + 4)
        sim[block, block] = 0.99
        np.fill_diagonal(sim[block, block], 1.0)
    return sim


class TestCategoryHitRate:
    def test_perfect_block_matrix(self, block_df, block_sim):
        # Топ-3 для каждого тура — внутри блока (та же категория)
        assert category_hit_rate(block_df, block_sim, top_k=3) == pytest.approx(1.0)

    def test_anti_block_matrix_gives_zero(self, block_df, anti_block_sim):
        # Топ-3 для каждого тура — из ДРУГОЙ категории → CHR должен быть 0.0
        assert category_hit_rate(block_df, anti_block_sim, top_k=3) == pytest.approx(0.0)

    def test_in_valid_range(self, mini_df, mini_sim):
        val = category_hit_rate(mini_df, mini_sim, top_k=3)
        assert 0.0 <= val <= 1.0


class TestCatalogCoverage:
    def test_full_coverage_with_block_matrix(self, block_sim):
        # Все 12 туров появятся в чужих топ-3 → coverage = 1.0
        assert catalog_coverage(block_sim, top_k=3) == pytest.approx(1.0)

    def test_in_valid_range(self, mini_sim):
        assert 0.0 <= catalog_coverage(mini_sim, top_k=3) <= 1.0


class TestMeanSimilarity:
    def test_ones_matrix(self):
        # Матрица единиц: топ-3 все с sim=1.0
        sim = np.ones((6, 6))
        assert mean_similarity(sim, top_k=3) == pytest.approx(1.0)

    def test_in_valid_range(self, mini_sim):
        assert 0.0 <= mean_similarity(mini_sim, top_k=3) <= 1.0


class TestIntraListDiversity:
    def test_identical_recommendations_give_zero(self):
        # Все пары внутри топа sim=1.0 → dist=0 → ILD=0.0
        sim = np.ones((6, 6))
        assert intra_list_diversity(sim, top_k=3) == pytest.approx(0.0)

    def test_top_k_one_no_crash(self, mini_sim):
        # top_k=1 → нет пар → не должен падать
        val = intra_list_diversity(mini_sim, top_k=1)
        assert isinstance(val, float)

    def test_in_valid_range(self, mini_sim):
        assert 0.0 <= intra_list_diversity(mini_sim, top_k=3) <= 1.0


class TestPersonalization:
    def test_in_valid_range(self, mini_sim):
        assert 0.0 <= personalization(mini_sim, top_k=3) <= 1.0

    def test_more_overlap_means_lower_personalization(self):
        # sim_a: строки 0–5 все рекомендуют {6,7,8} → максимальное перекрытие
        n, top_k = 9, 3
        sim_a = np.zeros((n, n))
        sim_a[:, 6] = 0.9; sim_a[:, 7] = 0.8; sim_a[:, 8] = 0.7
        np.fill_diagonal(sim_a, 1.0)  # diagonal > 0.9 → self всегда исключается первым

        # sim_b: у каждой строки уникальный набор соседей (циклический сдвиг) → низкое перекрытие
        sim_b = np.zeros((n, n))
        for i in range(n):
            sim_b[i, (i + 1) % n] = 0.9
            sim_b[i, (i + 2) % n] = 0.8
            sim_b[i, (i + 3) % n] = 0.7
        np.fill_diagonal(sim_b, 1.0)

        assert personalization(sim_a, top_k=top_k) < personalization(sim_b, top_k=top_k)
