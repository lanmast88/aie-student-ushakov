import numpy as np
import pandas as pd

from src.models.recommender import get_popularity_recs


def category_hit_rate(df: pd.DataFrame, sim_matrix: np.ndarray, top_k: int = 5) -> float:
    """Доля рекомендаций той же категории (luxury/comfort/budget), что и запрос."""
    hits = []
    for idx in range(len(df)):
        scores    = sim_matrix[idx]
        top_idx   = np.argsort(scores)[::-1][1:top_k + 1]
        query_cat = df.iloc[idx]["category"]
        hits.append((df.iloc[top_idx]["category"] == query_cat).mean())
    return float(np.mean(hits))


def catalog_coverage(sim_matrix: np.ndarray, top_k: int = 5) -> float:
    """Доля туров каталога, которые хотя бы раз попали в рекомендации."""
    n = len(sim_matrix)
    recommended = set()
    for idx in range(n):
        scores  = sim_matrix[idx]
        top_idx = np.argsort(scores)[::-1][1:top_k + 1]
        recommended.update(top_idx)
    return len(recommended) / n


def mean_similarity(sim_matrix: np.ndarray, top_k: int = 5) -> float:
    """Средняя cosine similarity топ-K рекомендаций."""
    scores_list = []
    for idx in range(len(sim_matrix)):
        scores  = sim_matrix[idx]
        top_idx = np.argsort(scores)[::-1][1:top_k + 1]
        scores_list.append(scores[top_idx].mean())
    return float(np.mean(scores_list))


def intra_list_diversity(sim_matrix: np.ndarray, top_k: int = 5) -> float:
    """Среднее попарное расстояние (1 - similarity) внутри топ-K рекомендаций.
    Выше — рекомендации разнообразнее."""
    diversities = []
    for idx in range(len(sim_matrix)):
        top_idx = np.argsort(sim_matrix[idx])[::-1][1:top_k + 1]
        pairs = [(i, j) for ii, i in enumerate(top_idx) for j in top_idx[ii + 1:]]
        if not pairs:
            continue
        dists = [1 - sim_matrix[i, j] for i, j in pairs]
        diversities.append(np.mean(dists))
    return float(np.mean(diversities))


def personalization(sim_matrix: np.ndarray, top_k: int = 5) -> float:
    """1 - среднее попарное пересечение списков рекомендаций между запросами.
    Выше — разным запросам выдаются разные туры."""
    n = len(sim_matrix)
    rec_sets = []
    for idx in range(n):
        top_idx = set(np.argsort(sim_matrix[idx])[::-1][1:top_k + 1])
        rec_sets.append(top_idx)
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap = len(rec_sets[i] & rec_sets[j]) / top_k
            overlaps.append(overlap)
    return float(1 - np.mean(overlaps))


def popularity_metrics(df: pd.DataFrame, sim_matrix: np.ndarray, top_k: int = 5):
    """CHR, Coverage, Mean Similarity, ILD и Personalization для quality baseline.

    sim_matrix используется для расчёта Mean Similarity и ILD (обычно sim_struct).
    """
    hits, recommended, sims, rec_lists = [], set(), [], []
    for idx in range(len(df)):
        cands = get_popularity_recs(df, idx, top_k)
        if not cands:
            continue
        query_cat = df.iloc[idx]["category"]
        hits.append((df.iloc[cands]["category"] == query_cat).mean())
        recommended.update(cands)
        sims.append(sim_matrix[idx, cands].mean())
        rec_lists.append(cands)

    diversities = []
    for cands in rec_lists:
        pairs = [(cands[i], cands[j]) for i in range(len(cands)) for j in range(i + 1, len(cands))]
        if pairs:
            diversities.append(np.mean([1 - sim_matrix[i, j] for i, j in pairs]))
    ild = float(np.mean(diversities)) if diversities else 0.0

    rec_sets = [set(c) for c in rec_lists]
    overlaps = [
        len(rec_sets[i] & rec_sets[j]) / top_k
        for i in range(len(rec_sets))
        for j in range(i + 1, len(rec_sets))
    ]
    pers = float(1 - np.mean(overlaps)) if overlaps else 1.0

    return float(np.mean(hits)), len(recommended) / len(df), float(np.mean(sims)), ild, pers
