from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # проверяем, что корреляция существует или DataFrame не пустой
    assert "age" in corr.columns or not corr.empty

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_on_sample_df():
    df = _sample_df()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert 0.0 <= flags["quality_score"] <= 1.0
    assert "age" in missing_df.index
    assert missing_df.loc["age", "missing_count"] == 1


def test_quality_flags_constant_and_high_cardinality():
    df = pd.DataFrame(
        {
            "const_col": [1, 1, 1, 1],
            "cat_col": [f"val_{i}" for i in range(150)],
            "num": range(150),
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    # Проверка constant columns
    assert "has_constant_columns" in flags
    assert flags["has_constant_columns"] is True
    assert flags["constant_columns"] == ["const_col"]

    # Проверка high-cardinality categoricals
    assert "has_high_cardinality_categoricals" in flags
    assert flags["has_high_cardinality_categoricals"] is True
    assert flags["high_cardinality_columns"] == ["cat_col"]
