from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]
    constant_columns: Optional[List[str]] = None
    high_cardinality_columns: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
            "constant_columns": self.constant_columns or [],
            "high_cardinality_columns": self.high_cardinality_columns or [],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []
    constant_cols: List[str] = []
    high_card_cols: List[str] = []
    high_card_threshold = 100

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val = max_val = mean_val = std_val = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        if unique <= 1:
            constant_cols.append(name)

        if (
            (ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype))
            and unique > high_card_threshold
        ):
            high_card_cols.append(name)

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        constant_columns=constant_cols,
        high_cardinality_columns=high_card_cols,
    )


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    return (
        pd.DataFrame({"missing_count": total, "missing_share": share})
        .sort_values("missing_share", ascending=False)
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}

    cat_cols = [
        col
        for col in df.columns
        if ptypes.is_object_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]

    for name in cat_cols[:max_columns]:
        vc = df[name].value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue

        share = vc / vc.sum()
        result[name] = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}

    # --- базовые эвристики ---
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = (
        float(missing_df["missing_share"].max())
        if not missing_df.empty
        else 0.0
    )
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # --- constant columns ---
    constant_columns = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]
    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns

    # --- high-cardinality categoricals ---
    high_card_threshold = 100
    high_cardinality_columns = [
        col
        for col in df.select_dtypes(include=["object", "category"]).columns
        if df[col].nunique(dropna=True) > high_card_threshold
    ]
    flags["has_high_cardinality_categoricals"] = (
        len(high_cardinality_columns) > 0
    )
    flags["high_cardinality_columns"] = high_cardinality_columns

    # --- quality score ---
    score = 1.0
    score -= max_missing_share

    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.1

    flags["quality_score"] = max(0.0, min(1.0, score))
    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )

    return pd.DataFrame(rows)
