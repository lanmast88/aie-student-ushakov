import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.recommender import get_similar_tours

_PROJECT_ROOT = Path(__file__).parents[2]
_CONFIG_PATH = Path(os.getenv("CONFIG_PATH", _PROJECT_ROOT / "configs" / "config.yaml"))

with _CONFIG_PATH.open() as _f:
    _cfg = yaml.safe_load(_f)

logging.basicConfig(
    level=getattr(logging, _cfg["service"]["log_level"]),
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = _PROJECT_ROOT / _cfg["artifacts_dir"]
_DEFAULT_TOP_K: int = _cfg["model"]["top_k"]

app = FastAPI(title="Tour Recommender API", version="1.0.0")

_tours_df: pd.DataFrame
_sim_matrix: np.ndarray


@app.on_event("startup")
def load_artifacts() -> None:
    global _tours_df, _sim_matrix

    logger.info("Config: %s", _CONFIG_PATH)
    logger.info("Loading artifacts from %s", _ARTIFACTS_DIR)
    tours = json.loads((_ARTIFACTS_DIR / "tours_active.json").read_text())
    _tours_df = pd.DataFrame(tours)
    _sim_matrix = np.load(_ARTIFACTS_DIR / "sim_combined.npy")
    logger.info(
        "Loaded %d tours, sim_matrix %s, default top_k=%d",
        len(_tours_df),
        _sim_matrix.shape,
        _DEFAULT_TOP_K,
    )


class PredictRequest(BaseModel):
    tour_id: str
    top_k: int = _DEFAULT_TOP_K


class TourRecommendation(BaseModel):
    id: str
    name: str
    city_name: str
    category: str
    hotel_stars: int
    meal_type: str
    price: float
    similarity: float


class PredictResponse(BaseModel):
    tour_id: str
    recommendations: list[TourRecommendation]
    elapsed_ms: float


@app.get("/tours")
def list_tours(city: str | None = None, category: str | None = None) -> list[dict]:
    df = _tours_df
    if city:
        df = df[df["city_name"].str.lower() == city.lower()]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    return df[["id", "name", "city_name", "category", "hotel_stars", "price"]].to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "tours_loaded": len(_tours_df),
        "sim_matrix_shape": list(_sim_matrix.shape),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()

    if req.tour_id not in _tours_df["id"].values:
        logger.warning("Unknown tour_id=%s", req.tour_id)
        raise HTTPException(status_code=404, detail=f"tour_id '{req.tour_id}' not found")

    top_k = max(1, min(req.top_k, 20))
    recs_df = get_similar_tours(_tours_df, req.tour_id, _sim_matrix, top_k=top_k)

    recommendations = []
    for _, row in recs_df.iterrows():
        recommendations.append(
            TourRecommendation(
                id=row["id"],
                name=row["name"],
                city_name=row["city_name"],
                category=row["category"],
                hotel_stars=int(row["hotel_stars"]),
                meal_type=row["meal_type"],
                price=float(row["price"]),
                similarity=float(row["similarity"]),
            )
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "predict tour_id=%s top_k=%d → %d recs in %.1fms",
        req.tour_id,
        top_k,
        len(recommendations),
        elapsed_ms,
    )
    return PredictResponse(
        tour_id=req.tour_id,
        recommendations=recommendations,
        elapsed_ms=elapsed_ms,
    )
