import pytest
from fastapi.testclient import TestClient

from src.service.app import app


@pytest.fixture(scope="module")
def client():
    # TestClient как контекстный менеджер запускает startup-событие (загрузка артефактов)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def any_tour_id(client):
    return client.get("/tours").json()[0]["id"]


class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_tours_and_matrix_loaded(self, client):
        data = client.get("/health").json()
        assert data["tours_loaded"] == 170
        assert data["sim_matrix_shape"] == [170, 170]


class TestListTours:
    def test_returns_nonempty_list(self, client):
        r = client.get("/tours")
        assert r.status_code == 200
        assert len(r.json()) > 0

    def test_required_fields_present(self, client):
        tour = client.get("/tours").json()[0]
        assert {"id", "name", "city_name", "category", "hotel_stars", "price"}.issubset(tour)

    def test_filter_by_city(self, client):
        tours = client.get("/tours?city=Istanbul").json()
        assert len(tours) > 0
        assert all(t["city_name"].lower() == "istanbul" for t in tours)

    def test_filter_by_category(self, client):
        tours = client.get("/tours?category=luxury").json()
        assert all(t["category"] == "luxury" for t in tours)

    def test_filter_no_match_returns_empty_list(self, client):
        assert client.get("/tours?city=nonexistent_xyz").json() == []


class TestPredict:
    def test_status_200(self, client, any_tour_id):
        r = client.post("/predict", json={"tour_id": any_tour_id})
        assert r.status_code == 200

    def test_response_structure(self, client, any_tour_id):
        body = client.post("/predict", json={"tour_id": any_tour_id}).json()
        assert {"tour_id", "recommendations", "elapsed_ms"}.issubset(body)

    def test_tour_id_echoed(self, client, any_tour_id):
        body = client.post("/predict", json={"tour_id": any_tour_id}).json()
        assert body["tour_id"] == any_tour_id

    def test_default_top_k_is_five(self, client, any_tour_id):
        body = client.post("/predict", json={"tour_id": any_tour_id}).json()
        assert len(body["recommendations"]) == 5

    def test_custom_top_k(self, client, any_tour_id):
        body = client.post("/predict", json={"tour_id": any_tour_id, "top_k": 3}).json()
        assert len(body["recommendations"]) == 3

    def test_top_k_clamped_to_max_20(self, client, any_tour_id):
        body = client.post("/predict", json={"tour_id": any_tour_id, "top_k": 999}).json()
        assert len(body["recommendations"]) == 20

    def test_top_k_clamped_to_min_1(self, client, any_tour_id):
        body = client.post("/predict", json={"tour_id": any_tour_id, "top_k": 0}).json()
        assert len(body["recommendations"]) == 1

    def test_query_not_in_recommendations(self, client, any_tour_id):
        recs = client.post("/predict", json={"tour_id": any_tour_id}).json()["recommendations"]
        assert all(r["id"] != any_tour_id for r in recs)

    def test_similarity_descending(self, client, any_tour_id):
        recs = client.post("/predict", json={"tour_id": any_tour_id}).json()["recommendations"]
        sims = [r["similarity"] for r in recs]
        assert sims == sorted(sims, reverse=True)

    def test_recommendation_fields(self, client, any_tour_id):
        rec = client.post("/predict", json={"tour_id": any_tour_id}).json()["recommendations"][0]
        assert {"id", "name", "city_name", "category", "hotel_stars", "meal_type", "price", "similarity"}.issubset(rec)

    def test_elapsed_ms_positive(self, client, any_tour_id):
        body = client.post("/predict", json={"tour_id": any_tour_id}).json()
        assert body["elapsed_ms"] > 0

    def test_unknown_tour_id_returns_404(self, client):
        r = client.post("/predict", json={"tour_id": "00000000-0000-0000-0000-000000000000"})
        assert r.status_code == 404
