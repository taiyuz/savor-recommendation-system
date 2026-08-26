from __future__ import annotations

from fastapi.testclient import TestClient

from savor.api.app import _catalog, _recommender, create_app
from savor.data.loader import Catalog


def test_health_and_recommend_contract(catalog: Catalog) -> None:
    _catalog.cache_clear()
    _recommender.cache_clear()
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["n_users"] == catalog.users.height
    assert body["n_items"] == catalog.items.height

    user_id = catalog.user_ids()[0]
    response = client.get("/recommend", params={"user_id": user_id, "k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == user_id
    assert payload["k"] == 5
    assert payload["strategy"] in {"two_stage", "cold_start_popularity"}
    assert len(payload["items"]) == 5
    required = {"item_id", "name", "cuisine", "neighborhood", "price_tier", "score", "sources"}
    for item in payload["items"]:
        assert required <= set(item)
        assert isinstance(item["sources"], list)


def test_unknown_user_is_404() -> None:
    _catalog.cache_clear()
    _recommender.cache_clear()
    client = TestClient(create_app())
    response = client.get("/recommend", params={"user_id": "does-not-exist"})
    assert response.status_code == 404
    assert "unknown user_id" in response.json()["detail"]
