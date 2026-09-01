"""FastAPI 엔드포인트 테스트. LLM 호출 없이(GOOGLE_API_KEY 없이) 검증 가능한 부분만 다룬다."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_molecule_properties_aspirin():
    resp = client.post(
        "/molecule/properties", json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 179 <= body["molecular_weight"] <= 181
    assert body["passes_lipinski_ro5"] is True


def test_molecule_properties_invalid_smiles():
    resp = client.post("/molecule/properties", json={"smiles": "not-valid!!"})
    assert resp.status_code == 422


def test_agent_query_without_api_key_returns_500(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resp = client.post("/agent/query", json={"question": "hello"})
    assert resp.status_code == 500
    assert "GOOGLE_API_KEY" in resp.json()["detail"]
