def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "duration" in data
    assert isinstance(data["duration"], float)
    assert data["duration"] >= 0

def test_health_models(client):
    response = client.get("/health/models")

    assert response.status_code == 200
    data = response.json()

    assert "models_available" in data
    assert "detection" in data["models_available"]
