def test_duration_present(client):
    response = client.get("/health")
    data = response.json()

    assert isinstance(data["duration"], float)
    assert round(data["duration"], 4) == data["duration"]
