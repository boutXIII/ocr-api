def test_ocr_endpoint_mocked(client, mocker):
    # Fake predictor
    fake_result = {
        "name": "test.jpg",
        "duration": 0.1,
        "orientation": {"value": 0, "confidence": None},
        "language": {"value": "fr", "confidence": 0.99},
        "dimensions": [100, 100],
        "items": []
    }

    mock_predictor = mocker.Mock(return_value=[fake_result])

    mocker.patch(
        "api.routes.ocr.init_predictor",
        return_value=mock_predictor
    )

    response = client.post(
        "/ocr",
        files={"file": ("test.jpg", b"fake", "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "test.jpg"
    assert data["duration"] >= 0
